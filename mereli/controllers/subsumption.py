import numpy as np
from mereli.controllers import RobotController
from mereli.register import controller_registry, controllers
from mereli.utils import compute_angle


@controller_registry(name="navigate")
class NavigateController(RobotController):
    def __init__(self, *args,  **kwargs):
        super(NavigateController, self).__init__(*args, **kwargs)
        self.flag = True 

    def step(self, state, reward=0):
        self.get_actuator('joint_velocity_actuator').action = np.ones(2) 


@controller_registry(name="load_battery")
class LoadBatteryController(RobotController):

    def __init__(self, *args,  **kwargs):
        super(LoadBatteryController, self).__init__(*args, **kwargs)
        self.flag = False
        self.bat_threshold = 0.5

    def step(self, state, reward=0):
        bat_lv = self.get_sensor_reading('battery_sensor')
        action = np.array([0,0])
        self.flag = False 
        # self.flags['load'] = False 
        if bat_lv <= self.bat_threshold:
            ls_read = self.get_sensor_reading('red_light_sensor')
            if ls_read[0] * ls_read[7] == 0:
                self.flag = True 
                light_left = np.sum(ls_read[[7,6,5,4]])
                light_right = np.sum(ls_read[[0,1,2,3]])
                if light_right > light_left:
                    action = np.array([-1, 1]) 
                else: 
                    action = np.array([1, -1]) 
        self.get_actuator('joint_velocity_actuator').action = action


@controller_registry(name='subsumption')
class SubsumptionController(RobotController):
    """
    """
    def __init__(self, *args, routines={'navigate' : 0}, **kwargs):
        super(SubsumptionController, self).__init__(*args, **kwargs)
        self.routines = {}
        self.priorities = {}
        self.activations = {}
        for rt, pr in routines.items(): 
            priority = pr if isinstance(pr, int) else pr['priority']
            rt_params = pr.get('params', {}) if isinstance(pr, dict) else {}
            self.priorities[rt] = priority
            self.routines[rt] =  controllers[rt](**rt_params)
            self.activations[rt] = np.zeros(2)

    def step(self, state, reward=0.0):
        for k, routine in self.routines.items():
            action = routine.step(state)
            self.activations[k] = self.get_actuator('joint_velocity_actuator').action
        return self.coordinate()

    def coordinate(self):
        names = [*self.priorities.keys()]
        names.sort(key=self.priorities.get)
        for k in names:   
            if self.routines[k].flag:
                action_wheels = self.activations[k]
                self.get_actuator('joint_velocity_actuator').action = np.array(action_wheels)

    def reset(self):
        for rt in self.routines.values():
            rt.controller_owner = self.controller_owner
            rt.reset()


@controller_registry(name='subsumption_garbage')
class SubsumptionGarbageController(RobotController):
    """
    """
    def __init__(self, *args,  **kwargs):
        super(SubsumptionGarbageController, self).__init__(*args, **kwargs)
        self.routines = ['forage', 'nav', 'load', 'avoid']
        self.activations = {'forage' : np.zeros(2), 'nav' : np.array([1., 1.]), 'load' : np.zeros(2), 'avoid' : np.zeros(2)} 
        self.flags = {k : False for k in self.routines} 
        self.bat_threshold = .5

    def step(self, state, reward=0.0):
        self.avoid_obstacles(state)
        self.load_battery(state)
        self.navigate(state)
        self.forage(state)
        return self.coordinate()


    def coordinate(self):
        if self.flags['avoid']:
            action_wheels = self.activations['avoid']
        elif self.flags['load']:
            action_wheels = self.activations['load']
        elif self.flags['forage']:
            action_wheels = self.activations['forage']
        else: 
            action_wheels = self.activations['nav']
        return {'joint_velocity_actuator' : np.array(action_wheels)}

    def navigate(self, state):
        pass

    def forage(self, state):
        mgs_read = state['memory_ground_sensor']
        self.flags['forage'] = False 
        if mgs_read == 1: # Garbage collected
            ls_read = state['red_light_sensor']
            if ls_read[0] * ls_read[7] == 0:
                self.flags['forage'] = True
                light_left= np.sum(ls_read[[7,6,5,4]])
                light_right = np.sum(ls_read[[0,1,2,3]])
                if light_right > light_left:
                    self.activations['forage'] = np.array([-1, 1]) 
                else: 
                    self.activations['forage'] = np.array([1, -1]) 


    def load_battery(self, state):
        bat_lv = state['battery_sensor']
        self.flags['load'] = False 
        if bat_lv <= self.bat_threshold:
            ls_read = state['blue_light_sensor']
            if ls_read[0] * ls_read[7] == 0:
                self.flags['load'] = True
                light_left= np.sum(ls_read[[7,6,5,4]])
                light_right = np.sum(ls_read[[0,1,2,3]])
                if light_right > light_left:
                    self.activations['load'] = np.array([-1, 1]) 
                else: 
                    self.activations['load'] = np.array([1, -1]) 

    def avoid_obstacles(self, state):
        st_ds = state['distance_sensor']
        self.flags['avoid'] = False
        prox_thresh = 0.5
        if any(st_ds[[0,1]] > prox_thresh):
            # print('Turn Left')
            self.activations['avoid'] = np.array([1, -1])
            self.flags['avoid'] = True 
        elif any(st_ds[[6,7]] > prox_thresh):
            # print('Turn Right')
            self.activations['avoid'] = np.array([-1, 1])
            self.flags['avoid'] = True 

    def avoid_obstacles2(self, state):
        prox_read = state['distance_sensor']
        oris = self.controller_owner.sensors['distance_sensor'].directions(self.controller_owner.orientation[-1])
        max_prox = np.max(prox_read) 
        if max_prox > 0.3:
            v_repel = np.sum([prox_read[i] * np.r_[np.cos(oris[i]), np.sin(oris[i])] for i in range(8)], 0)
            f_repel = np.arctan2(v_repel[0], v_repel[0]) - np.pi
            while (f_repel > np.pi):
                f_repel -= 2*np.pi
            while (f_repel < -np.pi):
                f_repel += 2*np.pi
            fc1 = 1 / np.pi
            fc_linear = 1 
            fc_angular = 1
            fv_linear = fc_linear * np.cos(f_repel / 2)
            fv_angular = f_repel 
            self.activations['avoid'] = np.array([fv_linear + fc1 * fv_angular, fv_linear - fc1 * fv_angular])
            self.flags['avoid'] = True
        else:
            self.flags['avoid'] = False 
