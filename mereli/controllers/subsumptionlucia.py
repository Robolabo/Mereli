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
    def __init__(self, *args,  wait_full_load=True, **kwargs):
        super(LoadBatteryController, self).__init__(*args, **kwargs)
        self.flag = False
        self.bat_threshold = 0.5
        self.wait_full_load = True
        self.charging = False

    def step(self, state, reward=0):
        bat_lv_array = self.get_sensor_reading('battery_sensor')
        bat_lv = bat_lv_array[0] #extrae la roja
        action = np.array([0,0])
        self.flag = False 
        if self.wait_full_load and bat_lv >= 0.95:
            self.charging = False
        if bat_lv <= self.bat_threshold or self.charging and bat_lv < 0.95:
            self.charging = self.wait_full_load 
            ls_read = self.get_sensor_reading('red_light_sensor')
            if np.max(ls_read) > 0.9:
                action = np.zeros(2)
                self.flag = True
            elif ls_read[0] * ls_read[7] == 0:
                self.flag = True 
                light_left = np.sum(ls_read[[7,6,5,4]])
                light_right = np.sum(ls_read[[0,1,2,3]])
                if light_right > light_left:
                    action = 0.2*np.array([-1, 1]) 
                else: 
                    action = 0.2*np.array([1, -1]) 
        self.get_actuator('joint_velocity_actuator').action = action


@controller_registry(name='subsumptionlucia')
class SubsumptionLuciaController(RobotController):
    """
    """
    def __init__(self, *args, routines={'navigate' : 0}, **kwargs):
        super(SubsumptionLuciaController, self).__init__(*args, **kwargs)
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
                break

    def reset(self):
        for rt in self.routines.values():
            rt.controller_owner = self.controller_owner
            rt.reset()

@controller_registry(name="simple_forage")
class SimpleForageController(RobotController):
    def __init__(self, *args,  wait_full_load=True, **kwargs):
        super(SimpleForageController, self).__init__(*args, **kwargs)
        self.flag = False

    def step(self, state, reward=0):
        mgs_read = self.get_sensor_reading('memory_ground_sensor')
        self.flag = False 
        if mgs_read == 1: # Garbage collected
            ls_read = self.get_sensor_reading('red_light_sensor')
            if ls_read[0] * ls_read[7] == 0:
                self.flag = True
                light_left= np.sum(ls_read[[7,6,5,4]])
                light_right = np.sum(ls_read[[0,1,2,3]])
                action = np.array([0., 0.])
                if light_right > light_left:
                    action = .1*np.array([-1, 1]) 
                else: 
                    action = .1*np.array([1, -1]) 
                self.get_actuator('joint_velocity_actuator').action = action


        
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



