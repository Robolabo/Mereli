import numpy as np
from mereli.controllers import RobotController
from mereli.register import controller_registry, controllers
from mereli.utils import compute_angle, angle_diff


@controller_registry(name='motor_schemas')
class MotorSchemasController(RobotController):
    """
    """
    def __init__(self, *args, routines={'navigate' : 1}, **kwargs):
        super(MotorSchemasController, self).__init__(*args, **kwargs)
        self.routines = {}
        self.activations = {}
        self.weights = {}
        self.forces = {}
        for rt, w in routines.items(): 
            rt_params = w.get('params', {}) if isinstance(w, dict) else {}
            weight = w if isinstance(w, float) else w['weight']
            self.weights[rt] = weight 
            self.routines[rt] =  controllers[rt](**rt_params)
            self.activations[rt] = np.zeros(2)
            self.forces[rt] = np.zeros(2).astype(float)

    def step(self, state, reward=0.0):
        for k, routine in self.routines.items():
            action = routine.step(state)
            self.activations[k] = self.get_actuator('joint_velocity_actuator').action
        return self.coordinate()

    def coordinate(self):
        v_force = np.array([0., 0.])
        for k, rt in self.routines.items():
            if rt.flag:
                weight = self.weights[k] 
                if hasattr(rt, 'vforce'):
                    vf = rt.vforce
                else:
                    dPos, dTh = self.compute_force(4*self.activations[k])
                    ori = self.robot.orientation[-1]
                    vf = dPos +  np.r_[np.cos(ori + dTh), np.sin(ori + dTh)]
                self.forces[k] = weight * vf
                v_force += weight * vf 
        fmod = np.linalg.norm(v_force)
        v_force = v_force /fmod 
        v_head = np.r_[np.cos(self.controller_owner.orientation[-1]), np.sin(self.controller_owner.orientation[-1])]
        a1 = compute_angle(v_force)
        a2 = compute_angle(v_head)
        angle = angle_diff(a1,a2)
        if angle <= 0.1:
            action = 0.5*np.array([1, 1])
        elif a1 > a2:
            if a1 - a2 > np.pi:
                action =  .1*np.array([-1., 1])
            else:
                action =  .1* np.array([1., -1])
        else:
            if a2-a1 >np.pi:
                action =  .1* np.array([1., -1])
            else:
                action =  .1*np.array([-1., 1])

        self.get_actuator('joint_velocity_actuator').action = 0.4*np.array(action)

    def compute_force(self, action):
        b = 0.12 # Dist between wheels
        R = 0.0390625 # Wheel radius
        dt = 10 * self.robot.physics_client.dt
        dUl = R * action[1] * dt
        dUr = R * action[0] * dt
        w = (dUr - dUl) / b
        V = (dUr + dUl) / 2
        dx = np.cos(self.robot.orientation[-1]) * V
        dy = np.sin(self.robot.orientation[-1]) * V
        dTh = w 
        # Update new estimates
        deltaPos = np.r_[dx, dy]
        print(dTh)
        return deltaPos,dTh 

    def reset(self):
        for rt in self.routines.values():
            rt.controller_owner = self.controller_owner
            rt.reset()



@controller_registry(name='motor_schemas2')
class MotorSchemas2Controller(RobotController):
    """
    """
    def __init__(self, *args, routines={'navigate' : 1}, **kwargs):
        super(MotorSchemas2Controller, self).__init__(*args, **kwargs)
        self.routines = ['nav', 'load', 'avoid']
        self.activations = {'nav' : np.array([1., 1.]), 'load' : np.zeros(2), 'avoid' : np.zeros(2)} 
        self.flags = {k : False for k in self.routines} 
        self.bat_threshold = .5

    def step(self, state, reward=0.0):
        self.avoid_obstacles(state)
        self.load_battery(state)
        # self.navigate(state)
        return self.coordinate()


    def coordinate(self):
        v_force = np.array([0., 0.])
        for pr in self.routines:
            if self.flags[pr]:
                v_force += self.activations[pr][1] * np.r_[np.cos(self.activations[pr][0]), np.sin(self.activations[pr][0])] 
        
        v_head = np.r_[np.cos(self.controller_owner.orientation[-1]), np.sin(self.controller_owner.orientation[-1])]
        if v_force.sum() == 0.: v_force = v_head 
        a1 = np.arctan2(v_head[1], v_head[0])
        a2 = np.arctan2(v_force[1], v_force[0])
        A = 0.1
        if angle_diff(a1, a2) <= 0.1:
            action = A * np.array([1, 1])
        elif a1 > a2:
            action =  A * np.array([1., -1])
        else:
            action = A* np.array([-1., 1])

        return {'joint_velocity_actuator' : action} 

    def navigate(self, state):
       self.activations['nav'] = np.array([self.controller_owner.orientation[-1], 0.01])
       self.flags['nav'] = True

    def load_battery(self, state):
        bat_lv = state['battery_sensor']
        self.flags['load'] = False 
        if bat_lv <= self.bat_threshold:
            ls_read = state['red_light_sensor']
            if ls_read[0] * ls_read[7] == 0:
                self.flags['load'] = True
                light_left= np.sum(ls_read[[7,6,5,4]])
                light_right = np.sum(ls_read[[0,1,2,3]])
                if light_right > light_left:
                    self.activations['load'] = np.array([-1, 1]) 
                else: 
                    self.activations['load'] = np.array([1, -1])

    def avoid_obstacles(self, state):
        prox_read = state['distance_sensor']
        oris = self.controller_owner.sensors['distance_sensor'].directions(self.controller_owner.orientation[-1])
        max_prox = np.max(prox_read) 
        if max_prox > 0.3:
            v_repel = np.sum([prox_read[i] * np.r_[np.cos(oris[i]), np.sin(oris[i])] for i in range(8)], 0)
            f_repel = np.arctan2(v_repel[1], v_repel[0]) - np.pi
            while (f_repel > np.pi):
                f_repel -= 2*np.pi
            while (f_repel < -np.pi):
                f_repel += 2*np.pi
            self.activations['avoid'] = np.array([f_repel, 1.])
            self.flags['avoid'] = True
        else:
            self.flags['avoid'] = False 
