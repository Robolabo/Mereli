import numpy as np
from mereli.controllers import RobotController
from mereli.register import controller_registry
from mereli.utils import compute_angle, angle_diff

@controller_registry(name='motor_schemas2')
class MotorSchemas2Controller(RobotController):
    """
    """
    def __init__(self, *args,  **kwargs):
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
