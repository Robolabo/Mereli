
import numpy as np
from mereli.controllers import RobotController, BasicObstacleAvoider
from mereli.register import controller_registry

@controller_registry(name='phototaxis')
class Phototaxis(RobotController):
    def __init__(self, *args, color='red', **kwargs):
        super(Phototaxis, self).__init__(*args, **kwargs)
        self.color = color
        self.ls_sensor = self.color + '_light_sensor' 
        self.obstacle_avoider = BasicObstacleAvoider(no_obstacle_action=[0,0]) 
        
    def step(self, state, reward=0.0):
        action_photo = np.zeros(2)
        action_obsav = self.obstacle_avoider.step(state, reward=reward)
        action_photo[0] = np.max(state[self.ls_sensor][[4, 5, 6, 7]])
        action_photo[1] = np.max(state[self.ls_sensor][[0, 1, 2, 3]])
        if np.max(state[self.ls_sensor]) == 0.0: # 
            action_photo = np.array([.2, .2])
        action = action_photo if np.sum(action_obsav['joint_velocity_actuator']) == 0 else action_obsav
        return {'joint_velocity_actuator' : 2*np.clip(action, a_min=-1, a_max=1)}
