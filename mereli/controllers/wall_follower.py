import numpy as np
from mereli.controllers import RobotController
from mereli.register import controller_registry

@controller_registry(name='wall_follower')
class WallFollowerController(RobotController):
    def __init__(self, *args, sensitivity=0.6, no_obstacle_action=[1.,1.],  **kwargs):
        super(WallFollowerController, self).__init__(*args, **kwargs)
        self.sensitivity = sensitivity

    def step(self, state, reward=0.0):
        st_ds = state['distance_sensor']
        obstacle_right = np.sum(st_ds[[1,2]]) > self.sensitivity        
        obstacle_front = np.sum(st_ds[[0,7]]) > self.sensitivity        
        obstacle_left = np.sum(st_ds[[6,5]]) > self.sensitivity        
        # if obstacle_left and obstacle_front and obstacle_right: 
        #     return {'joint_velocity_actuator' : np.array([1,-1])}
        # if not (obstacle_left and obstacle_front and obstacle_right): 
        #     return {'joint_velocity_actuator' : np.array([0.5,0.5])}
        
        if not obstacle_right:
            action = np.array([-0.5, 0.5])
        else:
            if not obstacle_front:
                #move forward
                action = np.array([0.7,0.7])
            else:
                # rotate counter-clockwise
                action = np.array([0.5, -0.5])
        return {'joint_velocity_actuator' : action}
