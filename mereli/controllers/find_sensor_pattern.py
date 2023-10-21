
import numpy as np
from mereli.controllers import RobotController
from mereli.register import controller_registry
from mereli.utils import compute_angle

@controller_registry(name='find_sensor_pattern') 
class FindSensorPattern(RobotController):
    def __init__(self, *args, **kwargs):
        super(FindSensorPattern, self).__init__(*args, **kwargs)
        self.t = 0
        self.flag = False
        # self.patterns = np.array([
        #         [0,0,0,1,1,0,0,0], 
        #         [1,0,1,0,0,0,0,0], 
        #         [0,0,0,0,0,1,0,1] 
        # ])
        self.patterns = np.array([
                [0,0,1,0,0,0,0,0], 
                [0,0,0,0,0,1,0,0] 
        ])
        self.pattern = None
    

    def select_coords_id(self):
        robnum = self.controller_owner.id - 5 
        self.pattern = self.patterns[robnum] 

    def step(self, state, reward=0.0):
        self.t += 1
        self.flag = True
        self.select_coords_id()
        st_ds = state['distance_sensor']
        mask = st_ds > 0.3
        max_i = np.argmax(st_ds)
        tar_max_i = np.argmax(self.pattern)
        if max_i == tar_max_i:
            return {'joint_velocity_actuator' : np.array([0, 0])}
        

        if max_i > tar_max_i:
            return {'joint_velocity_actuator' : np.array([-.2, .2])}
        else:
            return {'joint_velocity_actuator' : np.array([.2, -.2])}

        


#         dist_tar = np.linalg.norm(self.target_coords - curr_pos)
#         desired_dir = (self.target_coords - curr_pos) / dist_tar
#         # desired_dir -= v_obstacle


#         robot_ori = self.controller_owner.orientation[-1]
#         heading_ori = np.r_[np.cos(robot_ori), np.sin(robot_ori)]
#         a1 = compute_angle(desired_dir)
#         a2 = compute_angle(heading_ori)
#         A = 1 / (1 + np.exp(-20 * (dist_tar - .1)))
#         B = np.cos(a1 - a2)


#         if np.abs(a1 - a2) <= 0.5:
#             action = A * np.array([1, 1])
#         elif a1 > a2:
#             action =  np.array([1., 0])
#         else:
#             action =  np.array([-1., 0])
        
        # if np.linalg.norm(self.target_coords - curr_pos) < 0.1:
        #     action = np.array([0,0])

        return {'joint_velocity_actuator' : action/3}

    def reset(self):
        self.t = 0 
        self.flag = False
