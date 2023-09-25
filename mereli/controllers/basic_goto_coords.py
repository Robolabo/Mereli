import numpy as np
from mereli.controllers import RobotController
from mereli.register import controller_registry
from mereli.utils import compute_angle

@controller_registry(name='basic_goto_coords') 
class BasicGOTOCoords(RobotController):
    """ Controller devoted to the obstacle avoidance task. This means that the controller 
    will read from the distance sensor, process the measurements and return joint velocity 
    actions required to avoid colliding with any other tangible entity. This controller is 
    currently hard coded for the Epuck robot.

    :param float sensitivity: value in [0, 1] that defines the threshold in the distance sensor reading 
        to interpret an obstacle detection. 
    """
    def __init__(self, *args, sensitivity=0.1, no_obstacle_action=[1.,1.],  **kwargs):
        super(BasicGOTOCoords, self).__init__(*args, **kwargs)
        self.sensitivity = 0.5
    
    def step_obstacle_avoid(self, state):
        st_ds = state['distance_sensor']
        obstacle = False
        if any(st_ds[[0,1]] > self.sensitivity):
            obstacle = True 
            # print('Turn Left')
            action = np.array([1., 0])
        elif any(st_ds[[6,7]] > self.sensitivity):
            obstacle = True 
            # print('Turn Right')
            action = np.array([0, 1.])
        else:
            # print('GO straight over')
            action = np.array([0,0])
        return {'joint_velocity_actuator' : action}, obstacle
    
    def select_coords_id(self):
        robnum = self.controller_owner.id - 5 
        # if robnum < 5: 
        if robnum == 0: 
            self.target_coords = np.array([2,0]) 
        elif robnum == 1: 
            self.target_coords = np.array([-2, 0])
        elif robnum == 2: 
            self.target_coords = np.array([0, 2])
        # elif robnum < 20: 
        elif robnum == 3: 
            self.target_coords = np.array([0, -2])
        else:
            self.target_coords = np.array([0, -2])

    def select_coords_lmark(self):
        lmark = self.controller_owner.virtual_particle.lmark
        state = self.controller_owner.virtual_particle.state

        # if state[0] > 0:
        #      if state[1] > 0:
        #         self.target_coords = np.array([-1, 0]) 
        #      else:
        #         self.target_coords = np.array([1, 0]) 

        # else:
        #      if state[1] > 0:
        #         self.target_coords = np.array([0, -1]) 
        #      else:
        #         self.target_coords = np.array([0, 1]) 

        # return

        if lmark is None:
            self.target_coords =  np.array([0, 0]) 

        # if lmark == 0: 
        #     self.target_coords = np.array([-1, 0]) 
        # elif lmark in [1,2]: 
        #     self.target_coords = np.array([1, 0]) 
        # elif lmark in [3, 4, 5]: 
        #     self.target_coords = np.array([0, 1]) 
        # elif lmark in [6, 7, 8, 9]: 
        #     self.target_coords = np.array([0, -1]) 
        # else:
        #     self.target_coords = np.array([0, 0]) 
        # return
       
        n_robs = 30 
        all_lmarks = np.arange(n_robs)
        # np.random.shuffle(all_lmarks)
        gr1 = all_lmarks[:n_robs//3]
        gr2 = all_lmarks[n_robs//3:2*(n_robs//3)]
        gr3 = all_lmarks[2*(n_robs//3):]
        if lmark in gr1: 
            self.target_coords = np.array([-1, 0]) 
        elif lmark in gr2: 
            self.target_coords = np.array([1, 0]) 
        elif lmark in gr3: 
            self.target_coords = np.array([0, 1]) 
        # elif lmark in gr4: 
        #     self.target_coords = np.array([0, -1]) 
        else:
            self.target_coords = np.array([0, 0]) 

        # if lmark is not None and lmark < 5: 
        #     self.target_coords = np.array([-1, 0]) 
        # elif lmark is not None and lmark < 10: 
        #     self.target_coords = np.array([0, 1]) 
        # elif lmark is not None and lmark < 15: 
        #     self.target_coords = np.array([1, 0]) 
        # else:
        #     print("I shouldn't be here!")
        #     self.target_coords = np.array([0, 0]) 
        
        # if lmark == 0: 
        #     self.target_coords = np.array([1,0]) 
        # elif lmark == 1: 
        #     self.target_coords = np.array([-1, 0])
        # elif lmark == 2: 
        #     self.target_coords = np.array([0, 1])
        # elif lmark == 3: 
        #     self.target_coords = np.array([0, -1])
        # elif lmark == 4:
        #     self.target_coords = np.array([1, 1])
        # elif lmark == 5:
        #     self.target_coords = np.array([-1, -1])
        # elif lmark == 6:
        #     self.target_coords = np.array([1, -1])
        # elif lmark == 7:
        #     self.target_coords = np.array([-1, 1])
        # elif lmark == 8:
        #     self.target_coords = np.array([1, 1])
        # else:
        #     self.target_coords = np.array([0, -1])


    def step(self, state, reward=0.0):
        # __import__('pdb').set_trace()
        area_read = state['ground_sensor']
        curr_pos = state['own_position_sensor'][:2]
        self.select_coords_lmark()
        # if area_read > 0 and np.linalg.norm(curr_pos - self.target_coords) < 0.4:
        #     return {'joint_velocity_actuator' : np.array([0, 0])}
        action_obsav, is_obstacle = self.step_obstacle_avoid(state)
        if False:#is_obstacle: 
            return action_obsav
        else:
            desired_dir = (self.target_coords - curr_pos) / np.linalg.norm(self.target_coords - curr_pos)
            robot_ori = self.controller_owner.orientation[-1]
            heading_ori = np.r_[np.cos(robot_ori), np.sin(robot_ori)]
            a1 = compute_angle(desired_dir)
            a2 = compute_angle(heading_ori)
            if np.abs(a1 - a2) <= 0.4:
                action = np.array([1, 1])
            elif a1 > a2:
                action = np.array([1., -1])/4
            else:
                action = np.array([-1., 1])/4
            return {'joint_velocity_actuator' : action}
