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
        self.sensitivity = 0.7
        self.t = 0
        self.flag = False
    

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

    def select_coords_lmark_formation(self):
        lmark = self.controller_owner.virtual_particle.lmark
        state = self.controller_owner.virtual_particle.state
        formationA = np.array([[0, .5], [-.25, 0], [.25,0], [-.5, -.5], [.5, -.5], [0,-.5]])
        formationB = np.array([[0, 0.75], [-.25, 0], [.25,0], [-.5, 0], [.5, 0], [0,-.75]])
        xx = np.linspace(0, 2* np.pi, 9)
        formationD = .7 * np.array([np.r_[np.cos(x), np.sin(x)] for x in xx])

        center =  np.mean([epk.position for epk in self.controller_owner.neighbors], 0)[:2]

        self.target_coords = formationA[lmark] if lmark is not None else np.zeros(2)
        self.target_coords += center
        return 

    def select_coords_lmark_groups(self):
        lmark = self.controller_owner.virtual_particle.lmark
        state = self.controller_owner.virtual_particle.state

        if lmark is None:
            self.target_coords =  np.array([0, 0]) 

#         if lmark == 0: 
#             self.target_coords = np.array([-1, 0]) 
#         elif lmark in [1,2]: 
#             self.target_coords = np.array([1, 0]) 
#         elif lmark in [3, 4, 5]: 
#             self.target_coords = np.array([0, 1]) 
#         elif lmark in [6, 7, 8, 9]: 
#             self.target_coords = np.array([0, -1]) 
#         else:
#             self.target_coords = np.array([0, 0]) 
#         return
       
       #  if lmark == 0: 
       #      self.target_coords = np.array([-.5, 0]) 
       #  elif lmark == 1: 
       #      self.target_coords = np.array([.5, 0]) 
       #  elif lmark == 2: 
       #      self.target_coords = np.array([0, .5]) 
       #  elif lmark == 3: 
       #      self.target_coords = np.array([0, -.5]) 
       #  else:
       #      self.target_coords = np.array([0, 0]) 
       #  # self.target_coords *= 0.5
       #  return

        n_robs = 11 
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


    def step(self, state, reward=0.0):
        self.t += 1
        # if self.t == 1500:
        #     self.controller_owner.virtual_particle.disabled_lmarks.append(self.controller_owner.virtual_particle.lmark)

        area_read = state['ground_sensor']
        curr_pos = state['own_position_sensor'][:2]
        self.select_coords_lmark_groups()
        # if area_read > 0 and np.linalg.norm(curr_pos - self.target_coords) < 0.4:
        #     return {'joint_velocity_actuator' : np.array([0, 0])}



        dist_tar = np.linalg.norm(self.target_coords - curr_pos)
        desired_dir = (self.target_coords - curr_pos) / dist_tar
        # desired_dir -= v_obstacle


        robot_ori = self.controller_owner.orientation[-1]
        heading_ori = np.r_[np.cos(robot_ori), np.sin(robot_ori)]
        a1 = compute_angle(desired_dir)
        a2 = compute_angle(heading_ori)
        A = 1 / (1 + np.exp(-20 * (dist_tar - .1)))
        B = np.cos(a1 - a2)


        if np.abs(a1 - a2) <= 0.5:
            action = A * np.array([1, 1])
        elif a1 > a2:
            action =  np.array([1., 0])
        else:
            action =  np.array([-1., 0])
        
        # if np.linalg.norm(self.target_coords - curr_pos) < 0.1:
        #     action = np.array([0,0])
        self.flag = True
        return {'joint_velocity_actuator' : action/3}

    def reset(self):
        self.t = 0 
        self.flag = False
