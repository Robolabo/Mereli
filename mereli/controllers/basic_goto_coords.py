import numpy as np
from mereli.controllers import RobotController
from mereli.register import controller_registry
from mereli.utils import compute_angle, angle_diff

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
        self.flag = False

    def select_coords_lmark_formation(self):
        # import matplotlib.pyplot as plt
        lmark = self.controller_owner.virtual_particle.lmark
        if lmark is None:
            self.target_coords = np.zeros(2)
            return
        state = self.controller_owner.virtual_particle.state
        # form_triangle = {'nodes' : .5 * np.array([[-1, 0],[1,0],[0,1]]), 'edges' : [(0)]}
        formationA = {'nodes' : np.array([[0, .5], [-.25, 0], [.25,0], [-.5, -.5], [.5, -.5], [0,-.5]]),
                      'edges' : [[1,2], [0,2,3,5], [0,2,4,5], [1,5], [2,5], [1,2,3,4]]}
        formationB = {'nodes' : np.array([[0, .25], [-.25, 0], [.25,0], [-.5, -.5], [.5, -.5], [-1,-1], [-0.25, -1],[.25,-1],[1,-1], [0,-0.5] ]), 
                      'edges' : [[0,1,2], [1,0,2,3,9], [2,0,1,4,9], [3,1,5,6,9], [4,2,7,8,9], [5,3,6], [6,3,5,7,9], [7,4,6,8], [8,4,7], [9,1,2,3,4,6,7]]}
        formationC = {'nodes' : 0.5*np.array([[0, 0], [1, 0], [0.71,0.71], [0, 1], [-0.71, 0.71], [-1,0], [-0.71, -0.71], [0,-1], [0.71, -0.71]]),
                      'edges' : [[0, 1,2,3,4,5,6,7,8], [1,0,8,2], [2,0,1,3], [3,0,2,4], [4,0, 3,5], [5,0,4,6],[6,0,5,7], [7,0,6,8], [8,0,7,1]]}
        self.formation = formationC

        # plt.scatter(formationB[:,0], formationB[:,1])
        # plt.show()
        # __import__('pdb').set_trace()
        neigh_positions = [epk.position[:2] for epk in self.controller_owner.neighbors]
        center =np.mean(neigh_positions,0)
        try:
            center = np.mean([self.formation['nodes'][x] for x in self.formation['edges'][lmark]],0)
        except:
            __import__('pdb').set_trace()
        self.target_coords = self.formation['nodes'][lmark] if lmark is not None else np.zeros(2)
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
        # if self.t == 1500:
        #     self.controller_owner.virtual_particle.disabled_lmarks.append(self.controller_owner.virtual_particle.lmark)

        # area_read = state['ground_sensor']
        curr_pos = self.get_sensor_reading('own_position_sensor')[:2]
        self.select_coords_lmark_formation()
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
        angle = angle_diff(a1,a2)
        # if self.target_coords.sum() != 0: __import__('pdb').set_trace()
        if angle <= 0.5:
            action = A * np.array([1, 1])
        elif angle > np.pi / 2:
            action = np.array([-1,-1])
        elif a1 > a2:
            action =  np.array([1., -1])
        else:
            action =  np.array([-1., 1])
        
        # if np.linalg.norm(self.target_coords - curr_pos) < 0.1:
        #     action = np.array([0,0])
        self.flag = True
        self.get_actuator('joint_velocity_actuator').action = action / 3

    def reset(self):
        self.flag = False
        self.controller_owner.virtual_particle.lmark_priorities = np.ones(10) 
