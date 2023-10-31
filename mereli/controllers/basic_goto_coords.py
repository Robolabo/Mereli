import numpy as np
from mereli.controllers import RobotController
from mereli.register import controller_registry
from mereli.utils import compute_angle, angle_diff


def get_formation_neighbors(formation,node_id, max_dist=0.7):
    neighs = [node_id]
    node_pos = formation['nodes'][node_id] 
    for i in range(len(formation['nodes'])):
        if i == node_id:
            continue
        n2 = formation['nodes'][i]
        if np.linalg.norm(node_pos - n2) < max_dist:
            neighs.append(i)
    return neighs



def get_formation(formation_name):
    return { 
        'formationA' : {'nodes' : np.array([[0, .5], [-.25, 0], [.25,0], [-.5, -.5], [.5, -.5], [0,-.5]]),
                      'edges' : [[1,2], [0,2,3,5], [0,2,4,5], [1,5], [2,5], [1,2,3,4]]},
        'formationB' : {'leader' : 0, 'nodes' : 0.75*np.array([[0, .25], [-.25, 0], [.25,0], [-.5, -.5], 
                                                             [.5, -.5], [-1,-1], [-0.25, -1],[.25,-1],[1,-1], [0,-0.5] ]), 
                      'edges' : [[0,1,2], [1,0,2,3,9], [2,0,1,4,9], [3,1,5,6,9], [4,2,7,8,9], [5,3,6], [6,3,5,7,9], [7,4,6,8], [8,4,7], [9,1,2,3,4,6,7]]},
        'formationC' : {'leader' : 3, 'nodes' : 0.5*np.array([[0, 0], [1, 0], [0.71,0.71], [0, 1], 
                                                            [-0.71, 0.71], [-1,0], [-0.71, -0.71], [0,-1], [0.71, -0.71]]),
                      'edges' : [[0, 1,2,3,4,5,6,7,8], [1,0,8,2], [2,0,1,3], [3,0,2,4], [4,0, 3,5], [5,0,4,6],[6,0,5,7], [7,0,6,8], [8,0,7,1]]},
        'formationD' : {'leader' : 1, 'nodes' : 0.5*np.array([[-1, 1], [0, 1], [1,1], [-1,0], [0, 0], [1,0], [-1, -1], [0,-1], [1, -1]]),
                      'edges' : [[0,1,3], [1,0,2,4], [2,1,5], [3,0,4,6], [4,1,3,5,7], [5,2,4,8],[6,3,7], [7,4,6,8], [8,5,7]]},
        'formationE' : {'leader' : -1, 'nodes' : np.array([[0, 0], [-.25, -0.5], [.25,-0.5], [-.5, -1], [.5, -1], [0,-1],
                                                            [-.25, 0.5], [.25,0.5], [-.5,1], [.5,1], [0,1]])},
        'formationF' : {'leader' : -1, 'nodes' : np.array([[-2.5,0],[-2,0],[-1.5,0],[-1,0],[-0.5,0], [0,0], [.5,0],[1,0],[1.5,0], [2,0], [2.5,0]])},
            'formation_circle_25' : {'nodes' : 
		0.5 * np.array([[ 0,    0  ],
		 [ 1,    0  ],
		 [ 0.62, 0.78],
		 [-0.22,  0.97],
		 [-0.9,   0.43],
		 [-0.9,  -0.43],
		 [-0.22, -0.97],
		 [ 0.62, -0.78],
		 [ 1,    0  ],
		 [ 2,     0  ],
		 [ 1.83,  0.81],
		 [ 1.34,  1.49],
		 [ 0.62,  1.9 ],
		 [-0.21,  1.99],
		 [-1 ,   1.73],
		 [-1.62,  1.18],
		 [-1.96,  0.42],
		 [-1.96, -0.42],
		 [-1.62, -1.18],
		 [-1,   -1.73],
		 [-0.21, -1.99],
	 [ 0.62, -1.9 ],
	 [ 1.34, -1.49],
	 [ 1.83, -0.81],
		 [ 2,   0  ]])}
        }.get(formation_name)
    


@controller_registry(name='basic_goto_coords') 
class BasicGOTOCoords(RobotController):
    """ Controller devoted to the obstacle avoidance task. This means that the controller 
    will read from the distance sensor, process the measurements and return joint velocity 
    actions required to avoid colliding with any other tangible entity. This controller is 
    currently hard coded for the Epuck robot.

    :param float sensitivity: value in [0, 1] that defines the threshold in the distance sensor reading 
        to interpret an obstacle detection. 
    """
    def __init__(self, *args, formation='formationC', sensitivity=0.1, no_obstacle_action=[1.,1.],  **kwargs):
        super(BasicGOTOCoords, self).__init__(*args, **kwargs)
        self.formation_name = formation
        print(formation)
        self.flag = False

    def select_coords_lmark_formation(self):
        # import matplotlib.pyplot as plt
        lmark = self.controller_owner.virtual_particle.lmark
        self.formation = get_formation(self.formation_name)
        # self.formation = get_formation('formationE')
        if lmark is None:
            self.target_coords = np.zeros(2)
            return
        state = self.controller_owner.virtual_particle.state
        # form_triangle = {'nodes' : .5 * np.array([[-1, 0],[1,0],[0,1]]), 'edges' : [(0)]}

        # plt.scatter(formationB[:,0], formationB[:,1])
        # plt.show()
        # __import__('pdb').set_trace()

        neigh_positions = [epk.position[:2] for epk in self.controller_owner.neighbors]
        center = np.mean(neigh_positions,0)

        # fneighs = get_formation_neighbors(self.formation,lmark , max_dist=15)
        # center = np.zeros(2)
        # n = 0
        # for lm in range(len(self.formation['nodes'])):
        #     if lm not in fneighs:
        #         continue
        #     lm_agents = [nn.position[:2] for nn in self.robot.neighbors if nn.virtual_particle.lmark == lm]
        #     if len(lm_agents) == 0:
        #         continue
        #     center += lm_agents[0]
        #     n += 1
        # center /= n

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
        # if self.t < 1500:
        #     self.get_actuator('joint_velocity_actuator').action = np.zeros(2)
        #     return
        # if self.t == 1500:
        #     self.controller_owner.virtual_particle.disabled_lmarks.append(self.controller_owner.virtual_particle.lmark)

        # area_read = state['ground_sensor']
        curr_pos = self.get_sensor_reading('own_position_sensor')[:2]
        # if self.t > 3000:
        #     self.formation = get_formation('formationE')
        self.select_coords_lmark_formation()
        # if area_read > 0 and np.linalg.norm(curr_pos - self.target_coords) < 0.4:
        #     return {'joint_velocity_actuator' : np.array([0, 0])}



        dist_tar = np.linalg.norm(self.target_coords - curr_pos)
        if dist_tar > 10:
            __import__('pdb').set_trace()
        desired_dir = (self.target_coords - curr_pos) / dist_tar
        # desired_dir -= v_obstacle


        robot_ori = self.controller_owner.orientation[-1]
        heading_ori = np.r_[np.cos(robot_ori), np.sin(robot_ori)]
        a1 = compute_angle(desired_dir)
        a2 = compute_angle(heading_ori)
        alp = 5 # if len(self.formation.get('nodes', [])) <= 9 else 1
        # A = 1 / (1 + np.exp(-alp * (dist_tar - .05)))
        A = 1 / (1 + np.exp(-5* (dist_tar - .6)))
        if dist_tar<= 0.05: A = 0
        print(dist_tar, A)
        if dist_tar < 0.2:
            A = 0.4
        B = np.cos(a1 - a2)
        lmark = self.controller_owner.virtual_particle.lmark
        # if self.formation.get('leader', -1) == lmark and  dist_tar < 1 :
        #     action = np.zeros(2)
        if False and self.formation.get('leader', -1) == lmark and  dist_tar < .2:
            a1 = np.pi / 2 
            angle = angle_diff(a2, a1)
            if angle <= 0.3:
                action = 0.4*np.array([1, 1])
            elif np.abs(angle - np.pi) <= 0.3:
                action = 0.1*np.array([-1,-1])
            elif a1 > a2:
                if a1 - a2 > np.pi:
                    action =  .3*np.array([-1., 1])
                else:
                    action =  .3* np.array([1., -1])
            else:
                if a2-a1 >np.pi:
                    action =  .3* np.array([1., -1])
                else:
                    action =  .3*np.array([-1., 1])
        else:
            angle = angle_diff(a1,a2)
            # a1 = a1 % (2*np.pi)
            # a2 = a2 % (2*np.pi)
            if angle <= 0.5:
                action = A * np.array([1, 1])
            # elif np.abs(angle - np.pi) <= 0.3:
            #     action = A*np.array([-1,-1])
            elif a1 > a2:
                if a1 - a2 > np.pi:
                    action =  .3*np.array([-1., 1])
                else:
                    action =  .3* np.array([1., -1])
            else:
                if a2-a1 >np.pi:
                    action =  .3* np.array([1., -1])
                else:
                    action =  .3*np.array([-1., 1])
        
        # if np.linalg.norm(self.target_coords - curr_pos) < 0.1:
        #     action = np.array([0,0])
        self.flag = True
        self.get_actuator('joint_velocity_actuator').action = action / 3

    def reset(self):
        self.flag = False
        self.robot.virtual_particle.lmark_priorities = np.ones(len(self.robot.virtual_particle.landmarks)) 
        
