import numpy as np
from mereli.controllers import RobotController
from mereli.register import controller_registry, controllers
from mereli.utils import compute_angle, angle_diff 

@controller_registry(name='forage_comm_space') 
class ForageCommSpace(RobotController):
    def __init__(self, *args,  **kwargs):
        super(ForageCommSpace, self).__init__(*args, **kwargs)
        self.flag = False
        self.roles = ['NEST', 'FOOD_1', 'FOOD_2', 'LOAD_BAT']
        self.priorities = [2, 2, 1, 1, 3]
        self.curr_role = None
        self.waiting_bat = False
        self.obstacle_avoider = controllers['basic_obstacle_avoider'](sensitivity=0.8)
    
    def select_role(self):
        lmk = self.controller_owner.virtual_particle.lmark
        if lmk == 0: 
            self.curr_role = 'NEST'
        elif lmk == 1: 
            self.curr_role = 'NEST'
        elif lmk == 2: 
            self.curr_role = 'FOOD_1'
        elif lmk== 3: 
            self.curr_role = 'FOOD_2'
        elif lmk== 4: 
            self.curr_role = 'LOAD_BAT'
        else:
            self.curr_role = 'FOOD_2'

    def load_battery(self, state):
        bat_lv = self.get_sensor_reading('battery_sensor') 
        action = np.array([0,0])
        # self.flags['load'] = False 
        ls_read = self.get_sensor_reading('red_light_sensor') 
        if np.max(ls_read) > 0.85:
            self.get_actuator('joint_velocity_actuator').action = np.zeros(2)
            return 

        if ls_read[0] * ls_read[7] == 0:
            light_left = np.sum(ls_read[[7,6,5,4]])
            light_right = np.sum(ls_read[[0,1,2,3]])
            if light_right > light_left:
                action = 0.1 * np.array([-1, 1]) 
            else: 
                action = 0.1 * np.array([1, -1]) 
        else:
            action = 0.7 * np.array([1,1])
        self.get_actuator('joint_velocity_actuator').action = action 

    def step(self, state, reward=0.0):
        # if self.t == 1500:
        #     self.controller_owner.virtual_particle.disabled_lmarks.append(self.controller_owner.virtual_particle.lmark)

        area_read = self.get_sensor_reading('memory_ground_sensor')
        curr_pos = self.get_sensor_reading('own_position_sensor')[:2]
        bat = self.get_sensor_reading('battery_sensor')
        if bat < 0.5 or (self.waiting_bat and bat < 0.9):
            self.waiting_bat = True
            self.controller_owner.virtual_particle.disabled_lmarks = []
            for i in range(len(self.roles)): 
                if self.roles[i] != 'LOAD_BAT':
                    self.controller_owner.virtual_particle.disabled_lmarks.append(i)
        else:
            self.waiting_bat = False
            self.controller_owner.virtual_particle.disabled_lmarks = [len(self.priorities)-1]
            
        nest_pos = np.array([0,0])
        food_pos = np.array([[-2, 0],[2, 0]])
        self.select_role()
        print(self.curr_role)
        if self.curr_role == 'NEST':
            self.target_coords = nest_pos
        elif self.curr_role == 'LOAD_BAT':
            return self.load_battery(state)
        elif self.curr_role == 'FOOD_1':
            if area_read[0] == 0: 
                self.target_coords = food_pos[0] 
            else:
                self.target_coords = nest_pos
        elif self.curr_role == 'FOOD_2':
            if area_read[0] == 0: 
                self.target_coords = food_pos[1] 
            else:
                self.target_coords = nest_pos
        else:
            self.target_coords = food_pos[1] 

        # action = self.obstacle_avoider.step(state)
        # if self.obstacle_avoider.flag:
        #     return action 
        dist_tar = np.linalg.norm(self.target_coords - curr_pos)
        if dist_tar < 0.3 and self.curr_role == 'NEST':
            self.get_actuator('joint_velocity_actuator').action = np.zeros(2)
            return
        desired_dir = (self.target_coords - curr_pos) / dist_tar
        # desired_dir -= v_obstacle


        robot_ori = self.controller_owner.orientation[-1]
        heading_ori = np.r_[np.cos(robot_ori), np.sin(robot_ori)]
        a1 = compute_angle(desired_dir)
        a2 = compute_angle(heading_ori)
        A = 1 / (1 + np.exp(-20 * (dist_tar - .1)))
        B = np.cos(a1 - a2)


        if angle_diff(a1, a2) <= 0.5:
            action = A * np.array([1, 1])
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
        self.waiting_bat = False
        self.controller_owner.virtual_particle.lmark_priorities = self.priorities
        self.obstacle_avoider.reset()
        self.obstacle_avoider.controller_owner = self.controller_owner

