import numpy as np
from mereli.controllers import RobotController
from mereli.register import controller_registry, controllers
from mereli.utils import compute_angle, angle_diff 
from mereli.utils import angle_mean

@controller_registry(name='forage_comm_space') 
class ForageCommSpace(RobotController):
    def __init__(self, *args,  **kwargs):
        super(ForageCommSpace, self).__init__(*args, **kwargs)
        self.flag = False
        self.roles = ['NEST', 'FOOD_1', 'FOOD_2', 'LOAD_BAT']
        self.priorities = [2, 2, 1, 1, 5]
        self.curr_role = None
        self.waiting_bat = False
        self.obstacle_avoider = controllers['basic_obstacle_avoider'](sensitivity=0.3)
    
    def select_role(self):
        lmk = self.controller_owner.virtual_particle.lmark
        try:
            self.curr_role = self.roles[int(lmk)]
        except:
            __import__('pdb').set_trace()

    def load_battery(self, state):
        bat_lv = self.get_sensor_reading('battery_sensor') 
        action = np.array([0,0])
        # self.flags['load'] = False 
        ls_read = self.get_sensor_reading('red_light_sensor') 
        if np.max(ls_read) > 0.8:
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
        # if self.t > 1:
        if self.t < 100:#500:
            return

        area_read = self.get_sensor_reading('memory_ground_sensor')
        curr_pos = self.get_sensor_reading('own_position_sensor')[:2]
        bat = self.get_sensor_reading('battery_sensor')

        print(self.robot.virtual_particle.lmark_priorities)
        if bat < 0.5 or (self.waiting_bat and bat < 0.9):
            self.waiting_bat = True
            self.controller_owner.virtual_particle.disabled_lmarks = []
            self.robot.virtual_particle.lmark_priorities[-1] = 0
            # for i in range(len(self.roles)): 
            #     if self.roles[i] != 'LOAD_BAT':
            #         self.controller_owner.virtual_particle.disabled_lmarks.append(i)
        else:
            self.waiting_bat = False
            self.robot.virtual_particle.lmark_priorities[-1] = 3
            # self.controller_owner.virtual_particle.disabled_lmarks = [len(self.priorities)-1]
            
        nest_pos = [*self.robot.physics_client.ground_areas.values()][0]['center']
        food_pos = np.vstack(([*self.robot.physics_client.ground_areas.values()][2]['center'], 
                             [*self.robot.physics_client.ground_areas.values()][1]['center']))
        self.select_role()
        # if self.robot.id %2 ==0:
        #     self.curr_role = 'FOOD_1'
        # else:
        #     self.curr_role = 'FOOD_2'
        # print(self.curr_role)
        if self.curr_role == 'NEST':
            self.target_coords = nest_pos
        elif self.curr_role == 'LOAD_BAT':
            if 'led' in self.robot.actuators:
                self.get_actuator('led').action = 1
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

        dist_tar = np.linalg.norm(self.target_coords - curr_pos)
        # if dist_tar < 0.75 and self.curr_role == 'NEST':
        #     self.target_coords = curr_pos
            # self.get_actuator('joint_velocity_actuator').action = np.zeros(2)
            # return

        desired_dir = (self.target_coords - curr_pos) / dist_tar
        # desired_dir -= v_obstacle
        if self.curr_role == 'NEST' and dist_tar < 0.75:
            desired_dir = None
        st_ds = self.get_sensor_reading('distance_sensor')
        sens = 0.2 if self.curr_role =='NEST' else 0.4
        if np.max(st_ds) > sens:
            oris = self.controller_owner.sensors['distance_sensor'].directions(self.controller_owner.orientation[-1])
            obs_dir = -np.sum([st_ds[i] * np.r_[np.cos(oris[i]), np.sin(oris[i])] for i in range(len(st_ds))],0) 
            # max_i = np.argmax(st_ds)
            # obs_dir = -np.r_[np.cos(oris[max_i]), np.sin(oris[max_i])] 
            # obs_dir = np.argsort(st_ds[st_ds < 0.1])[0]
            # obs_dir = np.r_[np.cos(obs_dir), np.sin(obs_dir)]
            angle1 = compute_angle(obs_dir) 
            if desired_dir is not None:
                angle2 = compute_angle(desired_dir)
                # obs_dir /= np.linalg.norm(obs_dir)
                mean_angle = angle_mean([angle1, angle2], weights=[0.6, 0.4])
                desired_dir = np.r_[np.cos(mean_angle), np.sin(mean_angle)] 
            else:
                desired_dir = obs_dir
        if desired_dir is None:
            self.get_actuator('joint_velocity_actuator').action = np.zeros(2)
            return
        # self.robot.physics_client.draw_line(self.robot.position, self.robot.position + np.r_[desired_dir, 0.05] )
        robot_ori = self.controller_owner.orientation[-1]
        heading_ori = np.r_[np.cos(robot_ori), np.sin(robot_ori)]
        a1 = compute_angle(desired_dir)
        a2 = compute_angle(heading_ori)

        angle = angle_diff(a1,a2)
        # a1 = a1 % (2*np.pi)
        # a2 = a2 % (2*np.pi)
        if angle <= 0.5:
            action = np.array([1, 1])
        elif np.abs(angle - np.pi) <= 0.3:
            action = 0.7*np.array([-1,-1])
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

        self.flag = True
        self.get_actuator('joint_velocity_actuator').action = action 
        if 'led' in self.robot.actuators:
            self.get_actuator('led').action = {'NEST' : 2, 'FOOD_1' : 3, 'FOOD_2' : 4, 'LOAD_BAT' : 1}.get(self.curr_role, 0)



    def reset(self):
        self.flag = False
        self.waiting_bat = False
        n_lmarks = len(self.robot.virtual_particle.landmarks)
        self.roles = ['NEST'] * int((n_lmarks-1) // 2) + ['FOOD_1'] * int((n_lmarks - 1) // 4) + ['FOOD_2'] * int((n_lmarks - 1) // 4)  + ['LOAD_BAT']
        self.priorities = [{'NEST' : 2, 'FOOD_1' : 1, 'FOOD_2' : 1, 'LOAD_BAT' : 3}.get(role) for role in self.roles]
        self.controller_owner.virtual_particle.lmark_priorities = self.priorities
        self.obstacle_avoider.reset()
        self.obstacle_avoider.controller_owner = self.controller_owner

