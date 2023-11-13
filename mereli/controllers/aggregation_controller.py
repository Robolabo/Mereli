import numpy as np
from mereli.controllers import RobotController
from mereli.register import controller_registry
from mereli.utils import compute_angle, angle_diff




@controller_registry(name='comm_space_group_agg') 
class CommSpaceGroupAgg(RobotController):
    def __init__(self, *args,   **kwargs):
        super(CommSpaceGroupAgg, self).__init__(*args, **kwargs)
        self.flag = False
        self.ga_centers = np.array([[1,0], [-1,0], [0,1]])
        self.num_ga = [0.25, 0.25, 0.5]


    def select_coords(self):
        lmark = self.robot.virtual_particle.lmark
        state = self.robot.virtual_particle.state
        if lmark is None:
            self.target_coords = np.zeros(2)
            return
        # g1 = [0,1,2,3,4]
        # g2 = [5,6,7,8,9]
        # g3 = [10,11,12,13,14]
        g1 = np.arange(0, 10)
        g2 = np.arange(10, 20)
        g3 = np.arange(20, 40)
        my_group = None
        center1 = np.array([rob.position[:2] for rob in self.robot.neighbors if rob.virtual_particle.lmark in g1])
        center2 = np.array([rob.position[:2] for rob in self.robot.neighbors if rob.virtual_particle.lmark in g2])
        center3 = np.array([rob.position[:2] for rob in self.robot.neighbors if rob.virtual_particle.lmark in g3])
        if lmark in g1:
            if len(center1) > 0:
                center1 = np.vstack((center1, self.robot.position[:2]))
            else:
                center1 = self.robot.position[:2]
            my_group = 0
        if lmark in g2:
            if len(center2) > 0:
                center2 = np.vstack((center2, self.robot.position[:2]))
            else:
                center2 = self.robot.position[:2]
            my_group = 1
        if lmark in g3:
            try:
                if len(center3) > 0:
                    center3 = np.vstack((center3, self.robot.position[:2]))
                else:
                    center3 = self.robot.position[:2]
            except:
                __import__('pdb').set_trace()
            my_group = 2
        center1 = np.mean(center1,0) 
        center2 = np.mean(center2,0) 
        center3 = np.mean(center3,0)
        centers = [center1, center2, center3]
        my_center = centers[my_group]
        v_repel = np.zeros(2)
        dmin = 3
        for i,center in enumerate([center1, center2, center3]):
            if not np.isnan(center).any() and i != my_group:
                if np.linalg.norm(center-my_center)<3:
                    v_repel += center - my_center

        self.target_coords = my_center - v_repel



    def step(self, state, reward=0.0):
        # if self.t < 1500:
        #     self.get_actuator('joint_velocity_actuator').action = np.zeros(2)
        #     return
        # if self.t == 1500:
        #     self.controller_owner.virtual_particle.disabled_lmarks.append(self.controller_owner.virtual_particle.lmark)

        # area_read = state['ground_sensor']
        curr_pos = self.get_sensor_reading('own_position_sensor')[:2]
        self.select_coords()

        dist_tar = np.linalg.norm(self.target_coords - curr_pos)
        desired_dir = (self.target_coords - curr_pos) / dist_tar


        robot_ori = self.controller_owner.orientation[-1]
        heading_ori = np.r_[np.cos(robot_ori), np.sin(robot_ori)]
        a1 = compute_angle(desired_dir)
        a2 = compute_angle(heading_ori)
        lmark = self.controller_owner.virtual_particle.lmark
        angle = angle_diff(a1,a2)
        if angle <= 0.5:
            action = np.array([1, 1])
        elif np.abs(angle - np.pi) <= 0.3:
            action = np.array([-1,-1])
        elif a1 > a2:
            if a1 - a2 > np.pi:
                action =  .5*np.array([-1., 1])
            else:
                action =  .5* np.array([1., -1])
        else:
            if a2-a1 >np.pi:
                action =  .5* np.array([1., -1])
            else:
                action =  .5*np.array([-1., 1])

        
        # if np.linalg.norm(self.target_coords - curr_pos) < 0.1:
        #     action = np.array([0,0])
        self.flag = True
        self.get_actuator('joint_velocity_actuator').action = action

    def reset(self):
        self.flag = False
        self.robot.virtual_particle.lmark_priorities = np.ones(len(self.robot.virtual_particle.landmarks)) 
        
