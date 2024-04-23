
import numpy as np
from mereli.controllers import RobotController
from mereli.register import controller_registry
from mereli.utils import compute_angle, angle_diff
from mereli.communication import RobotMolecule


class EllipseFormation:
    def __init__(self, num_points, a=1, b=1):
        self._num_points = num_points 
        # self.e = e
        # self.c = c
        self._a = a 
        self._b = b
        self.coords = [] 
        self.compute_points()

    def compute_points(self):
        tt = np.linspace(0, 2*np.pi - 2*np.pi/self._num_points, num=self._num_points)
        xx = self._a * np.cos(tt)
        yy = self._b * np.sin(tt)
        self.coords = np.array([xx, yy]).T
        # self.rotate(np.radians(45))

        # import matplotlib.pyplot as plt
        # plt.scatter(self.coords[:,0], self.coords[:,1])
        # plt.xlim(-3,3);plt.ylim(-3,3)
        # plt.show()
        # __import__('pdb').set_trace()
    
    def rotate(self, angle):
        rot_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        self.coords = np.array([rot_mat.dot(pt) for pt in self.coords])

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b
    
    @a.setter
    def a(self, new_a):
        self._a = new_a
        self.compute_points()

    @b.setter
    def b(self, new_b):
        self._b = new_b
        self.compute_points()

@controller_registry(name='ellipse_formation') 
class EllipseFormationController(RobotController):
    """ Robot controller for the ellipse formation using virtual spaces. 
    """
    def __init__(self, *args,  **kwargs):
        super(EllipseFormationController, self).__init__(*args, **kwargs)
        self.formation_name = 'ellipse' 
        self.formation = EllipseFormation(8, a=1, b=1)
        self.flag = False
        self.init_center = None

    def plan_navigation(self, center):
        """ Computes the points in a ellipse formation based on a given center.
        The paramters of the ellipse are obtained via the formation attribute that contains an 
        instance of the class EllipseFormation.

        If the formation remains fixed during all the simulation then this method can be empty. 
        In case of a coordinated movement keeping a given formation, then the center of mass can be 
        moved using a given trajectory.
        """
        com_eps = 0.2
    
        if self.t < 700: # Still converging to formation
            self.init_center = center # No problem with odom
        else:
            pass
            # IMPORTANT: IGNORE COMMENTS, IS FOR SOMETHING ELSE.
            """ DOOR """
            # if np.abs(center[1]) > 0.05:
            #     center[1] = 0
            # center[0] += com_eps
            # if center[0] > 1.5 and center[0] < 5:
            #     self.formation.b = max(0.15, self.formation.b - 0.05) 
            # elif center[0] > 5:
            #     self.formation.b = min(1, self.formation.b + 0.05) 

            """ Ellipse oscill. excentriciy """
            # self.formation.b = 0.5  * (1 + np.cos(2*np.pi * .02 *  self.t * 0.05)) + 0.15

            """ Ellipse rotation """
            # com_eps = .1
            # center[0] -= com_eps
            # center[1] += com_eps
            # self.formation.b = 0.4
            # self.formation.rotate(np.radians(135)) 
            
            """ Trajectory """
            # com_eps = .1
            # self.formation.b = 0.4
            # self.formation.a = 0.4
            # center[0] -= 0.1
            # center[1] = np.sin(2*np.pi*0.2*center[0]- self.init_center[0])
        return center

    def select_coords_lmark_formation(self):
        """ The controller has access to the current landmark in the virtual space"""
        lmark = self.controller_owner.virtual_particle.lmark

        if lmark is None:
            self.target_coords = np.zeros(2)
            return
        state = self.controller_owner.virtual_particle.state
        """ In mereli neighbors are computed external to the controller. """
        neigh_positions = np.vstack([epk.position[:2] for epk in self.controller_owner.neighbors])
        # Estimate center of formation (in ellipse)
        center = np.mean(neigh_positions,0)
        center = self.plan_navigation(center)
        
        # Compute the target position in the formation (in the real world) where the robot where the 
        # robot should go. This coords are obtained based on the landmark and the center of mass. 
        self.target_coords = self.formation.coords[lmark].copy() if lmark is not None else np.zeros(2)
        self.target_coords += center


    def step(self, state, reward=0.0):
        """ Step method of the controller. 
        Remark:  the state input argument is currently useless, sensors are read using get_sensor_reading (see below). 
        In the future after a code clean the state arg will be removed. 
        """
        # Reading of GPS sensor
        curr_pos = self.get_sensor_reading('own_position_sensor')[:2]
        # Important: Selection of the formation coordinates based on virtual space. 
        self.select_coords_lmark_formation()

        # This part is not relevant for Nautilus, essentially it computes some simplified IK to 
        # move the robot towards the target position. 
        dist_tar = np.linalg.norm(self.target_coords - curr_pos)
        desired_dir = (self.target_coords - curr_pos) / dist_tar
        # desired_dir -= v_obstacle


        robot_ori = self.controller_owner.orientation[-1]
        heading_ori = np.r_[np.cos(robot_ori), np.sin(robot_ori)]
        a1 = compute_angle(desired_dir)
        a2 = compute_angle(heading_ori)
        alp = 5 # if len(self.formation.get('nodes', [])) <= 9 else 1
        A = 1 / (1 + np.exp(-5* (dist_tar - .6)))
        if dist_tar<= 0.05: A = 0
        # if dist_tar < 0.2:
        #     A = 0.4
        B = np.cos(a1 - a2)
        lmark = self.controller_owner.virtual_particle.lmark
        angle = angle_diff(a1,a2)
        if angle <= 0.5:
            action = A*np.array([1, 1])
        elif np.abs(angle - np.pi) <= 0.3:
            action = 0.5*np.array([-1,-1])
        elif a1 > a2:
            if a1 - a2 > np.pi:
                action =  .2*np.array([-1., 1])
            else:
                action =  .2* np.array([1., -1])
        else:
            if a2-a1 >np.pi:
                action =  .2* np.array([1., -1])
            else:
                action =  .2*np.array([-1., 1])
        self.flag = True
        self.get_actuator('joint_velocity_actuator').action = action

    def reset(self):
        self.flag = False
        # In this example all priorities to one. 
        self.robot.virtual_particle.lmark_priorities = np.ones(len(self.robot.virtual_particle.landmarks)) 



@controller_registry(name='ellipse_navigation') 
class EllipseNavigationController(RobotController):
    """ 
    """
    def __init__(self, *args,  **kwargs):
        super(EllipseNavigationController, self).__init__(*args, **kwargs)
        self.flag = False 
        self.formation_ctrl = EllipseFormationController()

    def get_effective_sensors(self, idx):
        return np.array({0 : [0,1,2, 5,6,7] , 1 : [5,6,7], 7 :[0,1,2] }.get(idx))
    
    def get_swarm_reading(self):
        ds_tot = np.zeros(12).astype(float)
        for robot in [self.robot] +  self.robot.neighbors:
            dsi = robot.sensors['distance_sensor'].reading
            lm = robot.virtual_particle.lmark
            if lm == 0:
                ds_tot[3:9] = dsi[self.get_effective_sensors(lm)]
            if lm == 1:
                ds_tot[:3] = dsi[self.get_effective_sensors(lm)]
            if lm == 7:
                ds_tot[9:12] = dsi[self.get_effective_sensors(lm)]
        return ds_tot

    def step(self, state, reward=0.0):
        if self.t == 0:
            return
        curr_pos = self.get_sensor_reading('own_position_sensor')[:2]
        # ds_all = self.get_swarm_reading()
        # print(ds_all)


        self.formation_ctrl.step(state)
        self.flag = True
     
    def reset(self):
        self.flag = False
        self.formation_ctrl.controller_owner = self.controller_owner
        self.formation_ctrl.reset()
