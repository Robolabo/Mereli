import numpy as np
from mereli.controllers import RobotController
from mereli.register import controller_registry
from mereli.utils import compute_angle, angle_diff

@controller_registry(name='basic_obstacle_avoider')
class BasicObstacleAvoider(RobotController):
    """ Controller devoted to the obstacle avoidance task. This means that the controller 
    will read from the distance sensor, process the measurements and return joint velocity 
    actions required to avoid colliding with any other tangible entity. This controller is 
    currently hard coded for the Epuck robot.

    :param float sensitivity: value in [0, 1] that defines the threshold in the distance sensor reading 
        to interpret an obstacle detection. 
    """
    def __init__(self, *args, sensitivity=0.4, no_obstacle_action=[1.,1.],  **kwargs):
        super(BasicObstacleAvoider, self).__init__(*args, **kwargs)
        self.sensitivity = sensitivity
        self.no_obstacle_action = no_obstacle_action
        self.flag = False

    def step(self, state, reward=0.0):
        """ Method to execute once the controller program. It reads the current distance sensor measurement, 
        and plans the action as follows:

        .. code-block:: 
        
            IF DS[0] > SENSITIVITY OR DS[1] > SENSITIVITY THEN
                TURN LEFT
            ELSE IF DS[6] > SENSITIVITY OR DS[7] > SENSITIVITY THEN
                TURN RIGHT
            ELSE THEN
                GO STRAIGHT OVER
        
        :param dict state: state with the sensor reading. The dict maps the reference name of the sensor to 
            the sensor np.ndarray reading.
        :param float reward: reward (if any). Not used in this controller.
        """
        # st_ds = state['distance_sensor']
        st_ds = self.get_sensor('distance_sensor').reading
        self.flag = np.max(st_ds) > self.sensitivity 
        # if any(st_ds[[0,1]] > 0.95): # critical
        #     action = np.array([-1., -1])
        # elif any(st_ds[[3,4]] > 0.9):
        #     # print('Turn Right')
        #     action = np.array([1, 1.])
        if any(st_ds[[0,1]] > self.sensitivity):
            # print('Turn Left')
            action = np.array([1., -1])
        elif any(st_ds[[6,7]] > self.sensitivity):
            # print('Turn Right')
            action = np.array([-1, 1.])
        # elif st_ds[3] > self.sensitivity:
        #     action = np.array([1, 1.])
        #     self.falg = True
        # elif st_ds[4] > self.sensitivity:
        #     action = np.array([-1, 1.])
        #     self.falg = True
        else:
            # print('GO straight over')
            action = np.array(self.no_obstacle_action)
        action *= 0.5 #!!!!!!!!!! 
        self.get_actuator('joint_velocity_actuator').action = action
        if self.is_actuator_enabled('led'):
            self.get_actuator('led').action = int(self.flag)
        # Turn on the LED of the obstacle direction.

@controller_registry(name='obstacle_avoider2')
class ObstacleAvoider2(RobotController):
    def __init__(self, *args, sensitivity=0.4, no_obstacle_action=[1.,1.],  **kwargs):
        super(ObstacleAvoider2, self).__init__(*args, **kwargs)
        self.sensitivity = sensitivity
        self.no_obstacle_action = no_obstacle_action
        self.flag = False

    def step(self, state, reward=0.0):
        st_ds = self.get_sensor('distance_sensor').reading
        self.flag = False
        if np.max(st_ds) > self.sensitivity:
            self.flag = True
            oris = self.controller_owner.sensors['distance_sensor'].directions(self.controller_owner.orientation[-1])
            desired_dir = -np.sum([st_ds[i] * np.r_[np.cos(oris[i]), np.sin(oris[i])] for i in range(len(st_ds))],0) 
            robot_ori = self.controller_owner.orientation[-1]
            heading_ori = np.r_[np.cos(robot_ori), np.sin(robot_ori)]
            a1 = compute_angle(desired_dir)
            a2 = compute_angle(heading_ori)
            angle = angle_diff(a1,a2)
            if angle <= 0.5:
                action = np.array([1, 1])
            elif np.abs(angle - np.pi) <= 0.3:
                action = 0.7*np.array([-1,-1])
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
        
            self.get_actuator('joint_velocity_actuator').action = action
 

@controller_registry(name='epuck_obstacle_avoider')
class EpuckObstacleAvoid(RobotController):
    def __init__(self, *args, sensitivity=0.5, no_obstacle_action=[1.,1.],  **kwargs):
        super(EpuckObstacleAvoid, self).__init__(*args, **kwargs)
        self.sensitivity = sensitivity
        self.no_obstacle_action = no_obstacle_action
        self.flag = False # True if obstacle detected

    def step(self, state, reward=0.0):
        prox_read = state['distance_sensor']
        oris = self.controller_owner.sensors['distance_sensor'].directions(self.controller_owner.orientation[-1])
        max_prox = np.max(prox_read) 
        action = np.array(self.no_obstacle_action)
        self.flag = False
        if max_prox > self.sensitivity:
            v_repel = np.sum([prox_read[i] * np.r_[np.cos(oris[i]), np.sin(oris[i])] for i in range(8)], 0)
            f_repel = compute_angle(v_repel) -  np.pi
            fc1 = 1 / np.pi
            fc_linear = 1 
            fc_angular = 1
            fv_linear = fc_linear * np.cos(f_repel / 2)
            fv_angular = f_repel 
            action = np.array([fv_linear + fc1 * fv_angular, fv_linear - fc1 * fv_angular])
            self.flag = True 
        return {'joint_velocity_actuator' : action}

    def reset(self):
        self.obstacle_detected = False

