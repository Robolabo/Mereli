import numpy as np
from mereli.controllers import RobotController, BasicObstacleAvoider
from mereli.register import controller_registry

@controller_registry(name='test_wheels')
class TestWheelsController(RobotController):
    """
    """
    def step(self, state, reward=0.0):
        action_wheels = (1, 1)
        self.get_actuator('joint_velocity_actuator').action = np.array(action_wheels)

@controller_registry(name='test_led')
class TestLEDController(RobotController):
    """
    """
    def step(self, state, reward=0.0):
        i = self.t // 100
        action = np.zeros(8)
        action[(self.t // 50) % 8] = 1 # Control the individual leds
        # action *= i % 7 # Iterate colors every 100 ticks
        self.get_actuator('led').action = action 
        
@controller_registry(name='test_contact')
class TestContactController(RobotController):
    """
    """
    def step(self, state, reward=0.0):
        contact_st = self.get_sensor('contact_sensor').reading
        print(f'Reading of Contact Sensor is {contact_st}')
        self.get_actuator('joint_velocity_actuator').action = np.array([1, 1])
        self.get_actuator('led').action = contact_st 

@controller_registry(name='test_proximity')
class TestProximityController(RobotController):
    """
    """
    def step(self, state, reward=0.0):
        prox_read = self.get_sensor('distance_sensor').reading
        self.get_actuator('joint_velocity_actuator').action = np.array([1, 1])
        self.get_actuator('led').action = int(np.max(prox_read) > 0.5)
        print(f'Reading of Proximity Sensor is {prox_read}')

@controller_registry(name='test_red_light_sensor')
class TestRedLightSensorController(RobotController):
    """
    """
    def step(self, state, reward=0.0):
        ls_read = self.get_sensor_reading('red_light_sensor') 
        print(f'Reading of Proximity Sensor is {ls_read}')
        self.get_actuator('joint_velocity_actuator').action = np.array([1, 1])

@controller_registry(name='test_green_light_sensor')
class TestGreenLightSensorController(RobotController):
    """
    """
    def step(self, state, reward=0.0):
        ls_read = self.get_sensor_reading('green_light_sensor') 
        print(f'Reading of Proximity Sensor is {ls_read}')
        self.get_actuator('joint_velocity_actuator').action = np.array([1, 1])

@controller_registry(name='test_blue_light_sensor')
class TestBlueLightSensorController(RobotController):
    """
    """
    def step(self, state, reward=0.0):
        ls_read = self.get_sensor_reading('blue_light_sensor') 
        print(f'Reading of Proximity Sensor is {ls_read}')
        self.get_actuator('joint_velocity_actuator').action = np.array([1, 1])
        
@controller_registry(name='test_switch_light')
class TestSwitchLight(RobotController):
    """
    """
    def __init__(self, *args,  **kwargs):
        super(TestSwitchLight, self).__init__(*args, **kwargs)

    def step(self, state, reward=0.0):
        action = 0
        ls_st = self.get_sensor_reading('red_light_sensor')
        # print(f'Light Sensor reading is {ls_st}')
        # print(self.t )
        
        if self.t % 150 == 0:
            # if self.t > 500:
            action = 1
        self.get_actuator('switch_light').action = np.array(action)




@controller_registry(name='test_switch_light_color')
class TestSwitchLightColor(RobotController):
    """
    """
    def __init__(self, *args,  **kwargs):
        super(TestSwitchLightColor, self).__init__(*args, **kwargs)
        self.color_sequence = ['red', 'green', 'blue', 'yellow'] 
        self.curr_color = 0 

    def step(self, state, reward=0.0):
        action = 0
        if self.t % 150 == 0:
            self.curr_color = (self.curr_color + 1) % 4
        action = self.color_sequence[self.curr_color]
        self.get_actuator('switch_light_color').action = action

@controller_registry(name='test_encoder')
class TestEncoderController(RobotController):
    """
    """
    def __init__(self, *args,  **kwargs):
        super(TestEncoderController, self).__init__(*args, **kwargs)
        self.estim_pos = np.zeros(2)
        self.estim_ori = 0.0
        self.dist_walked = 0
        self.error_position = np.zeros(2)
        self.error_orientation = 0.0 

    
    def compute_odom(self,state):
        dt = 10 * self.robot.physics_client.dt
        b = 0.12 # Dist between wheels
        R = 0.0390625 # Wheel radius
        enc = self.get_sensor_reading('encoder')
        dUl = R * enc[1] * dt
        dUr = R * enc[0] * dt
        w = (dUr - dUl) / b
        # r = (b/2) * (dUr + dUl) / (dUr - dUl + 1e-3)
        V = (dUr + dUl) / 2
        dx = np.cos(self.estim_ori) * V
        dy = np.sin(self.estim_ori) * V
        dTh = w 
        self.estim_pos += np.r_[dx, dy] 
        self.estim_ori += dTh
        self.estim_ori = self.estim_ori % (2*np.pi)


    def step(self, state, reward=0.0):
        self.compute_odom(state)
        real_pos = self.controller_owner.position[:2]
        real_ori = self.robot.orientation[-1]
        error_ori = min(np.abs(self.estim_ori- real_ori), 2*np.pi - np.abs(self.estim_ori- real_ori))
        error= self.estim_pos - real_pos
        self.error_orientation = error_ori
        self.error_position = error
        print('.....')
        print(f'Pos Estimation {self.estim_pos}, real={real_pos} and  error={error}' )
        print(f'Orientation Estimation {self.estim_ori}, real={real_ori} and  error={error_ori}' )
        pos = self.estim_pos
        ori = self.estim_ori

        if np.linalg.norm(pos - np.r_[1,0]) < 1e-2:
            if np.abs(ori - np.pi / 2) > 0.05:
                action_wheels = (1,-1)
            else:
                action_wheels = (1,1)
        elif np.linalg.norm(pos - np.r_[1,1]) < 3e-2:
            if np.abs(ori - np.pi) > 0.05:
                action_wheels = (1,-1)
            else:
                action_wheels = (1,1)
        elif np.linalg.norm(pos - np.r_[0,1]) < 3e-2:
            if np.abs(ori - 1.5*np.pi) > 0.05:
                action_wheels = (1,-1)
            else:
                action_wheels = (1,1)
        elif np.linalg.norm(pos - np.r_[0,0]) < 3e-2:
            if min(np.abs(ori), 2*np.pi - np.abs(ori)) > 0.1:
                action_wheels = (1,-1)
            else:
                action_wheels = (1,1)
        else:
            action_wheels = (1,1)
        red = 0.5
        self.get_actuator('joint_velocity_actuator').action = red * np.array(action_wheels)

@controller_registry(name='test_odometry')
class TestOdometryController(RobotController):
    """
    """
    def __init__(self, *args,  **kwargs):
        super(TestOdometryController, self).__init__(*args, **kwargs)

    def step(self, state, reward=0.0):
        estim_pos = self.get_sensor_reading('odometry')['position']
        estim_ori = self.get_sensor_reading('odometry')['orientation']
        # print(f'Pos Estimation {self.estim_pos}, real={real_pos} and  error={error}' )
        # print(f'Orientation Estimation {self.estim_ori}, real={real_ori} and  error={error_ori}' )
        pos = estim_pos
        ori = estim_ori
        eps = 5e-2
        if np.linalg.norm(pos - np.r_[1,0]) < eps:#2e-2:
            if np.abs(ori - np.pi / 2) > 0.05:
                action_wheels = (1,-1)
            else:
                action_wheels = (1,1)
        elif np.linalg.norm(pos - np.r_[1,1]) < eps:# 3e-2:
            if np.abs(ori - np.pi) > 0.05:
                action_wheels = (1,-1)
            else:
                action_wheels = (1,1)
        elif np.linalg.norm(pos - np.r_[0,1]) < eps:#3e-2:
            if np.abs(ori - 1.5*np.pi) > 0.05:
                action_wheels = (1,-1)
            else:
                action_wheels = (1,1)
        elif np.linalg.norm(pos - np.r_[0,0]) < eps:#3e-2:
            if min(np.abs(ori), 2*np.pi - np.abs(ori)) > 0.1:
                action_wheels = (1,-1)
            else:
                action_wheels = (1,1)
        else:
            action_wheels = (1,1)
        red = 0.1
        self.get_actuator('joint_velocity_actuator').action = red * np.array(action_wheels)

@controller_registry(name='test_battery')
class TestBattery(RobotController):
    """
    """
    def __init__(self, *args,  **kwargs):
        super(TestBattery, self).__init__(*args, **kwargs)

    def step(self, state, reward=0.0):
        battery_lv = self.get_sensor_reading('battery_sensor')
        rls = self.get_sensor_reading('red_light_sensor')
        print('Battery Level is ', battery_lv)
        action_wheels = (1,1)
        self.get_actuator('joint_velocity_actuator').action = np.array(action_wheels)

import time
@controller_registry(name='test_camera')
class TestCamera(RobotController):
    """
    """
    def __init__(self, *args,  **kwargs):
        super(TestCamera, self).__init__(*args, **kwargs)
        self.tprev = time.time()
        # self.add_routine(BasicObstacleAvoider)
        self.obstacle_avoider = BasicObstacleAvoider(sensitivity=0.7)

    def step(self, state, reward=0.0):
        print(time.time()-self.tprev)
        self.tprev = time.time()
        self.obstacle_avoider.step(state)
        # self.get_actuator('joint_velocity_actuator').action = np.ones(2)

    def reset(self,):
        self.obstacle_avoider.controller_owner = self.controller_owner
        self.obstacle_avoider.reset()


@controller_registry(name='nav_trajectory')
class NavigationTrajectory(RobotController):
    """
    """
    def __init__(self, *args,  **kwargs):
        super(NavigationTrajectory, self).__init__(*args, **kwargs)
        T = 500
        self.trajectory = lambda t : np.array([np.cos(2*np.pi*t/T), t/T, (np.pi / 2 + 2*np.pi*t / T) % (2*np.pi)]) 
        self.tar_state = np.zeros(3).astype(float)
       
    def control_step(self, ref, ori):
        error = min(np.abs(ref-ori), 2*np.pi - np.abs(ref-ori))
        if np.abs(ref-ori) > 2*np.pi -np.abs(ref-ori):
            sign = 1 if ref > ori else -1
        else:
            sign = -1 if ref > ori else 1
        arot = -1*sign * error 
        print(ref, ori)
        return arot, error

    def step(self, state, reward=0.0):
        pos = self.get_sensor_reading('odometry')['position'] 
        ori = self.get_sensor_reading('odometry')['orientation'] 

        t = (self.t+1) * 10 * self.robot.physics_client.dt 
        self.tar_state = self.trajectory(t)
        arot, err = self.control_step(self.tar_state[-1], ori)

        if err > 0.05:
            action_wheels = (arot, -arot)
        else:
            action_wheels = (0.5,0.5)
        self.get_actuator('joint_velocity_actuator').action = 0.3*np.array(action_wheels)

    def reset(self,):
        self.tar_state = np.zeros(3).astype(float)


from mereli.communication import IRFrame
from mereli.utils import angle_diff, compute_angle 
@controller_registry(name='testIRcomm')
class TestIRComm(RobotController):
    """
    """
    def __init__(self, *args,  **kwargs):
        super(TestIRComm, self).__init__(*args, **kwargs)
        self.flag = False
        self.counter = 0

    def rotate_clockwise(self, w=0.3):
        self.get_actuator('joint_velocity_actuator').action = w * np.array([-1, 1])

    def rotate_counterclockwise(self, w=0.3):
        self.get_actuator('joint_velocity_actuator').action = w * np.array([1, -1])

    def step(self, state, reward=0.0):
        # Send frame
        self.get_actuator('IRCommTX').action = IRFrame(1).set_msg(1.).set_sender(self.robot.id)

        fr = self.get_sensor_reading('IRCommRX')
        if fr.is_null:
            return 
        # adiff = angle_diff(fr.tx_ori, (fr.rx_ori + np.pi) % (2*np.pi))
        # adiff = angle_diff(fr.tx_ori, fr.rx_ori)
        utx = np.r_[np.cos(fr.tx_ori), np.sin(fr.tx_ori)]
        urx = np.r_[np.cos(fr.rx_ori), np.sin(fr.rx_ori)]

        a1 = compute_angle(utx)
        a2 = compute_angle(urx)
        angle = angle_diff(a1,a2)
        w = 0.02 if angle < 0.5 else 0.1
        self.flag = True 
        if np.abs(angle-np.pi) <= 0.4:
            # self.flag = False 
            self.counter += 1
            if self.counter >= 5:
                self.get_actuator('joint_velocity_actuator').action = 0.1 * np.array([1, 1])
            else:
                self.get_actuator('joint_velocity_actuator').action = 0.1 * np.array([0, 0])
            # return
        elif a1 > a2:
            self.counter = 0
            if a1 - a2 > np.pi:
                self.rotate_clockwise(w=w)
            else:
                self.rotate_counterclockwise(w=w)
        else:
            self.counter = 0
            if a2-a1 >np.pi:
                self.rotate_counterclockwise(w=w)
            else:
                self.rotate_clockwise(w=w)

@controller_registry(name='testIRcommV2')
class TestIRCommV2(RobotController):
    """
    """
    def __init__(self, *args,  **kwargs):
        super(TestIRCommV2, self).__init__(*args, **kwargs)
        self.flag = False

    def rotate_clockwise(self, w=0.3):
        self.get_actuator('joint_velocity_actuator').action = w * np.array([1, -1])

    def rotate_counterclockwise(self, w=0.3):
        self.get_actuator('joint_velocity_actuator').action = w * np.array([-1, 1])

    def step(self, state, reward=0.0):
        # Send frame
        self.get_actuator('IRCommTX').action = IRFrame(1).set_msg(1.).set_sender(self.robot.id)

        frames = self.get_sensor_reading('IRCommRX')
        if len(frames) == 0:
            print('OMGGG')
            return 

        # adiff = angle_diff(fr.tx_ori, (fr.rx_ori + np.pi) % (2*np.pi))
        # adiff = angle_diff(fr.tx_ori, fr.rx_ori)
        utx = np.r_[np.cos(fr.tx_ori), np.sin(fr.tx_ori)]
        urx = np.r_[np.cos(fr.rx_ori), np.sin(fr.rx_ori)]

        a1 = compute_angle(utx)
        a2 = compute_angle(urx)
        angle = angle_diff(a1,a2)
        w = 0.1 if angle < 0.5 else 0.1
        self.flag = True 
        if np.abs(angle-np.pi) <= 0.2 and self.t > 1000:
            # self.flag = False 
            self.get_actuator('joint_velocity_actuator').action = 0.1 * np.array([1, 1])
            # return
        elif a1 > a2:
            if a1 - a2 > np.pi:
                self.rotate_clockwise(w=w)
            else:
                self.rotate_counterclockwise(w=w)
        else:
            if a2-a1 >np.pi:
                self.rotate_counterclockwise(w=w)
            else:
                self.rotate_clockwise(w=w)



