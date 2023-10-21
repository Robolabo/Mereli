import numpy as np
from mereli.controllers import RobotController, BasicObstacleAvoider
from mereli.register import controller_registry

@controller_registry(name='test_wheels')
class TestWheelsController(RobotController):
    """
    """
    def step(self, state, reward=0.0):
        action_wheels = (1,-1)
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
        


@controller_registry(name='test_encoder')
class TestEncoderController(RobotController):
    """
    """
    def __init__(self, *args,  **kwargs):
        super(TestEncoderController, self).__init__(*args, **kwargs)
        self.estim_pos = np.zeros(2)
        self.estim_ori = 0.0
        self.dist_walked = 0

    def compute_odom(self,state):
        b = 0.2# Dist between wheels
        R = 0.2# Wheel radius
        enc = self.get_sensor_reading('encoder')
        dUl = R*enc[0]*.01
        dUr = R*enc[1] * .01
        d_theta = (dUr - dUl) / b 
        d_rho = (dUr + dUl) / 2
        r = np.tan(d_theta) * d_rho 
        d_rho2 = (r + b/2) * 2 * np.sin(d_theta / 2)
        self.estim_pos += d_rho2 * np.r_[np.cos(self.estim_ori + d_theta / 2),np.sin(self.estim_ori+ d_theta / 2)] 
        self.estim_ori += d_theta


    def step(self, state, reward=0.0):
        action_wheels = (1,0.5)
        self.compute_odom(state)
        real_pos = self.controller_owner.position[:2]
        error = self.estim_pos - real_pos
        print(f'Pos Estimation {self.estim_pos}, real={real_pos} and  error={error}' )


        # print(state['encoder'])
        # R = 0.00197
        # self.dist_walked += state['encoder'][0] * R
        # real_dist_walked = self.controller_owner.position[0]
        if real_pos[0] > 3.999:
            __import__('pdb').set_trace()
        # error = self.dist_walked - real_dist_walked
        # print(f'Distance walked {self.dist_walked}, real={real_dist_walked} and  error={error}' )
        self.get_actuator('joint_velocity_actuator').action = np.array(action_wheels)

@controller_registry(name='test_contact')
class TestContactController(RobotController):
    """
    """

    def step(self, state, reward=0.0):
        contact_st = self.get_sensor('contact_sensor').reading
        print(f'Reading of Contact Sensor is {contact_st}')
        self.get_actuator('joint_velocity_actuator').action = np.array([1, 1])



@controller_registry(name='test_proximity')
class TestProximityController(RobotController):
    """
    """
    def __init__(self, *args,  **kwargs):
        super(TestProximityController, self).__init__(*args, **kwargs)
        self.t = 0
        self.prox_data = []

    def step(self, state, reward=0.0):
        prox_read = self.get_sensor('distance_sensor').reading
        self.prox_data.append(prox_read[0])
        print(f'Reading of Proximity Sensor is {prox_read}')

        if self.t == 1500:
            import matplotlib.pyplot as plt 
            plt.plot(self.prox_data) 
            plt.show()
        self.t += 1
        self.get_actuator('joint_velocity_actuator').action = np.array([1, 1])

    def reset(self):
        self.t = 0


@controller_registry(name='test_light_sensor')
class TestLightSensorController(RobotController):
    """
    """
    def __init__(self, *args,  color='red', **kwargs):
        super(TestLightSensorController, self).__init__(*args, **kwargs)
        self.color = color
        self.ls_data = []

    def step(self, state, reward=0.0):
        ls_read = self.get_sensor_reading('red_light_sensor') 
        self.ls_data.append(ls_read[0])
        print(f'Reading of Proximity Sensor is {ls_read}')

        if self.t == 500:
            import matplotlib.pyplot as plt 
            plt.plot(self.ls_data) 
            plt.show()
        self.get_actuator('joint_velocity_actuator').action = np.array([1, 1])

    def reset(self):
        self.ls_data = []


# TestSwitchLightSensor
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


@controller_registry(name='test_battery')
class TestBattery(RobotController):
    """
    """
    def __init__(self, *args,  **kwargs):
        super(TestBattery, self).__init__(*args, **kwargs)

    def step(self, state, reward=0.0):
        battery_lv = self.get_sensor_reading('battery_sensor')
        rls = self.get_sensor_reading('red_light_sensor')
        print(battery_lv, max(rls))
        if max(rls) > 0.1 and battery_lv < 0.9:
            action_wheels = (0,0)
        else:
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
