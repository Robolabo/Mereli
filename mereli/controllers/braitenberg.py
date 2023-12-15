import numpy as np
from mereli.controllers import RobotController
from mereli.register import controller_registry

@controller_registry(name='braitenberg2a')
class Braitenberg2A(RobotController):
    """ Controller based on the Braitenberg vehicle 2A. This vehicle reacts to incoming 
    light by escaping from the light source.
    """
    def __init__(self, *args, **kwargs):
        super(Braitenberg2A, self).__init__(*args, **kwargs)

    def step(self, state, reward=0.0):
        """ Method to execute once the controller program. It reads the current light sensor measurement, 
        and plans the action as follows:
        
        .. math::

            a_{wl} = \max\{ls_4, ls_5, ls_6, ls_7\}

            a_{wr} = \max\{ls_0, ls_1, ls_2, ls_3\}

        where :math`a_{wr}` and :math:`a_{wl}` are the right and left wheel actions and :math:`ls_i` 
        is the light sensor of the :math:`i`-th sector. If no light is detected at all, then the 
        actions are :math`a_{wr}=a_{wl}=0.2`.

        :param dict state: state with the sensor reading. The dict maps the reference name of the sensor to 
            the sensor np.ndarray reading.
        :param float reward: reward (if any). Not used in this controller.
        """
        action = np.zeros(2)
        ls = self.get_sensor_reading('red_light_sensor')
        action[0] = np.max([ls[4], ls[5], ls[6], ls[7]])
        action[1] = np.max([ls[0], ls[1], ls[2], ls[3]])
        # action[0] = 0.5 + (ls[0] / 2)
        # action[1] = 0.5 + (ls[7] / 2)
        if np.max(ls) == 0.0:
            action = np.array([1., 1.])
        self.get_actuator('joint_velocity_actuator').action = np.clip(action, a_min=-1, a_max=1)

@controller_registry(name='braitenberg2b')
class Braitenberg2B(RobotController):
    """ Controller based on the Braitenberg vehicle 2B. This vehicle reacts to incoming 
    light by following the light gradient.
    """
    def __init__(self, *args, **kwargs):
        super(Braitenberg2B, self).__init__(*args, **kwargs)

    def step(self, state, reward=0.0):
        """ Method to execute once the controller program. It reads the current light sensor measurement, 
        and plans the action as follows:
        
        .. math::

            a_{wr} = \max\{ls_4, ls_5, ls_6, ls_7\}

            a_{wl} = \max\{ls_0, ls_1, ls_2, ls_3\}

        where :math`a_{wr}` and :math:`a_{wl}` are the right and left wheel actions and :math:`ls_i` 
        is the light sensor of the :math:`i`-th sector. If no light is detected at all, then the 
        actions are :math`a_{wr}=a_{wl}=0.2`.

        :param dict state: state with the sensor reading. The dict maps the reference name of the sensor to 
            the sensor np.ndarray reading.
        :param float reward: reward (if any). Not used in this controller.
        """
        action = np.zeros(2)
        ls = self.get_sensor_reading('red_light_sensor')
        action[0] = 0.5 + (ls[7] / 2)
        action[1] = 0.5 + (ls[0] / 2)
        if np.max(ls) == 0.0: # 
            action = np.array([.2, .2])
        self.get_actuator('joint_velocity_actuator').action = np.clip(action, a_min=-1, a_max=1)

@controller_registry(name='braitenberg3c')
class Braitenberg3C(RobotController):
    def __init__(self, *args, **kwargs):
        super(Braitenberg3C, self).__init__(*args, **kwargs)

    def step(self, state, reward=0.0):
        action = np.zeros(2)
        ls = self.get_sensor_reading('red_light_sensor')
        action[1] = 1 - np.mean(ls[[0,1,2,3]])
        action[0] = 1 - np.mean(ls[[7,6,5,4]])
        # if np.max(state['red_light_sensor']) == 0.0: # 
        #     action = np.array([.2, .2])
        self.get_actuator('joint_velocity_actuator').action = np.clip(action, a_min=-1, a_max=1)
