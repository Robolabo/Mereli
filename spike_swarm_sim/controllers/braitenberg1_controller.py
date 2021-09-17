import numpy as np
from spike_swarm_sim.controllers import RobotController
from spike_swarm_sim.register import controller_registry

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
        action[0] = np.max(state['light_sensor'][[4, 5, 6, 7]])
        action[1] = np.max(state['light_sensor'][[0, 1, 2, 3]])
        if np.max(state['light_sensor']) == 0.0: # 
            action = np.array([.2, .2])
        return {'joint_velocity_actuator' : 2 * np.clip(action, a_min=-1, a_max=1)}


@controller_registry(name='Braitenberg2a')
class Braitenberg2A(RobotController):
    """ Controller based on the Braitenberg vehicle 2b. This vehicle reacts to incoming 
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
        action[0] = np.max(state['red_light_sensor'][[0, 1, 2, 3]])
        action[1] = np.max(state['red_light_sensor'][[4, 5, 6, 7]])
        if np.max(state['red_light_sensor']) == 0.0:
            action = np.array([1., 1.])
        return {'joint_velocity_actuator' : np.clip(2 * action, a_min=-1, a_max=1)}