import numpy as np
from mereli.controllers import RobotController
from mereli.register import controller_registry


@controller_registry(name='random_walk')
class RandomMovementController(RobotController):
    """ Extremely simple controller that just performs a robot random walk. 
    It requires no sensing information as just samples the velocities of 
    each joint from the uniform distribution :math:`\mathcal{U}(0, 1)`.
    """
    def __init__(self, *args, **kwargs):
        super(RandomMovementController, self).__init__(*args, **kwargs)
        
    def step(self, state, reward=0.0):
        """ Method to execute once the controller program. It samples the velocities of 
        each joint from the uniform distribution :math:`\mathcal{U}(0, 1)`.

        :param dict state: state with the sensor reading. The dict maps the reference name of the sensor to 
            the sensor np.ndarray reading. Not used in this controller.
        :param float reward: reward (if any). Not used in this controller.
        """
        return {'joint_velocity_actuator' : np.random.random(size=2)}