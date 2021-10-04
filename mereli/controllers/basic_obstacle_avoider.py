import numpy as np
from mereli.controllers import RobotController
from mereli.register import controller_registry

@controller_registry(name='basic_obstable_avoider')
class BasicObstacleAvoider(RobotController):
    """ Controller devoted to the obstacle avoidance task. This means that the controller 
    will read from the distance sensor, process the measurements and return joint velocity 
    actions required to avoid colliding with any other tangible entity. This controller is 
    currently hard coded for the Epuck robot.

    :param float sensitivity: value in [0, 1] that defines the threshold in the distance sensor reading 
        to interpret an obstacle detection. 
    """
    def __init__(self, *args, sensitivity=0.15, **kwargs):
        super(BasicObstacleAvoider, self).__init__(*args, **kwargs)
        self.sensitivity = sensitivity

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
        
        st_ds = state['distance_sensor']
        if any(st_ds[[0,1]] > self.sensitivity):
            # print('Turn Left')
            action = np.array([1., -1])
        elif any(st_ds[[6,7]] > self.sensitivity):
            # print('Turn Right')
            action = np.array([-1, 1.])
        else:
            # print('GO straight over')
            action = np.array([1., 1.])
        # Turn on the LED of the obstacle direction.
        if 'led_actuator' in self.enabled_actuators:
            led_action = st_ds > self.sensitivity
            return {'joint_velocity_actuator' : action, 'led_actuator' : led_action.astype(int)}
        else:    
            return {'joint_velocity_actuator' : action}