import numpy as np
from spike_swarm_sim.register import sensors, actuators, controller_registry
from spike_swarm_sim.utils import increase_time

class Controller:
    """ Base class for entity controllers. """
    def __init__(self):
        raise NotImplementedError
    
    def step(self, state):
        """ Base step method for executing once the controller. 
        Precise Controllers must overwrite this method with the corresponding 
        initialization."""
        raise NotImplementedError

    def reset(self):
        """ Base reset method for initializing the dynamical variables of the controller 
        (if any). Precise Controllers must overwrite this method with the corresponding 
        initialization."""
        pass

class RobotController(Controller):
    """ Base class for Robot Controllers. Any Controller devoted to the management of robot 
    actions must inherit from this class. The controller functionality must be implemented 
    within the overwritten ``Controller.step`` method. In order to use them, sensors and actuators 
    must be enabled by the controller first (controllers normally use only a subset of all available 
    sensors and actuators). Sensor can be activated through either ``add_sensor`` to enable them one 
    by one or using ``add_sensors_from_dict``, to activate all at once. Similarly, actuators can be 
    included using either ``add_actuator`` or ``add_actuators_from_dict``.
    
    :param Robot controller_owner: Instance of the robot owning the controller.

    :var dict enabled_sensors: dict with storing the sensors enabled by the controller. 
        It maps sensor reference names to a dict with the paramters of the sensor.
    :var dict enabled_actuators: dict with storing the actuators enabled by the controller. 
        It maps sensor reference names to a dict with the paramters of the actuators.
    """
    def __init__(self, controller_owner=None):
        self.controller_owner = controller_owner
        self.enabled_sensors = {}
        self.enabled_actuators = {}
    
    def add_sensors_from_dict(self, robot_sensors):
        """ Add the sensors that the controller can make use of in the form of a python dict.
        The dict structure must be {"sensor_name" : sensor_params}.
        Notice that, at this point, the sensor is not yet activated by the robot. In order to 
        activate it, the robot object has to be created afterwards or the ``reset`` method of the robot
        has to be called.

        :param dict robot_sensors: dict with all the sensors to be activated, mapping the sensor reference 
            name and another dict (subdict) with the sensor parameters.

        Example::

        >>> robot_sensors = {'distance_sensor' : {'n_sectors' : 8, 'range' : 1}}
        """
        self.enabled_sensors = {sensor : sensor_config for sensor, sensor_config in robot_sensors.items()}
    
    def add_actuators_from_dict(self, robot_actuators):
        """ Add the actuators that the controller can make use of in the form of a python dict. 
        Notice that, at this point, the actuator is not yet activated by the robot. In order to 
        activate it, the robot object has to be created afterwards or the ``reset`` method of the robot
        has to be called.

        The dict structure must be {"actuator_name" : actuator_params}.

        :param dict robot_actuators: dict with all the actuators to be activated, mapping the actuator reference 
            name and another dict (subdict) with the actuator parameters.
        
        Example::

        >>> robot_actuators = {'joint_velocity_actuator' : {'joint_ids' : [0,1], 'max_velocity' : 10}}
        """
        self.enabled_actuators = {actuator : actuator_config for actuator, actuator_config in robot_actuators.items()}


    def add_sensor(self, sensor_name, sensor_params):
        """ Adds a single sensor to the enabled sensors by the robot.             
        Notice that, at this point, the sensor is not yet activated by the robot. In order to 
        activate it, the robot object has to be created afterwards or the ``reset`` method of the robot
        has to be called.

        :param str sensor_name: reference name of the sensor to be enabled.
        :param dict sensor_name: dict with the parameters of the sensor to be enabled.
        """
        self.enabled_sensors.update({sensor_name : sensor_params})

    def add_actuator(self, actuator_name, actuator_params):
        """ Adds a single sensor to the enabled actuators by the robot.             
        Notice that, at this point, the sensor is not yet activated by the robot. In order to 
        activate it, the robot object has to be created afterwards or the ``reset`` method of the robot
        has to be called.

        :param str actuator_name: reference name of the actuator to be enabled.
        :param dict actuator_params: dict with the parameters of the actuator to be enabled.
        """
        self.enabled_actuators.update({actuator_name : actuator_params})





# @controller_registry(name='prey_controller')
# class PreyController(RobotController):
#     """ Class for the control of light sources mimicking prey escape. 
#     The controller deterministically computes the escape direction (steering) 
#     based on the known positions of the predators. If a predator is at a distance 
#     lower than 30 (3cm), then the prey is hunted and stops its motion.
#     """
#     def __init__(self,  *args, **kwargs):
#         super(PreyController, self).__init__(*args, **kwargs)
#         self.t = 1
#         self.direction = np.r_[np.cos(np.pi/4), np.sin(np.pi/4)]
#         self.hunted = 0

#     @increase_time
#     def step(self, state, rew=0.0):
#         import pdb; pdb.set_trace()
#         # new_pos = my_pos[:2].copy()
#         # if len(robot_positions) == 0:
#         #     return my_pos
#         # # robot_light_vecs = np.stack([toroidal_difference(robot_pos[:2], my_pos[:2]) for robot_pos in robot_positions])
#         # robot_light_vecs = np.stack([robot_pos[:2] - my_pos[:2] for robot_pos in robot_positions])
#         # distances = np.array([np.linalg.norm(v) for v in robot_light_vecs])
#         # near_robots = [d < 1 for d in distances]
#         # if any(near_robots):
#         #     weights = (5 - distances[near_robots]) / 5
#         #     weights /= sum(weights)
#         #     self.direction = -normalize(np.dot(weights, robot_light_vecs[near_robots]))
#         # if not self.hunted:
#         #     new_pos += 0.01 * self.direction
#         #     self.hunted = any([np.linalg.norm(v) < 0.2 for v in robot_light_vecs])
        
#         # if len(my_pos) == 3:
#         #     new_pos = np.r_[new_pos, my_pos[-1].copy()]

#         # #! luz no puede pasar de la pared.


#         # return new_pos

#     def reset(self):
#         self.t = 1
#         self.hunted = 0