import numpy as np
from spike_swarm_sim.register import sensors, actuators, controller_registry
from spike_swarm_sim.utils import increase_time

class Controller:
    """ Base class for entity controllers. """
    def __init__(self):
        raise NotImplementedError
    
    def step(self, state):
        raise NotImplementedError

    def reset(self):
        pass

class RobotController(Controller):
    """ Base class for Robot Controllers. """
    def __init__(self, controller_owner=None):
        self.controller_owner = controller_owner
        self.enabled_sensors = {} #{sensor : sensor_config for sensor, sensor_config in robot_sensors.items()}
        self.enabled_actuators = {} #{actuator : actuator_config for actuator, actuator_config in robot_actuators.items()}
    
    def add_sensors_from_dict(self, robot_sensors):
        """ Add the sensors that the controller can make use of in the form of a python dict.
            The dict structure must be {"sensor_name" : sensor_params}.
            Example:
                robot_sensors = {'distance_sensor' : {'n_sectors' : 4, 'range' : 1}}
        """
        self.enabled_sensors = {sensor : sensor_config for sensor, sensor_config in robot_sensors.items()}
    
    def add_actuators_from_dict(self, robot_actuators):
        """ Add the actuators that the controller can make use of in the form of a python dict.
            The dict structure must be {"actuator_name" : actuator_params}.
            Example:
                robot_actuators = {'joint_velocity_actuator' : {'joint_ids' : [0,1], 'max_velocity' : 10}}
        """
        self.enabled_actuators = {actuator : actuator_config for actuator, actuator_config in robot_actuators.items()}


    def add_sensor(self, sensor_name, sensor_params):
        pass

    def add_actuator(self, actuator_name, actuator_params):
        pass

@controller_registry(name='prey_controller')
class PreyController(RobotController):
    """ Class for the control of light sources mimicking prey escape. 
    The controller deterministically computes the escape direction (steering) 
    based on the known positions of the predators. If a predator is at a distance 
    lower than 30 (3cm), then the prey is hunted and stops its motion.
    """
    def __init__(self,  *args, **kwargs):
        super(PreyController, self).__init__(*args, **kwargs)
        self.t = 1
        self.direction = np.r_[np.cos(np.pi/4), np.sin(np.pi/4)]
        self.hunted = 0

    @increase_time
    def step(self, state, rew=0.0):
        import pdb; pdb.set_trace()
        # new_pos = my_pos[:2].copy()
        # if len(robot_positions) == 0:
        #     return my_pos
        # # robot_light_vecs = np.stack([toroidal_difference(robot_pos[:2], my_pos[:2]) for robot_pos in robot_positions])
        # robot_light_vecs = np.stack([robot_pos[:2] - my_pos[:2] for robot_pos in robot_positions])
        # distances = np.array([np.linalg.norm(v) for v in robot_light_vecs])
        # near_robots = [d < 1 for d in distances]
        # if any(near_robots):
        #     weights = (5 - distances[near_robots]) / 5
        #     weights /= sum(weights)
        #     self.direction = -normalize(np.dot(weights, robot_light_vecs[near_robots]))
        # if not self.hunted:
        #     new_pos += 0.01 * self.direction
        #     self.hunted = any([np.linalg.norm(v) < 0.2 for v in robot_light_vecs])
        
        # if len(my_pos) == 3:
        #     new_pos = np.r_[new_pos, my_pos[-1].copy()]

        # #! luz no puede pasar de la pared.


        # return new_pos

    def reset(self):
        self.t = 1
        self.hunted = 0