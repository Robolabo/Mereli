from spike_swarm_sim.register import sensors, actuators

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
    def __init__(self, robot_sensors, robot_actuators, controller_owner=None):
        self.controller_owner = controller_owner
        self.enabled_sensors = {sensor : sensor_config for sensor, sensor_config in robot_sensors.items()}
        self.enabled_actuators = {actuator : actuator_config for actuator, actuator_config in robot_actuators.items()}