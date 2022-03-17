import logging
import copy
import numpy as np
# from shapely.geometry import Point
from mereli.objects import WorldObject
from mereli.actuators.base_actuator import HighLevelActuator
from mereli.register import sensors, actuators, world_object_registry


@world_object_registry(name='robot')
class Robot(WorldObject):
    """
    Base class for the robot entity. Any robot should inherit from this class. 

    :param position: 
    :param orientation:
    :param str model_file: name of the file where the robot model is defined.

    :var dict sensors:
    :var dict actuators:
    :var dict planned_actions: 
    """
    def __init__(self, position, orientation,  *args, model_file='entities/epuck/epuck', **kwargs):
        super(Robot, self).__init__(model_file, position, orientation,\
                        static=False, luminous=False, tangible=True, \
                        *args, **kwargs)
        self._food = False
        if self.controllable:
            #* Initialize sensors and actuators according to controller requirements
            self.sensors = {k : s(self, **self.controller.enabled_sensors[k])\
                                for k, s in sensors.items()\
                                if k in self.controller.enabled_sensors.keys()}
            self.actuators = {k : a(self, **self.controller.enabled_actuators[k])\
                                for k, a in actuators.items()\
                                if k in self.controller.enabled_actuators.keys()}
        #* Communication system. Only used if receiver and transmitter sensors are used.
        self.comm_sys = None
        #* Storage for actions selected by the controllers to be fed to actuators
        self.planned_actions = {k : [None] for k in actuators.keys()}

        #* Current Reward perceived by the robot.
        self.reward_generator = None
        self.reward = np.array([0])
        # self.reset()

    def step(self, neighborhood, perturbations=None):
        """ Step method of the robots. 
        It is composed by the following main steps:

        1. Firstly steps and reads all the sensors in order to perceive the environment.
        2. The robot executes its controller in order to compute the
           actions based on the sensory information.
        3. The actions are stored as planned actions to be eventually executed.

        :param list neighborhood: list filled with the neighboring world objects.
        :param float reward: reward to be fed to the controller update rules, if any.
        :param list perturbations: list of ``PostProcessingPerturbation`` to apply 
            to the stimuli before controller step. If there are no perturbations
            to apply the paramter is ``None``.

        :returns: state and action tuple of the current timestep. Both of them are expressed as 
            a dict with the sensor/actuator name and the corresponding stimuli/action.
        """
        #* Sense environment surroundings.
        state = self.perceive(neighborhood)
        #* Add reward as a new state entry.
        state['reward'] = self.reward

        #* Apply perturbations to stimuli 
        if perturbations is not None:
            for pert in perturbations:
                state = pert(state, self)

        #* Apply communication system pre step (previous to controller) 
        if self.comm_sys is not None:
            state[self.comm_sys.rx_name] = self.comm_sys.step_pre(state[self.comm_sys.rx_name])

        #* Obtain actions using controller.
        actions = self.controller.step(state, reward=self.reward)

        #* Apply communication system pre step (previous to controller) 
        if self.comm_sys is not None:
            actions = self.comm_sys.step_post(actions)

        #* Plan actions for future execution
        self.plan_actions(actions)
        #* Convert again tx frame to dict for its use in the opt. algs. 
        if self.comm_sys is not None:
            actions[self.comm_sys.tx_name] = {**actions[self.comm_sys.tx_name].as_dict, **{'state' : self.comm_sys.comm_state_code}}
        #* Compute robot reward.
        if self.reward_generator is not None:
            self.reward = self.reward_generator(actions, state, self, neighborhood)
        # print(self.reward)
        return state, actions

    def plan_actions(self, actions):
        for actuator, action in actions.items():
            self.planned_actions[actuator] = (actuator == 'wheel_actuator')\
                    and [action, self.position, self.orientation]  or [action]

    def actuate(self, neighborhood):
        """ Executes the previously planned actions in order to be processed in the world.
        
        :param list neighborhood: list of neighoboring entities to be used by some high level 
            actuators.
        """
        for actuator_name, actuator in self.actuators.items():
            if issubclass(type(actuator), HighLevelActuator):
                actuator.step(*iter(self.planned_actions[actuator_name]), neighborhood)
            else:
                actuator.step(*iter(self.planned_actions[actuator_name]))

    def perceive(self, neighborhood):
        """
        Computes the observed stimuli by steping each of the active sensors one by one.
        
        :param list neighborhood:  list filled with the neighboring world objects.
        
        :returns: a ``dict`` with each sensor name as key and the sensor readings as value.
        """
        readings = {}
        # IR receiver reads both the received frame and the distance sensor measurement to 
        # optimize the simulation.
        if 'IR_receiver' in self.sensors:
            ir_reading = self.sensors['IR_receiver'].step(neighborhood)
            readings.update({'IR_receiver' : ir_reading[0], 'distance_sensor' : ir_reading[1]})
        for sensor_name, sensor in self.sensors.items():
            if sensor_name == 'IR_receiver':
                continue
            reading = sensor.step(neighborhood)
            if isinstance(reading, dict):
                readings.update(reading)
            else:
                readings[sensor_name] = reading
        return readings

    def reset(self, seed=None):
        """
        Resets the robot dynamics, sensors, actuators and controller. Position and orientation 
        can be randomly initialized or fixed. In the former case a seed can be specified.

        :param int seed: seed for random initialization.
        """
        self._food = False
        self.reward = np.array([0]) #* Current Reward perceived by the robot.
        if self.reward_generator is not None:
            self.reward_generator.reset()
        if self.controllable:
            # Check if new sensors or actuator has been enabled from the controller. If so, 
            # activate them.
            if not all(key in self.sensors.keys() for key in self.controller.enabled_sensors):
                self.sensors = {k : s(self, **self.controller.enabled_sensors[k])\
                            for k, s in sensors.items()\
                            if k in self.controller.enabled_sensors.keys()}
            if not all(key in self.actuators.keys() for key in self.controller.enabled_actuators):
                self.actuators = {k : a(self, **self.controller.enabled_actuators[k])\
                                for k, a in actuators.items()\
                                if k in self.controller.enabled_actuators.keys()}
            #* Reset controller
            self.controller.reset()
            #* Reset Actuators
            for actuator in self.actuators.values():
                if hasattr(actuator, 'reset'):
                    actuator.reset()
            #* Reset Sensors
            for sensor in self.sensors.values():
                if hasattr(sensor, 'reset'):
                    sensor.reset()
        #* Reset Comm Sys
        if self.comm_sys is not None:
            self.comm_sys.set_owner(self.id)
            self.comm_sys.reset()


    def add_communication(self, comm_sys):
        if comm_sys.tx_name not in self.actuators:
            raise Exception(logging.error('Trying to create communication system {}, but sensor '\
                '{} has not been enabled.'.format(type(comm_sys).__name__, comm_sys.tx_name)))
        # if comm_sys.rx_name not in self.sensors:
        #     raise Exception(logging.error('Trying to create communication system {}, but sensor '\
        #         '{} has not been enabled.'.format(type(comm_sys).__name__, comm_sys.rx_name)))
        self.comm_sys = comm_sys
        

    @property
    def food(self):
        """ Getter for the food attribute. It is a boolean attribute active if the robot stores food.
        """
        return self._food

    @food.setter
    def food(self, hasfood):
        """ Setter for the food attribute. It is a boolean attribute active if the robot stores food.
        """
        self._food = hasfood

@world_object_registry(name='minitaur')
class Minitaur(Robot):
    """ Class for the Minitaur robot. """
    def __init__(self, *args, **kwargs):
        super(Minitaur, self).__init__(*args, model_file='quadruped/minitaur', z_offset=0.5,**kwargs)


@world_object_registry(name='epuck')
class Epuck(Robot):
    """ Class for the Epuck. """
    def __init__(self, *args, **kwargs):
        super(Epuck, self).__init__(*args, model_file='entities/epuck/epuck.urdf.xacro', **kwargs)
        self.scaling = 1/4.13