import logging
import copy
import numpy as np
# from shapely.geometry import Point
from mereli.objects import WorldObject
from mereli.actuators.base_actuator import HighLevelActuator
from mereli.register import sensors, actuators, world_object_registry
from mereli.globals import global_states 


class Battery:
    def __init__(self, robot, discharge_coef=0.001, charge_coef=0.005, 
                 charge_range=0.3, init_level=1.0, discharge_only_moving=True, stop_wheels=False):
        self.robot = robot
        self.color = None 
        self.discharge_coef = discharge_coef 
        self.charge_coef = charge_coef 
        self.charge_range = charge_range 
        self.init_level = init_level if init_level != "random" else np.round(np.random.uniform(low=0.7, high=1.0),3)
        print(self.init_level)
        self.level = init_level 
        self.discharge_only_moving = discharge_only_moving 
        self.stop_wheels = stop_wheels 

    def step(self, lights):
        if len(lights) == 0:
            self.discharge()
            return
        lights = list(lights.values())
        clst_light_idx = np.argmin([np.linalg.norm(ent.position[:2] - self.robot.position[:2]) for ent in lights])
        clst_light = lights[clst_light_idx] 
        if np.linalg.norm(clst_light.position[:2] - self.robot.position[:2]) <= self.charge_range:
            self.charge()
        else:
            if self.discharge_only_moving:
                wheels = self.robot.actuators['joint_velocity_actuator'].action
                if wheels is not None:
                    if  abs(wheels[0]) > 0.1 or np.abs(wheels[1]) > 0.1:
                        self.discharge()
                # if wheels[0] > 0.1 or np.anwheels[]
            else:
                self.discharge()
        # print('BATTERY LEVEL: ', self.level)


    def charge(self):
        self.level +=  self.charge_coef * (1 - self.level ** 2)

    def discharge(self):
        self.level = max(self.level - self.discharge_coef, 0)  

    def reset(self):
        self.level = self.init_level 


class Map:
    def __init__(self):
        self.data = None
        self.current_tile = None
        self.symbols = {'O' : 1, 'F' : 0, 'B' : 2, 'G' : 3}

    def reset(self): 
        pass



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
        self.t = 0
        self.battery = None
        self.battery_enabled = False
        self._food = False
        self.sensors = {}
        self.actuators = {}
        if self.controllable:
            #* Initialize sensors and actuators according to controller requirements
            if len(self.controller.enabled_sensors) > 0:
                self.sensors = {k : s(self, **self.controller.enabled_sensors[k])\
                                    for k, s in sensors.items()\
                                    if k in self.controller.enabled_sensors.keys()}
            if len(self.controller.enabled_actuators) > 0:
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
        self.task = np.array([0])
        self.state = {}
        self.actions = {}
        # self.reset()
        self._neighbors = []
        self.static_neighbors = []
        self.awaken = False
        self.is_focussed = False
    
    def add_sensor(self, sensor_name, **kwargs): 
        if sensor_name in self.sensors or self.controller.is_sensor_enabled(sensor_name):
            # Sensor already enabled
            return

        self.controller.add_sensor(sensor_name, kwargs)
        sensor = sensors[sensor_name](self, **kwargs)
        self.sensors.update({sensor_name : sensor})
    def add_actuator(self, actuator_name, **kwargs): 
        if actuator_name in self.actuators or self.controller.is_actuator_enabled(actuator_name):
            # Actuator already enabled
            return

        self.controller.add_actuator(actuator_name, kwargs)
        actuator = actuators[actuator_name](self, **kwargs)
        self.actuators.update({actuator_name : actuator})


    def step(self):
        """ Step method of the robots. 
        It is composed by the following main steps:

        1. Firstly steps and reads all the sensors in order to perceive the environment.
        2. The robot executes its controller in order to compute the
           actions based on the sensory information.
        3. The actions are stored as planned actions to be eventually executed.
        """
        if not self.awaken:
            self.t += 1
            return
        #* Step sensors and update their sensor readings 
        self.perceive()
        state = {name : sensor.reading for name, sensor in self.sensors.items()}

        #* Add reward as a new state entry.
        state['reward'] = np.array([self.reward]).flatten()
        state['task'] = self.task

        #* Apply communication system pre step (previous to controller) 
        if self.comm_sys is not None:
            state[self.comm_sys.rx_name] = self.comm_sys.step_pre(state[self.comm_sys.rx_name])

        #* Obtain actions using controller.
        actions = self.controller.step(state, reward=self.reward)

        #* Apply communication system pre step (previous to controller) 
        if self.comm_sys is not None:
            actions = self.comm_sys.step_post(actions)

        ##* Plan actions for future execution
        #self.plan_actions(actions)
        #* Convert again tx frame to dict for its use in the opt. algs. 
        # if self.comm_sys is not None:
        #     actions[self.comm_sys.tx_name] = {**actions[self.comm_sys.tx_name].as_dict, **{'state' : self.comm_sys.comm_state_code}}

        #* Compute robot reward.
        # if self.reward_generator is not None:
        #     self.reward = self.reward_generator(actions, state, self, neighborhood)
        self.state = state
        self.actions = actions
        self.t += 1
        if self.battery_enabled:
            self.battery.step(self.static_neighbors)
            if self.battery.level == 0 and self.battery.stop_wheels and 'joint_velocity_actuator' in self.actuators:
                self.actuators['joint_velocity_actuator'].action = np.zeros(2)

    def plan_actions(self):
        for actuator, action in self.actions.items():
            self.planned_actions[actuator] = (actuator == 'wheel_actuator')\
                    and [action, self.position, self.orientation]  or [action]

    def actuate(self):
        """ Executes the previously planned actions in order to be processed in the world.
        """
        if not self.awaken:
            return
        for actuator_name, actuator in self.actuators.items():
            actuator.step()            
            # if self.planned_actions[actuator_name][0] is None:
            #     continue
            # if issubclass(type(actuator), HighLevelActuator):
            #     actuator.step(*iter(self.planned_actions[actuator_name]), neighborhood)
            # else:
            #     actuator.step(*iter(self.planned_actions[actuator_name]))

    def perceive(self):
        """
        Computes the observed stimuli by steping each of the active sensors one by one.
        
        :returns: a ``dict`` with each sensor name as key and the sensor readings as value.
        """
        if len(self.controller.enabled_sensors) == 0:
            return
        for sensor_name, sensor in self.sensors.items():
            # IRCommRX updates distance_sensor too 
            if 'IRCommRX' in self.sensors and sensor_name == 'distance_sensor':
                continue
            sensor.step()
            # if isinstance(reading, dict):
            #     readings.update(reading)
            # else:
            #     readings[sensor_name] = reading
        # return readings


    def reset(self, seed=None):
        """
        Resets the robot dynamics, sensors, actuators and controller. Position and orientation 
        can be randomly initialized or fixed. In the former case a seed can be specified.

        :param int seed: seed for random initialization.
        """
        super().reset(seed=seed) #init pos and orientation
        self.t = 0
        self.awaken = False
        self.state = {}
        self.actions = {}
        self._food = False
        self.reward = np.array([0]) #* Current Reward perceived by the robot.
        self.task = np.array([0])
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
        # Reset Battery
        if self.battery_enabled:
            self.battery.reset()

    def add_communication(self, comm_sys):
        if comm_sys.tx_name not in self.actuators:
            raise Exception(logging.error('Trying to create communication system {}, but sensor '\
                '{} has not been enabled.'.format(type(comm_sys).__name__, comm_sys.tx_name)))
        # if comm_sys.rx_name not in self.sensors:
        #     raise Exception(logging.error('Trying to create communication system {}, but sensor '\
        #         '{} has not been enabled.'.format(type(comm_sys).__name__, comm_sys.rx_name)))
        self.comm_sys = comm_sys
        
    def add_battery(self, **battery_kw):
        self.battery_enabled = True 
        self.battery = Battery(self, **battery_kw)
    


    def remove_battery(self):
        self.battery_enabled = False
    
    @property
    def neighbors(self):
        return self._neighbors

    @neighbors.setter
    def neighbors(self, neighbors):
        self._neighbors = neighbors

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
        self.scaling = 1/2 # 1/4.13

@world_object_registry(name='epuck_simple')
class EpuckSimple(Robot):
    """ Class for the Epuck. """
    def __init__(self, *args, **kwargs):
        super(EpuckSimple, self).__init__(*args, model_file='entities/epuck/epuck_simple.urdf', **kwargs)
        # self.scaling = 1/2 # 1/4.13
        
@world_object_registry(name='particle')
class Particle(Robot):
    """ Class for the Epuck. """
    def __init__(self, *args, **kwargs):
        super(Particle, self).__init__(*args, model_file=None, **kwargs)

    @property
    def vertices(self): 
        vertices = [[10,0],[-10,-10],[-10,10]]
        tf_mat = np.array([
             [np.cos(self.orientation), -np.sin(-self.orientation)],
             [np.sin(-self.orientation), np.cos(self.orientation)],
        ])
        return [list(self.position + np.array(vert).dot(tf_mat)) for vert in vertices]
