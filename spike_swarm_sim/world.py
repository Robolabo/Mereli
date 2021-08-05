import logging
import time
from collections import deque
import numpy as np
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc


from spike_swarm_sim.objects import  Robot, LightSource, Wall
from spike_swarm_sim.register import controllers, world_objects, initializers, env_perturbations, rewards
from spike_swarm_sim.utils import (increase_time, mov_average_timeit, isinstance_of_any)
from spike_swarm_sim.globals import global_states
from .physics_engine import Engine3D, Engine2D


class MultiWorldWrapper:
    """ Wrapper class for paralellizing genotype evaluations. 
    
    :param int n_cpu: number of cores (and parallel simulations).
    :param float height: height in metres of the square arena.
    :param float width: width in metres of the square arena.
    :param float world_delay: deprecated, to be removed in next ver.

    .. todo:: #TODO: Extend to 2D worlds.
    """
    def __init__(self, n_cpu, height=10, width=10, world_delay=1):
        self.n_cpu = n_cpu
        self._worlds = [World3D(height=height, width=width, world_delay=1) for _ in range(n_cpu + 1)]

    def build_from_dict(self, world_dict, ann_topology=None):
        """Build all the created worlds from the config dicts. """
        for world in self._worlds:
            world.build_from_dict(world_dict, ann_topology=ann_topology)

    @property
    def all(self):
        """ Return all the worlds as a list. 

        :returns: list of World instances of length N_worlds + 1.
        """
        return self._worlds

    @property
    def robots(self):
        """ Return all the robots of the first world. """
        return self._worlds[0].robots

    def get_world(self, idx):
        """ 
        Get world by index. 
        
        :param int idx: index of the world queried within [0, N_worlds]

        :returns: World instance requested.
        """
        return self._worlds[idx]


class World(object):
    # language=rst
    """ Base class of the world or environment. This class is never used directly in an experiment but 
    any experiment environment/world must inherit from it. The main function of World classes is to act as
    containers and orchestrator of the simulation. It stores all the objects that have been instantiated, 
    calls the :py:meth:`spike_swarm_sim.objects.Robot.step` method of every robot in order to map states into 
    actions and communicates with the physics and render engines in order to simulate and visualize rigid object 
    realistic physics and collisions. 
    Currently only 2D and 3D square arenas are implemented. 

    :param Engine physics_engine: physics engine to be used.
    :param float height: height in metres of the square arena.
    :param float width: width in metres of the square arena.
    :param float world_delay: deprecated, to be removed in next ver.

    :var dict hierarchy: dictionary storing all the entities instantiated in the world.
    :var dict initializers: dictionary mapping groups of entities to initializers of the positions 
            and orientations
    :var dict env_perturbations: dictionary mapping object groups to environmental perturbations (``EnvironmentalPerturbation``) applied 
        to robot states or actions.

    Example::

    >>> # Example of an obstacle avoidance experiment with 5 robots in 3D. 
    >>> from spike_swarm_sim.world import World3D
    >>> from spike_swarm_sim.objects import Robot3D
    >>> from spike_swarm_sim.controllers import BasicObstacleAvoider
    >>> from spike_swarm_sim.utils.initializers import InitializerHandler, RandomUniformInitializer
    >>> n_robots = 5
    >>> world = World3D(height=10, width=10)
    >>> world_cfg = {
    >>>     "engine" : "3D",
    >>>     "world_delay" : 1,
    >>>     "height": 10,
    >>>     "width":  10,
    >>>     "objects" : {
    >>>        "robotA" : {
    >>>            "type" : "robot",
    >>>            "num_instances" : n_robots,
    >>>            "controller" : "basic_obstable_avoider",
    >>>            "sensors" : {
    >>>                "distance_sensor" : {"n_sectors" : 4, "range" : 1}
    >>>            },  
    >>>            "actuators" : {
    >>>                "joint_velocity_actuator" : {"joint_ids" : [0, 1], "max_velocity" : 13}
    >>>            },
    >>>            "initializers" : {
    >>>                "positions" : {"name" : "random_uniform",  "params" : {"low" : [-3, -3], "high" : [3, 3], "size" : 2}},
    >>>                "orientations" : {"name" : "random_uniform",  "params" : {"low":0, "high" : 6.28, "size" : 1}}
    >>>            },
    >>>            "perturbations" : {
    >>>            },
    >>>            "params" : {"trainable" : True}
    >>>        }
    >>>     }
    >>> }
    >>> world.build_from_dict(world_cfg)
    >>> world.connect()
    >>> world.reset()
    >>> while(True):
    >>>     state, action = world.step()

    """
    def __init__(self, physics_engine, height=10, width=10, world_delay=1):
        self.physics_engine = physics_engine
        self.height = height
        self.width = width
        self.world_delay = world_delay
        self.render = global_states.RENDER

        #* Dict storing all objects
        self.hierarchy = {}
        #* Dict mapping object names to object groups
        self.groups = {}
        #* Dict storing how objects should be initialized as a group.
        self.initializers = {}
        #* Dict mapping object groups to environmental perturbations
        self.env_perturbations = {}

        self.reward_generator = None
        self.prev_states = None
        self.prev_actions = None
        self.t = 0


    @increase_time
    @mov_average_timeit
    def step(self):
        # language=rst
        """ Step function of the world to run it one timestep. This method is must be executed at every step of 
        the simulation in order to iterate the physics and robot controllers.
        
        The main functions of the method are:

        * It computes the swarm rewards based on the previous action and states.
        * For each instantiated controllable entity, the :py:meth:`step` method is executed. This results in the partially observable state measured by the robot sensors and the corresponding 
          actions elaborated by the controller. These states and actions are python ``dict`` objects mapping sensor 
          and actuator names to numpy arrays of measured states and actions. The states and actions of all the robots 
          in the swarm are gathered as a numpy array of python ``dict`` objects (each corresponding to a robot).
            
          Example::

          >>> state_obj = {'distance_sensor' : np.array([0, 0, 0, 1]), 'ground_sensor' : np.array([0])}
          >>> action_obj = {'joint_velocity_actuator' : np.array([0.4, -0.1])}

        * Perturbations are applied to the planned actions. For example, a robot communication transmitter can 
          be broken and its action is, therefore, inhibited.
        * Actuators of the robots are executed with the actions planned by the controllers.
        * The render and physics engines are iterated. The render engine is iterated only if 
          :py:attr:`spike_swarm_sim.World.render` is ``True``. 

        :returns: A tuple with state and action numpy arrays of length equal to the number of robots. 
                  Each of these arrays contain python ``dict`` objects representing the states and actions of each controllable entity.
        """
        states = deque()
        actions = deque()
        pre_perturbations = []
        #* Compute rewards
        rewards = self.reward_generator(self.prev_actions, self.prev_states, info=self.hierarchy)\
                if self.t > 0 and self.reward_generator is not None else None

        #* Step controllers
        for idx, (obj_name, obj) in enumerate(self.controllable_objects.items()):
            if not issubclass(type(obj), Robot):
                obj.step(self.neighborhood(obj))
                continue
            if len(self.env_perturbations) > 0:
                pre_perturbations = [pert for pert in self.env_perturbations[self.group_of(obj_name)]\
                            if not pert.postprocessing and idx in pert.affected_robots]
            reward = rewards[idx] if rewards is not None and self.reward_generator is not None else None
            state_obj, action_obj = obj.step(self.neighborhood(obj), reward=reward, perturbations=pre_perturbations) #!
            # if self.reward_generator is not None:
            #     self.rewards[idx] = self.reward_generator(action_obj, state_obj, entity_name=obj_name, info=self.hierarchy)
            states.append(state_obj)
            actions.append(action_obj)
        states = np.stack(states)
        actions = np.stack(actions)

        #* Apply environmental perturbations (Postprocessing)
        if len(self.env_perturbations) > 0:
            for perturbation in tuple(self.env_perturbations.values())[0]:
                if perturbation.postprocessing:
                    states, actions = perturbation(states, actions, self.robots)

        #* Actuate
        for obj in self.controllable_objects.values():
            if obj.tangible:
                obj.actuate(self.hierarchy)

        #* Render and physics step.
        self.physics_engine.step_physics()
        if self.render:
            self.physics_engine.step_render()

        #* Retain prev states and actions to compute rewards.
        self.prev_states = states.copy()
        self.prev_actions = actions.copy()
        return states, actions

    def register_entity(self, name, obj, group=None):
        """ 
        Adds an object to the world registry. Assigns a unique identifier to the object.
        Additionally, if the object belongs to a group of world objects it also registers it.

        :param str name: name of the object.
        :param WorldObject obj: instance of the world object to be added.
        :param str group: name of the group to which obj belong to. If none a new group is created with
                           obj as unique element.
        """
        self.hierarchy.update({name : obj})
        #* Register group element
        if group is None:
            group = name
        if group in self.groups.keys():
            self.groups[group].append(name)
        else:
            self.groups[group] = [name]
        obj.group = group 

    def set_initializer(self, group_name, initializer_pos, initializer_ori=None):
        """ Bounds and registers entity initializers to groups of entities. 
        Therefore, every time that an experiment is reset, the settled initializers are used 
        to establish the positions and orientations (only if the entity is a robot) of 
        the world entities. 
        The use of this method is compulsory if the experiment is created without using 
        neither :py:meth:`spike_swarm_sim.World.build_from_dict` nor :py:meth:`spike_swarm_sim.World.add_group` 
        (see e.g. examples/basic_examples). 

        :param str group_name: name of the group of entities to be created.
        :param Initializer initializer_pos: initializer class of the positions of the group.
        :param Initializer initializer_ori: initializer class of the orientations of the group.
        """
        self.initializers[group_name] = {
            'positions' : initializer_pos,
            'orientations' : initializer_ori
        }

    def add_group(self, group_name, entity_cls,  initializer_pos, initializer_ori=None, controller=None):
        """ Adds entities belonging to a certain group to the world. For instance, it can create all the 
        homogeneous robots within a swarm.

        :param str group_name: name of the group of entities to be created.
        :param WorldObject entity_cls: precise class of the entities of the group.
        :param Initializer initializer_pos: initializer class of the positions of the group.
        :param Initializer initializer_ori: initializer class of the orientations of the group.
        :param Controller controller: controller (if any) of the entities of the group. If the entity is not 
                controllable then its content must be None
        """
        #* Create group intializers.
        self.initializers[group_name] = {
            'positions' : initializer_pos,
            'orientations' : initializer_ori
        }
        positions = self.initializers[group_name]['positions']()
        orientations = self.initializers[group_name]['orientations']()\
                        if initializer_ori is not None else 5*[[0,0,0]]
        for i, pos, ori in enumerate(zip(positions, orientations)):
            entity = entity_cls(pos, ori, controller=controller)
            ent_name = group_name + '_' + i
            self.add_entity(ent_name, entity_cls, pos, ori, controller=controller, group_name=group_name)
        
    def build_from_dict(self, world_dict, ann_topology=None):
        """ 
        Initializes all the entities and adds them to the world/environment using a ``dict`` structure as input.
        The ``world_dict`` fully defines the environment and the instatiated robots and the ``ann_topology`` 
        entirely establishes the ANN controller topology (if :py:class:`spike_swarm_sim.controllers.NeuralController` is used).
        For a dedicated description of the configuration files fields see `Configuration Files <configuration_files.html>`__ .

        :param dict world_dict: configuration ``dict`` of the environment (parameters, objects, ...).
        :param dict ann_topology:  configuration ``dict`` of the neural network.
        """
        engine = world_dict['engine']
        #TODO: esto implica que el tipo/generador de reward es igual para todos los robots.
        if ann_topology is not None and ann_topology.get('learning_rule', {}).get('reward') is not None:
            self.reward_generator = rewards.get(ann_topology.get('learning_rule', {}).get('reward'))()
        for obj_name, obj in world_dict['objects'].items():
            object_cls = world_objects[engine][obj['type']]
            #! Prov implementation for TFM regarding the task scheduler
            if object_cls.__name__ == 'TaskScheduler':
                world_obj = object_cls(**obj['params'])
                self.register_entity(obj_name + '_' + str(i), world_obj, group=obj_name)
                continue
            #* Create group intializers.
            self.initializers[obj_name] = {
                key : initializers[value['name']](obj['num_instances'], engine=engine, variable=key, **value['params']) 
                        for key, value in obj['initializers'].items()
            }
            #* Loop entities and add them to the world.
            #* Distinguish between robots and the other objects.
            entity_positions = self.initializers[obj_name]['positions']()
            if issubclass(object_cls, Robot):# or issubclass(object_cls, Robot3D):
                entity_orientations = self.initializers[obj_name]['orientations']()
                #* Add entities one by one at their position and orientation
                for i, (position, orientation) in enumerate(zip(entity_positions, entity_orientations)):
                    controller = None
                    if obj['controller'] is not None:
                        #* Create Controller and add sensors and actuators
                        controller_cls = controllers[obj['controller']]
                        controller = controller_cls()
                        controller.add_sensors_from_dict(obj['sensors'])
                        controller.add_actuators_from_dict(obj['actuators'])
                        if issubclass(controller_cls, controllers['neural_controller']):
                            controller.add_ann_from_dict(ann_topology)
                    robot = object_cls(position, orientation, controller=controller, **obj['params'])
                    self.register_entity(obj_name + '_' + str(i), robot, group=obj_name)

                #* Add perturbations (if any) to the robot states and actions (not physical perturbs)
                #* For example: inhibit a certain sensor reading or ignore some action of a robot.
                if len(obj['perturbations']) > 0:
                    object_perturbations = []
                    for pert_name, perturbations in obj['perturbations'].items():
                        if not isinstance(perturbations, list):
                            object_perturbations.append(env_perturbations[pert_name](obj['num_instances'], **perturbations))
                        else:
                            for i, pert in enumerate(perturbations):
                                object_perturbations.append(env_perturbations[pert_name](obj['num_instances'], **pert))
                    self.env_perturbations.update({obj_name : object_perturbations})
            else: #* Non robot objects
                controller_cls = controllers.get(obj.get('controller'))
                controller = controller_cls is not None and controller_cls() or None
                for i, position in enumerate(entity_positions):
                    world_obj = object_cls(position, [0, 0, 0], controller=controller, **obj['params'])
                    self.register_entity(obj_name + '_' + str(i), world_obj, group=obj_name)

    def reset(self, seed=None):
        """ Resets the world and all its objects. It also initializes
        the dynamics (positions, orientation, ...) of entities.

        :param int seed: seed to initialize at some known random state. If no seed is used then the 
                argument to be fed must be None
        """
        self.t = 0
        self.prev_states = None
        self.prev_actions = None
        if self.reward_generator is not None:
            self.reward_generator.reset()
        #* Initialize object dynamics.
        self.run_initializers(seed=seed)
        #* Reset objects
        for obj in self.hierarchy.values():
            obj.reset(seed=seed)

        for group_pert in self.env_perturbations.values():
            for pert in group_pert:
                pert.reset()

    def connect(self):
        """ Connect to the physics engine. """
        self.physics_engine.connect(self.hierarchy.values())

    def disconnect(self):
        """ Disconnect physics engine. """
        self.physics_engine.disconnect()

    def run_initializers(self, seed=None):
        """ Executes the initializers of the positions and orientations of each group of world entities.
        As all entities in a group are initialized jointly, initializers are associated to groups.
        
        :param int seed: seed to initialize to a known random state or None if no seed is used.
        """
        if seed is not None:
            np.random.seed(seed)
        for group in self.groups:
            if group in self.initializers.keys():
                group_initializer = self.initializers[group]
                group_elements = self.group_objects(group)
                #* Initialize positions
                if 'positions' in group_initializer.keys():
                    positions = group_initializer['positions']()
                    for pos, obj in zip(positions, group_elements):
                        obj.position = pos
                #* Initialize orientations
                if 'orientations' in group_initializer.keys():
                    orientations = group_initializer['orientations']()
                    for orientation, obj in zip(orientations, group_elements):
                        obj.orientation = orientation
        np.random.seed()

    def group_objects(self, group):
        """ List all the objects belonging to a group.

        :param str group: name of the group to be listed.
        :returns: List of WorldObjects belonging to the group.
        """
        return [self.hierarchy[element] for element in self.groups[group]]

    def group_of(self, obj_name):
        """ Get the group to which the object belongs. """
        return [key for key, group_members in self.groups.items() if obj_name in group_members][0]

    def entities(self, obj_type):
        """ Dict with all world objects of some object type (robot, light_source, ...).
        TODO: Not finished: 2D implementation pending
        """
        # if obj_type not in world_objects['3D'].keys():
        #     logging.warning('Wrong world object. Known world objects are: {}'.format(tuple(world_objects)))
        #     return {}
        obj_cls = world_objects['3D'][obj_type]
        return {name : obj for name, obj in self.hierarchy.items()\
                if isinstance(obj, obj_cls)}

    @property
    def robots(self):
        """ Dict with all robots. """
        return {name : obj for name, obj in self.hierarchy.items()\
            if issubclass(type(obj), Robot)}

    @property
    def lights(self):
        """ Dict with all light sources. """
        return {name : obj for name, obj in self.hierarchy.items()\
                if type(obj).__name__ in ['LightSource', 'LightSource3D']}
    @property
    def controllable_objects(self):
        """ Dict with all controllable objects (ie with a controller). """
        return {name : obj for name, obj in self.hierarchy.items()\
                if obj.controllable}
    @property
    def uncontrollable_objects(self):
        """ Dict with all uncontrollable objects. """
        return {name : obj for name, obj in self.hierarchy.items()\
                if not obj.controllable}
    @property
    def tangible_objects(self):
        """ Dict with all tangible objects. """
        return {name : obj for name, obj in self.hierarchy.items()\
                if obj.tangible}
    @property
    def intangible_objects(self):
        """ Dict with all intangible objects. """
        return {name : obj for name, obj in self.hierarchy.items()\
                if not obj.tangible}
    @property
    def moving_objects(self):
        """ Dict with all objects with movement capabilities. """
        return {name : obj for name, obj in self.hierarchy.items() if not obj.static}
    @property
    def static_objects(self):
        """ Dict with all static objects. """
        return {name : obj for name, obj in self.hierarchy.items() if obj.static}
    @property
    def luminous_objects(self):
        """ Dict with all luminous objects. """
        return {name : obj for name, obj in self.hierarchy.items() if obj.luminous}

#TODO implementar p.disconnect(). Ctx manager?
class World3D(World):
    """ World class of 3D bounded arenas. """
    def __init__(self, *args, **kwargs):
        super(World3D, self).__init__(Engine3D(), *args, **kwargs)
        #* Add world limits
        self.add_limiting_walls()

    def add_limiting_walls(self):
        """ Creates the limiting walls of the 3D arena. """
        self.register_entity('wall_side_up', Wall([self.width/2, 0, 1], [0, 0, np.pi/2], height=1,\
            width=self.width-1), group='side_wall')
        self.register_entity('wall_side_bottom', Wall([-self.width/2, 0, 1], [0, 0, np.pi/2], height=1,\
            width=self.width-1), group='side_wall')
        self.register_entity('wall_side_left', Wall([0, self.height/2, 1], [0, 0, -np.pi/2], height=self.height+1,\
             width=1), group='side_wall')
        self.register_entity('wall_side_right', Wall([0, -self.height/2, 1], [0, 0, -np.pi/2], height=self.height+1,\
            width=1), group='side_wall')

    def step(self):
        """ Step function of the world to run it one timestep.
        Steps all objects are stores the state and actions.
        It also renders new world.

        :returns: A tuple with state and action dicts. These dicts map sensor names with observed states 
            and actuator names to actions. 
        """
        states, actions = super().step()
        if self.render:
            for l in self.lights.values():
                if self.physics_engine.engine.readUserDebugParameter(self.physics_engine.gui_params['light_coverage']) % 2 == 0:
                    l.show_coverage()
                else:
                    l.hide_coverage()
        return states, actions

    def neighborhood(self, robot):
        """ 
        .. todo:: #TODO: Not finished
        Method that returns the list of neighboring world objects of a robot.
        An object is considered to be in the vicinity if it is contained in the ball
        of radius equal to:
            a) The maximum range of distance or comunication sensors if the object is a robot.
            b) The range of the light sensor if the object is a light source.
        If the object is none of the abovementioned entities, then it is always in the vicinity (for simplicity).

        :param Robot3D robot: The Robot object whose vicinity has to be computed.

        :returns: List of neighboring WorldObject.
        """

        neighbors = []
        #!
        if isinstance_of_any(robot, [LightSource]):
            return self.robots.values()
        if not issubclass(type(robot), Robot) or len(self.hierarchy) == 1:
            return neighbors
        max_robot_dist = None
        if len(self.robots) > 1:
            #! ---
            max_robot_dist = np.max([robot.sensors[sensor].range \
                            for sensor in ['IR_receiver', 'distance_sensor', 'RF_receiver'] \
                            if sensor in robot.sensors.keys()])
            #! ---
        #* Robots
        for obj in self.hierarchy.values():
            #!
            if issubclass(type(obj), Robot) and obj.id != robot.id:
                if max_robot_dist is not None and obj.id != robot.id:
                    if np.linalg.norm(obj.position - robot.position) <= max_robot_dist:
                        neighbors.append(obj)
            elif isinstance(obj, LightSource):
                # #! PROV
                # ls_sensor = {'LightSource' : 'light_sensor', 'LightSource3D' : 'light_sensor3D'}[type(obj).__name__]

                # ls_sensor = {'LightSource' : 'light_sensor', 'LightSource3D' : 'light_sensor3D'}
                # if ls_sensor not in robot.sensors:
                #     continue
                # if np.linalg.norm(obj.position - robot.position) <= robot.sensors[ls_sensor].range:
                    # neighbors.append(obj)
                neighbors.append(obj) #TODO: Esto esta simplificado. TODO FIX.
            else:
                neighbors.append(obj)
        return neighbors


class World2D(World):
    """ World class of 2D bounded arenas. """
    def __init__(self, *args, **kwargs):
        physics_engine = Engine2D()
        super(World2D, self).__init__(physics_engine, *args, **kwargs)
        self.add_limiting_walls()

    def add_limiting_walls(self):
        """ Creates the limiting walls of the 2D arena. """
        self.register_entity('wall_side_up', Wall([self.width/2, 0], np.pi/2, height=0.5,\
            width=self.width), group='side_wall')
        self.register_entity('wall_side_bottom', Wall([-self.width/2,0], np.pi/2, height=0.5,\
            width=self.width), group='side_wall')
        self.register_entity('wall_side_left', Wall([0, self.height/2], -np.pi/2, height=self.height-0.5,\
            width=0.5), group='side_wall')
        self.register_entity('wall_side_right', Wall([0, -self.height/2], -np.pi/2, height=self.height-0.5,\
            width=0.5), group='side_wall')

    def assign_unique_id(self):
        """ Returns a unique identifier to be assigned to a new entity. """
        obj_id = np.random.randint(1000)
        while(len(self.hierarchy) > 0 and obj_id in [obj.id for obj in self.hierarchy.values()]):
            obj_id = np.random.randint(1, 1000)
        return obj_id

    def register_entity(self, name, obj, group=None):
        """ Adds entity to the world registry. """
        obj.id = self.assign_unique_id()
        super().register_entity(name, obj, group=group)

    def neighborhood(self, robot):
        """ 
        .. todo:: #TODO: Not implemented yet
        Method that returns the list of neighboring world objects of a robot.
        An object is considered to be in the vicinity if it is contained in the ball
        of radius equal to:
            a) The maximum range of distance or comunication sensors if the object is a robot.
            b) The range of the light sensor if the object is a light source.
        If the object is none of the abovementioned entities, then it is always in the vicinity (for simplicity).
        
        :param Robot2D robot: The Robot object whose vicinity has to be computed.

        :returns: List of neighboring world objects.
        """
        return self.hierarchy
        import pdb; pdb.set_trace()
        neighbors = []
        if isinstance(robot, LightSource):
            return self.robots.values()
        if not isinstance(robot, Robot) or len(self.hierarchy) == 1:
            return neighbors
        max_robot_dist = None
        if len(self.robots) > 1:
            max_robot_dist = np.max([robot.sensors[sensor].range \
                            for sensor in ['IR_receiver', 'distance_sensor'] \
                            if sensor in robot.sensors.keys()])
        #* Robots
        for obj in self.hierarchy.values():
            if isinstance(obj, Robot) and obj.id != robot.id:
                if max_robot_dist is not None and obj.id != robot.id:
                    if np.linalg.norm(obj.position - robot.position) <= max_robot_dist:
                        neighbors.append(obj)
            elif isinstance(obj, LightSource) and 'light_sensor' in robot.sensors.keys():
                if np.linalg.norm(obj.position - robot.position) <= robot.sensors['light_sensor'].range:
                    neighbors.append(obj)
            else:
                neighbors.append(obj)
        return neighbors





class GymWorld(object):
    def __init__(self, env_name):
        pass