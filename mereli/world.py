import logging
import time
import copy
from collections import deque
import numpy as np
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc


from mereli.objects import  Robot, LightSource, Wall, Map
from mereli.physics_engines.pybullet_engine import PybulletEngine
from mereli.register import (controllers, world_objects, initializers, 
        env_perturbations, rewards, communication_systems, world_registry)
from mereli.utils import (increase_time, mov_average_timeit, isinstance_of_any)
from mereli.globals import global_states


def map_parser():
    file = 'mereli/models/maps/map1.txt'
    with open(file) as f:
        map_mat = np.array([[int(ch) if ch != '' else 0 for ch in line.split(';')[0].split(' ')] for line in f.readlines()])
    import pdb; pdb.set_trace()


class World(object):
    # language=rst
    """ Base class of the world or environment. This class is never used directly in an experiment but 
    any experiment environment/world must inherit from it. The main function of World classes is to act as
    containers and orchestrator of the simulation. It stores all the objects that have been instantiated, 
    calls the :py:meth:`mereli.objects.Robot.step` method of every robot in order to map states into 
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
    >>> from mereli import SquareArena
    >>> from mereli.physics_engines import PybulletEngine
    >>> from mereli.objects import Robot3D
    >>> from mereli.controllers import BasicObstacleAvoider
    >>> from mereli.utils.initializers import InitializerHandler, RandomUniformInitializer
    >>>
    >>> n_robots = 5
    >>> phy_engine = PybulletEngine(dt=0.02)
    >>> world = SquareArena(phy_engine, height=10, width=10)
    >>> world_cfg = {
    >>>     "world_delay" : 1,
    >>>     "height": 10,
    >>>     "width":  10,
    >>>     "objects" : {
    >>>        "robotA" : {
    >>>            "type" : "robot",
    >>>            "num_instances" : n_robots,
    >>>            "controller" : "basic_obstable_avoider",
    >>>            "sensors" : {
    >>>                "distance_sensor" : {"n_sectors" : 8, "range" : 1}
    >>>            },  
    >>>            "actuators" : {
    >>>                "joint_velocity_actuator" : {"joint_ids" : [0, 1], "max_velocity" : 5}
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
    >>> with world:
    >>>     while(True):
    >>>         state, action = world.step()

    """
    def __init__(self, physics_engine):
        self.physics_engine = physics_engine
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
          :py:attr:`mereli.World.render` is ``True``. 

        :returns: A tuple with state and action numpy arrays of length equal to the number of robots. 
                  Each of these arrays contain python ``dict`` objects representing the states and actions of each controllable entity.
        """
        states = deque()
        actions = deque()
        pre_perturbations = []

        #* Step controllers
        for idx, (obj_name, obj) in enumerate(self.controllable_objects.items()):
            if not issubclass(type(obj), Robot):
                obj.step(self.hierarchy.values())
                continue
            if len(self.env_perturbations) > 0:
                pre_perturbations = [pert for pert in self.env_perturbations[self.group_of(obj_name)]\
                            if not pert.postprocessing and idx in pert.affected_robots]
            #* Compute robot reward 
            reward = self.reward_generator(self.prev_actions, self.prev_states, obj, info=self.hierarchy.values())\
                    if self.reward_generator is not None else None
            state_obj, action_obj = obj.step(self.hierarchy.values(), reward=reward, perturbations=pre_perturbations) #!
            # if self.reward_generator is not None:
            #     self.rewards[idx] = self.reward_generator(action_obj, state_obj, entity_name=obj_name, info=self.hierarchy)
            states.append(state_obj)
            actions.append(action_obj)
        if len(states) > 0:
            states = np.stack(states)
        if len(actions) > 0:
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
        entirely establishes the ANN controller topology (if :py:class:`mereli.controllers.NeuralController` is used).
        For a dedicated description of the configuration files fields see `Configuration Files <configuration_files.html>`__ .

        :param dict world_dict: configuration ``dict`` of the environment (parameters, objects, ...).
        :param dict ann_topology:  configuration ``dict`` of the neural network.
        """
        #TODO: esto implica que el tipo/generador de reward es igual para todos los robots.
        if ann_topology is not None and ann_topology.get('learning_rule', {}).get('reward') is not None:
            self.reward_generator = rewards.get(ann_topology.get('learning_rule', {}).get('reward'))()
        for obj_name, obj in world_dict['objects'].items():
            object_cls = world_objects[self.physics_engine.engine_type][obj['type']]
            #! Prov implementation for TFM regarding the task scheduler
            if object_cls.__name__ == 'TaskScheduler':
                world_obj = object_cls(None, np.zeros(3), np.zeros(3), **obj['params']) #! ojo 2D
                self.register_entity(obj_name + '_' + str(i), world_obj, group=obj_name)
                continue
            #* Create group intializers.
            self.initializers[obj_name] = {
                key : initializers[value['name']](obj['num_instances'], engine=self.physics_engine.engine_type, 
                        variable=key, **value['params']) for key, value in obj['initializers'].items()
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
                    #* Add communication system (if any)
                    if "comm_sys" in obj:
                        robot.add_communication(communication_systems[obj['comm_sys']['name']](**obj['comm_sys']['params']))
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

    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()

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
                if 'positions' in group_initializer:
                    positions = group_initializer['positions']()
                    for pos, obj in zip(positions, group_elements):
                        obj.position = pos
                #* Initialize orientations
                if 'orientations' in group_initializer and group_initializer['orientations'] is not None:
                    orientations = group_initializer['orientations']()
                    for orientation, obj in zip(orientations, group_elements):
                        obj.orientation = orientation
        if seed is not None:
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
                if type(obj).__name__ in ['LightSource']}
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

    def set_camera_focus(self, obj, distance):
        """  Sets the focus of the camera of the render engine on the position of an entity.

        :param WorldObject obj: focused entity.
        :param float distance: distance between the entity and the camera.
        """
        self.physics_engine.set_camera_focus(obj.position, distance)


@world_registry(name='square_arena')
class SquareArena(World):
    """ World class for square arenas of a given height and width. 

    :param float width: width of the square arena in meters.
    :param float height: height of the square arena in meters.
    """
    def __init__(self, *args, width=10, height=10, **kwargs):
        super(SquareArena, self).__init__(*args, **kwargs)
        self.height = height
        self.width = width
        #* Add world limits
        self.__add_limiting_walls()

    def __add_limiting_walls(self):
        """ Private method for customizing the size of the limiting walls of the arena. 
        It creates the wall objects individually. They are stored under the group 'side_wall'.
        """
        self.register_entity('wall_side_up', Wall([self.width/2, 0, .5], [0, 0, np.pi/2], height=0.5,\
            width=self.width-.5), group='side_wall')
        self.register_entity('wall_side_bottom', Wall([-self.width/2, 0, .5], [0, 0, np.pi/2], height=.5,\
            width=self.width-.5), group='side_wall')
        self.register_entity('wall_side_left', Wall([0, self.height/2, .5], [0, 0, -np.pi/2], height=self.height+.5,\
             width=.5), group='side_wall')
        self.register_entity('wall_side_right', Wall([0, -self.height/2, .5], [0, 0, -np.pi/2], height=self.height+.5,\
            width=.5), group='side_wall')


@world_registry(name='circular_arena')
class CircularArena(World):
    """ World class for environments with an empty circular arena. 

    :param float radius: radius of the circular wall contraining the arena.
    """
    def __init__(self, *args, radius=5.0, **kwargs):
        super(CircularArena, self).__init__(*args, **kwargs)
        self.radius = radius
        self.__resize_circle_arena()
        self.register_entity('map', Map('circle_arena/circle_arena', np.zeros(3), np.zeros(3)), group='maps')
        
    def __resize_circle_arena(self):
        """ Private method for customizing the radius of the circular arena. 
        It reads the 3D obj mesh file, modifies the vertex positions and saves the file with the changes.
        """
        file = 'mereli/models/maps/circle_arena/circle_arena.obj'
        with open(file, "r") as f:
            lines = f.readlines()
            vertices = []
            for line in lines:
                elems = line.rstrip('\n').split(' ')
                if elems[0] == 'v':
                    vert = np.array(elems[1:]).astype(float)
                    vertices.append(vert)
            vertices = np.vstack(vertices)
            old_rads = np.unique(np.sqrt(vertices[:,0] ** 2 + vertices[:,2] ** 2).round(3))
            assert len(old_rads) == 2
            scaling = self.radius / old_rads.min()
            vertices[:,[0,2]] *= scaling
        with open(file, "r+") as f:
            lines = f.readlines()
            f.seek(0)
            vert_iter = iter(vertices)
            lines = ['v {} {} {}\n'.format(*tuple(next(vert_iter))) if line.split(' ')[0] == 'v' else line for line in lines ]
            f.writelines(lines)
            f.truncate()


@world_registry(name='custom_world')
class CustomWorld(World):
    """ World class for environments with custom map. The map is defined by means of a previously 
    defined and stored URDF file (with the corresponding obj files). The map file must be stored 
    in the folder 'mereli/models/maps/'.

    :param str model_file: path to the URDF file defining the map. It is relative to 'mereli/models/maps/' 
        and the file extension is not required
    """
    def __init__(self, *args, map_file='simple_map_1/simple_map_1', **kwargs):
        super(CustomWorld, self).__init__(*args, **kwargs)
        self.map_file = map_file
        self.register_entity('map', Map(self.map_file, np.zeros(3), np.zeros(3)), group='maps')


class MultiWorldWrapper:
    """ Wrapper class for parallelizing genotype evaluations. 
    
    :param int n_cpu: number of cores (and parallel simulations).
    :param World world: created instance of world or environment to be 
        cloned and parallelized.

    .. todo:: #TODO: Needs to be revisited!
    """
    def __init__(self, n_cpu, world):
        self.n_cpu = n_cpu
        #! Mucho ojo. Son objetos totalmente desacoplados?
        self._worlds = [SquareArena(PybulletEngine(), height=7, width=7) for _ in range(n_cpu + 1)]
        # self._worlds = [copy.deepcopy(world)] * (n_cpu + 1)

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

# class World2D(World):
#     """ World class of 2D bounded arenas. """
#     def __init__(self, *args, **kwargs):
#         physics_engine = Engine2D()
#         super(World2D, self).__init__(physics_engine, *args, **kwargs)
#         self.add_limiting_walls()

#     def add_limiting_walls(self):
#         """ Creates the limiting walls of the 2D arena. """
#         self.register_entity('wall_side_up', Wall([self.width/2, 0], np.pi/2, height=0.5,\
#             width=self.width), group='side_wall')
#         self.register_entity('wall_side_bottom', Wall([-self.width/2,0], np.pi/2, height=0.5,\
#             width=self.width), group='side_wall')
#         self.register_entity('wall_side_left', Wall([0, self.height/2], -np.pi/2, height=self.height-0.5,\
#             width=0.5), group='side_wall')
#         self.register_entity('wall_side_right', Wall([0, -self.height/2], -np.pi/2, height=self.height-0.5,\
#             width=0.5), group='side_wall')

#     def assign_unique_id(self):
#         """ Returns a unique identifier to be assigned to a new entity. """
#         obj_id = np.random.randint(1000)
#         while(len(self.hierarchy) > 0 and obj_id in [obj.id for obj in self.hierarchy.values()]):
#             obj_id = np.random.randint(1, 1000)
#         return obj_id

#     def register_entity(self, name, obj, group=None):
#         """ Adds entity to the world registry. """
#         obj.id = self.assign_unique_id()
#         super().register_entity(name, obj, group=group)

class GymWorld(object):
    def __init__(self, env_name):
        pass