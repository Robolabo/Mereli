import logging
import time
from collections import deque
import numpy as np
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc


from spike_swarm_sim.objects import  Robot3D, LightSource, Wall
from spike_swarm_sim.objectives.reward import GoToLightReward
from spike_swarm_sim.register import controllers, world_objects, initializers, env_perturbations
from spike_swarm_sim.utils import angle_diff, compute_angle, normalize, increase_time, mov_average_timeit
from spike_swarm_sim.globals import global_states
from .physics_engine import PybulletEngine

class MultiWorldWrapper:
    def __init__(self, n_cpu, height=10, width=10, world_delay=1):
        self.n_cpu = n_cpu
        self._worlds = [World3D(height=height, width=width, world_delay=1) for _ in range(n_cpu+1)]

    def build_from_dict(self, world_dict, ann_topology=None):
        for world in self._worlds:
            world.build_from_dict(world_dict, ann_topology=ann_topology)

    @property
    def all(self):
        return self._worlds 
    
    def get_world(self, idx):
        return self._worlds[idx]


#! OJO implementar p.disconnect(). Ctx manager?
class World3D(object):
    def __init__(self, height=10, width=10, world_delay=1):
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

        #* Engine
        self.physics_engine = PybulletEngine()
      

        #* Add world limits
        self.add_limiting_walls()
        # self.reward_generator = GoToLightReward()
        self.t = 0
        self.aux = 0

    def add_limiting_walls(self):
        self.add('wall_side_up', Wall([self.width/2, 0, 1], [0, 0, np.pi/2], height=1,\
            width=self.width-1, physics_client=self.physics_engine.engine), group='side_wall')
        self.add('wall_side_bottom', Wall([-self.width/2, 0, 1], [0, 0, np.pi/2], height=1,\
            width=self.width-1, physics_client=self.physics_engine.engine), group='side_wall')
        self.add('wall_side_left', Wall([0, self.height/2, 1], [0, 0, 0], height=self.height+1,\
             width=1, physics_client=self.physics_engine.engine), group='side_wall')
        self.add('wall_side_right', Wall([0, -self.height/2, 1], [0, 0, 0], height=self.height+1,\
            width=1, physics_client=self.physics_engine.engine), group='side_wall')
        

    @increase_time
    @mov_average_timeit
    def step(self):
        """ Step function of the world to run it one timestep.
        Steps all objects are stores the state and actions.
        It also renders new world.
        ======================================================
        - Args: None
        - Returns:
            A tuple with state and action dicts.
        ======================================================
        """
        states = deque()
        actions = deque()
        pre_perturbations = []
        reward = 0.0
        #* Step controllers
        for idx, (n, obj) in enumerate(self.controllable_objects.items()):
            if not isinstance(obj, Robot3D):
                obj.step(self.neighborhood(obj))
                continue
            rew = 0
            if len(self.env_perturbations) > 0:
                pre_perturbations = [pert for pert in tuple(self.env_perturbations.values())[0]\
                            if not pert.postprocessing and idx in pert.affected_robots]
            state_obj, action_obj = obj.step(self.neighborhood(obj), reward=rew, perturbations=pre_perturbations) #!
            # reward = self.reward_generator(action_obj, state_obj) if isinstance(obj, Robot) else None
            states.append(state_obj)
            actions.append(action_obj)
        states = np.stack(states)
        actions = np.stack(actions)
        
        #* Apply environmental perturbations (Postprocessing)
        if len(self.env_perturbations) > 0:
            for perturbation in tuple(self.env_perturbations.values())[0]:
                if perturbation.postprocessing:
                    states, actions = perturbation(states, actions, self.robots)

        #! Actuate
        for obj in self.controllable_objects.values():
            if obj.tangible:
                obj.actuate()

        # #* Apply mirror
        # for robot in self.hierarchy.values(): #! OJO fall en las esquinas
        #     if  robot.controllable:
        #         robot.pos[robot.pos > 1000+15] = 20
        #         robot.pos[robot.pos < -13] = 1000-20


        #* Render and physics step.
        self.physics_engine.step_physics()
        if self.render:
            self.physics_engine.step_render()
        return states, actions
    
    def add(self, name, obj, group=None):
        """ Adds an object to the world registry. Assigns a unique identifier to the object.
        Additionally, if the object belongs to a group of world objects it also registers it.
        ============================
        - Args:
            name [str] -> name of the object.
            obj [WorldObject] -> instance of the world object to be added.
            group [str] -> name of the group to which obj belong to. If none a new group is created with
                           obj as unique element.
        - Returns: None
        ============================
        """
        self.hierarchy.update({name : obj})
        #* Register group element
        if group is None:
            group = name
        if group in self.groups.keys():
            self.groups[group].append(name)
        else:
            self.groups[group] = [name]

    
    def build_from_dict(self, world_dict, ann_topology=None):
        """ Initialize all objects and add them into the world using a dictionary structure.
        =========================================================================================
        - Args:
            world_dict [dict] : configuration dict of the environment (parameters, objects, ...).
            ann_topology [dict] :  configuration dict of the neural network.
        - Returns: None
        =========================================================================================
        """
        engine = world_dict['engine']
        for obj_name, obj in world_dict['objects'].items():
            #* Create group intializer.
            self.initializers[obj_name] = {
                key :  initializers[value['name']](obj['num_instances'], **value['params'])\
                        for key, value in obj['initializers'].items()
            }
            if obj['type'] == 'robot':
                object_cls = world_objects[engine][obj['type']]
                robot_positions = map(lambda x: (x[0], x[1], 0.), self.initializers[obj_name]['positions']())
                robot_orientations = map(lambda x: (0., 0., x[0]), self.initializers[obj_name]['orientations']())
                
                #* Add robots one by one at their position and orientation
                for i, (position, orientation) in enumerate(zip(robot_positions, robot_orientations)):
                    if obj['controller'] is not None:
                        controller_cls = controllers[obj['controller']]
                        if obj['controller'] in ['neural_controller', 'cascade_controller']: # assuming only single ANN controller
                            controller = controller_cls(ann_topology, obj['sensors'], obj['actuators'])
                        else: # non-trainable robot controllers
                            controller = controller_cls(obj['sensors'], obj['actuators'])
                    else:
                        controller = None
                    robot = object_cls(position, orientation, controller=controller, physics_client=self.physics_engine.engine, **obj['params'])
                    self.add(obj_name + '_' + str(i), robot, group=obj_name)
                if len(obj['perturbations']) > 0:
                    self.env_perturbations.update({obj_name : [env_perturbations[pert](obj['num_instances'], **pert_params)\
                            for pert, pert_params in obj['perturbations'].items()]})
            else: # Non robot objects
                positions = self.initializers[obj_name]['positions']()
                controller = controllers[obj['controller']]() if obj['controller'] is not None else None
                for position in positions:
                    world_obj = object_cls(position, controller=controller,\
                                    physics_client=self.physics_engine.engine, **obj['params'])
                    self.add(obj_name + '_' + str(i), world_obj, group=obj_name)

    def group_objects(self, group):
        """ List all the objects belonging to a group.
        ==================================================
        - Args:
            group [str] -> name of the group to be listed.
        - Returns:
            List of WorldObjects belonging to the group.
        ==================================================
        """
        return [self.hierarchy[element] for element in self.groups[group]]
            
    def run_initializers(self, seed=None):
        """ Executes the initialization procedures of each group of objects.
        As all obj in a group are initialized jointly, initializers are associated to groups.
        =====================================================================================
        - Args:
            seed [int] -> seed to initialize to a known random state.
        - Returns: None
        =====================================================================================
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
                    positions = map(lambda x: (x[0], x[1], 0.), positions)
                    for pos, obj in zip(positions, group_elements):
                        obj.position = pos
                #* Initialize orientations
                if 'orientations' in group_initializer.keys():
                    orientations = group_initializer['orientations']()
                    orientations = map(lambda x: (0., 0., x[0]), orientations)
                    for orientation, obj in zip(orientations, group_elements):
                        obj.orientation = orientation
        np.random.seed()


    def reset(self, seed=None):
        """ Resets the world and all its objects. It also initalizes
        the dynamics (pos, orientation, ...) of objects.
        ================================================================
        - Args:
            seed [int] -> seed to initialize at some known random state.
        - Returns: None
        ================================================================
        """
        self.t = 0
        self.aux = 0
        #* Initialize object dynamics.
        self.run_initializers(seed=seed)
        #* Reset objects
        for obj in self.hierarchy.values():
            obj.reset()
           
        for group_pert in self.env_perturbations.values():
            for pert in group_pert:
                pert.reset()

    def disconnect(self):
        self.physics_engine.disconnect

    def connect(self):
        self.physics_engine.connect(self.hierarchy.values())
       
    def neighborhood(self, robot):
        """ Method that returns the list of neighboring world objects of a robot.
        An object is considered to be in the vicinity if it is contained in the ball
        of radius equal to:
            a) The maximum range of distance or comunication sensors if the object is a robot.
            b) The range of the light sensor if the object is a light source.
        If the object is none of the abovementioned entities, then it is always in the vicinity (for simplicity).
        =========================================================================================================
        - Args:
            robot -> The Robot object whose vicinity has to be computed.
        - Returns:
            List of neighboring world objects.
        =========================================================================================================
        """
        neighbors = []
        #!
        if isinstance(robot, LightSource):
            return self.robots.values()
        if not isinstance(robot, Robot3D) or len(self.hierarchy) == 1:
            return neighbors
        max_robot_dist = None
        if len(self.robots) > 1:
            max_robot_dist = np.max([robot.sensors[sensor].range \
                            for sensor in ['wireless_receiver', 'distance_sensor3D'] \
                            if sensor in robot.sensors.keys()])
        #* Robots
        for obj in self.hierarchy.values():
            #!
            if isinstance(obj, Robot3D) and obj.id != robot.id:
                if max_robot_dist is not None and obj.id != robot.id:
                    if np.linalg.norm(obj.position - robot.position) <= max_robot_dist:
                        neighbors.append(obj)
            elif isinstance(obj, LightSource) and 'light_sensor' in robot.sensors.keys():
                if np.linalg.norm(obj.position - robot.position) <= robot.sensors['light_sensor'].range:
                    neighbors.append(obj)
            else:
                neighbors.append(obj)
        return neighbors

    def world_objects(self, obj_type):
        """ Dict with all world objects of some object type (robot, light_source, ...).
        """
        if obj_type not in world_objects.keys():
            logging.warning('Wrong world object. Known world objects are: {}'.format(tuple(world_objects)))
            return {}
        obj_cls = type(world_objects[obj_type])
        return {name : obj for name, obj in self.hierarchy.items()\
                if isinstance(type(obj), obj_cls)}

    @property
    def robots(self):
        """ Dict with all robots.
        """
        return {name : obj for name, obj in self.hierarchy.items()\
                if type(obj).__name__ == 'Robot3D'}
    @property
    def lights(self):
        """ Dict with all light sources.
        """
        return {name : obj for name, obj in self.hierarchy.items()\
                if type(obj).__name__ == 'LightSource'}
    @property
    def controllable_objects(self):
        """ Dict with all controllable objects (ie with a controller).
        """
        return {name : obj for name, obj in self.hierarchy.items()\
                if obj.controllable}
    @property
    def uncontrollable_objects(self):
        """ Dict with all uncontrollable objects.
        """
        return {name : obj for name, obj in self.hierarchy.items()\
                if not obj.controllable}
    @property
    def tangible_objects(self):
        """ Dict with all tangible objects.
        """
        return {name : obj for name, obj in self.hierarchy.items()\
                if obj.tangible}
    @property
    def intangible_objects(self):
        """ Dict with all intangible objects.
        """
        return {name : obj for name, obj in self.hierarchy.items()\
                if not obj.tangible}
    @property
    def moving_objects(self):
        """ Dict with all objects with movement capabilities.
        """
        return {name : obj for name, obj in self.hierarchy.items() if not obj.static}
    @property
    def static_objects(self):
        """ Dict with all static objects.
        """
        return {name : obj for name, obj in self.hierarchy.items() if obj.static}
    @property
    def luminous_objects(self):
        """ Dict with all luminous objects.
        """
        return {name : obj for name, obj in self.hierarchy.items() if obj.luminous}
