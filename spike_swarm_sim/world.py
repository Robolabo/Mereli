import time
import logging
from collections import deque
import tkinter as tk
import numpy as np
from spike_swarm_sim.objects import  Robot, LightSource
from spike_swarm_sim.objectives.reward import GoToLightReward
from spike_swarm_sim.register import controllers, world_objects, initializers, env_perturbations
from spike_swarm_sim.utils import angle_diff, compute_angle, normalize, increase_time, mov_average_timeit
from spike_swarm_sim.globals import global_states

WORLD_MODES = ['EVOLUTION', 'EVALUATION', 'DEBUGING']
class World(object):
    def __init__(self, height=1000, width=1000, render_connections=True, world_delay=1):
        self.height = height
        self.width = width
        self.render_connections = render_connections
        self.world_delay = world_delay
        self.render = global_states.RENDER
        if global_states.RENDER:
            self.root = tk.Tk(className='SpikeSwarmSim')
            self.root.geometry(str(width) + 'x' + str(height))
            self.canvas = tk.Canvas(self.root, height=self.height, width=self.width, bg='grey')
            self.canvas.pack(side='left')
            frame = tk.Frame(self.root)
            frame.pack(side='right')
            # Create limiting walls
            self.canvas.create_rectangle(0, 0, 20, height, fill='black')
            self.canvas.create_rectangle(0, 0, width, 20, fill='black')
            self.canvas.create_rectangle(width - 20, 0, width, height, fill='black')
            self.canvas.create_rectangle(0, height - 20, width, height, fill='black')
        #* Dict storing all objects
        self.hierarchy = {}
        #* Dict mapping object names to object groups
        self.groups = {}
        #* Dict storing how objects should be initialized as a group.
        self.initializers = {}
        #* Dict mapping object groups to environmental perturbations
        self.env_perturbations = {}
        if render_connections and global_states.RENDER:
            self.connection_graph = {}

        self.reward_generator = GoToLightReward()
        self.t = 0
        self.aux = 0

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
    ==========================================================
        """
        t0 = time.time()
        states = deque()
        actions = deque()
        pre_perturbations = []
        reward = 0.0

        #* Step controllers
        for idx, (n, obj) in enumerate(self.controllable_objects.items()):
            if not isinstance(obj, Robot):
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

        #* Actuate
        for obj in self.controllable_objects.values():
            if obj.tangible:
                obj.actuate()
            if self.render:
                self.canvas = obj.render(self.canvas)
        #* Apply mirror
        for robot in self.hierarchy.values(): #! OJO fall en las esquinas
            if  robot.controllable:
                robot.pos[robot.pos > 1000+15] = 20
                robot.pos[robot.pos < -13] = 1000-20
        #* Render step
        if self.render:
            if self.render_connections:
                self.draw_connections()
            self.canvas.update()
            self.root.after(self.world_delay)
        # print(time.time() - t0)
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
        obj_id = np.random.randint(1000)
        while(len(self.hierarchy) > 0 and obj_id in [obj.id for obj in self.hierarchy.values()]):
            obj_id = np.random.randint(1, 1000)
        obj.id = obj_id
        self.hierarchy.update({name : obj})
        #* Register group element
        if group is None:
            group = name
        if group in self.groups.keys():
            self.groups[group].append(name)
        else:
            self.groups[group] = [name]

        if self.render:
            self.canvas = obj.initialize_render(self.canvas)
    
    def build_from_dict(self, world_dict, ann_topology=None):
        """ Initialize all objects and add them into the world using a dictionary structure.
        =========================================================================================
        - Args:
            world_dict [dict] : configuration dict of the environment (parameters, objects, ...).
            ann_topology [dict] :  configuration dict of the neural network.
        - Returns: None
        =========================================================================================
        """
        for obj_name, obj in world_dict['objects'].items():
            #* Create group intializer.
            self.initializers[obj_name] = {
                key :  initializers[value['name']](obj['num_instances'], **value['params'])\
                        for key, value in obj['initializers'].items()
            }
            if obj['type'] == 'robot':
                robot_positions = self.initializers[obj_name]['positions']()
                robot_orientations = self.initializers[obj_name]['orientations']()
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
                    robot = Robot(position, orientation=orientation[0], controller=controller, **obj['params'])
                    self.add(obj_name + '_' + str(i), robot, group=obj_name)
                if len(obj['perturbations']) > 0:
                    self.env_perturbations.update({obj_name : [env_perturbations[pert](obj['num_instances'], **pert_params)\
                            for pert, pert_params in obj['perturbations'].items()]})
            else: # Non robot objects
                positions = self.initializers[obj_name]['positions']()
                controller = controllers[obj['controller']]() if obj['controller'] is not None else None
                for position in positions:
                    world_obj = world_objects[obj['type']](position, controller=controller, **obj['params'])
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
                    for pos, obj in zip(positions, group_elements):
                        obj.pos = pos
                #* Initialize orientations
                if 'orientations' in group_initializer.keys():
                    orientations = group_initializer['orientations']()
                    for orientation, obj in zip(orientations, group_elements):
                        obj.theta = orientation[0]
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
            if self.render:
                self.canvas = obj.render(self.canvas)
        for group_pert in self.env_perturbations.values():
            for pert in group_pert:
                pert.reset()

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
        if isinstance(robot, LightSource):
            return self.robots.values()
        if not isinstance(robot, Robot) or len(self.hierarchy) == 1:
            return neighbors
        max_robot_dist = None
        if len(self.robots) > 1:
            max_robot_dist = np.max([robot.sensors[sensor].range \
                            for sensor in ['wireless_receiver', 'distance_sensor'] \
                            if sensor in robot.sensors.keys()])
        #* Robots
        for obj in self.hierarchy.values():
            if isinstance(obj, Robot) and obj.id != robot.id:
                if max_robot_dist is not None and obj.id != robot.id:
                    if np.linalg.norm(obj.pos - robot.pos) <= max_robot_dist:
                        neighbors.append(obj)
            elif isinstance(obj, LightSource) and 'light_sensor' in robot.sensors.keys():
                if np.linalg.norm(obj.pos - robot.pos) <= robot.sensors['light_sensor'].range:
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
                if type(obj).__name__ == 'Robot'}
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

    def draw_connections(self):
        """
        Renders a line between robots iif both robots share a communication channel.
        That is, if the distance between robots is less than the comm range.
        """
        nodes = [obj for obj in self.controllable_objects.values() if obj.tangible]
        any(self.canvas.delete(item_id) for item_id in self.connection_graph.values())
        for i, nodeA in enumerate(nodes):
            for j, nodeB in enumerate(nodes):
                if i != j and np.linalg.norm(nodeA.pos - nodeB.pos) <= nodeB.actuators['wireless_transmitter'].range: #nodeA.sensors['wireless_transmitter'].range:
                    self.connection_graph[str(i)+'-'+str(j)] = self.canvas.create_line(nodeA.pos[0], nodeA.pos[1],\
                            nodeB.pos[0], nodeB.pos[1], fill='black', width=.7)

    def draw_node_coords(self):
        """
        Renders a text with the position in the world of each robot.
        """
        any([self.canvas.create_text(int(obj.pos[0]), int(obj.pos[1]-20), font="Purisa",
                    text=str(int(obj.pos[0]))+','+str(int(obj.pos[1])))\
                    for obj in self.controllable_objects.values()])