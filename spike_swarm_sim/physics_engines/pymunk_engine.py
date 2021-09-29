import os
import json
import numpy as np
import contextlib
with contextlib.redirect_stdout(None):
    import pygame
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
    import pymunk
    import pymunk.pygame_util
    from pygame.color import THECOLORS
from spike_swarm_sim.globals import global_states
from .base_engine import BaseEngine
from spike_swarm_sim.register import physics_engine_registry
#!TODO NOT READY TO BE USED YET


def json_parser(file, position, orientation):
    """ Function that parses the json file describing the 2D entities.
    """
    file = "spike_swarm_sim/objects/urdf/" + file + '.json'
    with open(file) as json_file:
        obj_dict = json.load(json_file)
    #* Parse Links
    bodies = {}
    for body_cfg in obj_dict['links']:
        moment = pymunk.moment_for_circle(body_cfg['mass'], body_cfg['moment']['params']['radius'],  body_cfg['moment']['params']['radius'])\
                if body_cfg['moment']['type'] == 'circle' else pymunk.moment_for_poly(body_cfg['mass'], body_cfg['moment']['params']['vertices'])
        body = pymunk.Body(body_cfg['mass'], moment)
        body.position = tuple(position + np.array(body_cfg['xy']))
        body.angle = orientation + body_cfg['rpy']
        bodies.update({body_cfg['name'] : body})
    #* Parse Shapes
    shapes = []
    for shape_cfg in obj_dict['shapes']:
        body = bodies[shape_cfg['body']]
        shape = None
        if shape_cfg['type'] == 'circle':
            shape = pymunk.Circle(body, shape_cfg['params']['radius'], shape_cfg['params'].get('offset', (0,0)))
        elif shape_cfg['type'] == 'poly':
            shape = pymunk.Poly(body, shape_cfg['params']['vertices'], transform=pymunk.Transform(**shape_cfg['params'].get('transform', {})))
        shape.color = [*map(int, shape_cfg['color'].split(','))] + [255]
        shape.friction = shape_cfg.get('friction', 0.0)
        shape.mass = body.mass
        shapes.append(shape)
    #TODO Parse Joints
    joints = []
    for joint_cfg in obj_dict['joints']:
        pass
    return [*bodies.values()], shapes

@physics_engine_registry(name='pymunk')
class PymunkEngine(BaseEngine):
    """ 2D Physics and Render Engine class. Its role in the simulation is to iterate the 
    2D physic simulations and collision detections of the entities in the environment and render 
    the 2D graphics. For these purposes it uses the `pymunk library <https://pymunk.org>`_  for 
    as physics engine and `pygame library <https://pygame.org>`_ as render engine.

    :param float height: height of the graphics screen.
    :param float width: width of the graphics screen.
    
    :var pymunk.Space engine: pymunk client engine.
    :var bool render: flag indicating if the simulation is run in visual or render mode.
    :var bool connected: whether the engine is connected or not.
    """
    def __init__(self, *args, height=1000, width=1000, **kwargs):
        super(PymunkEngine, self).__init__('2D', *args, **kwargs)
        self.height = height
        self.width = width
        self.render = global_states.RENDER
        self.screen = None
        self.engine = None
        self.draw_options = None
        self.objects = {}
        self.groups = {}
        self.cat_pointer = 0b01

    def connect(self, objects):
        """ Connects to the pymunk and pygame based physics and render engines. 
        It creates the ``pymunk.Space``, sets up all the physics constants (gravity, etc.) 
        and adds all the ``WorldObjects`` to the engine. It also sets up the screen where 
        graphics will be rendered.

        :param iterable objects: iterable of WorldObjects whose physics have to be simulated.
        """
        self.engine = pymunk.Space()
        self.engine.gravity = (0.0, 0.0)
        self.add_objects(objects)
        self.connected = True
        if self.render:
            pygame.init()
            self.screen = pygame.display.set_mode((self.height, self.width))
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            self.clock = pygame.time.Clock()


    def add_physics(self, obj):
        """
        Adds the requested entity to the engine so that its physics can be taken into account 
        during the simulation. It parses the entity 2D model from a json asset. 
        The precise json file describing the entity is an attribute of the corresponding WorldObject class. 
        Besides, it assigns the mask and category mask of pymunk to filter the object individually of within 
        a group of entities. For example, the robots of a swarm would belong to the same group and would have 
        the same category mask. However, each swarm member would have a different mask. Untangible entities 
        (e.g. ``LightSources``) have the 0b0 mask to ignore collisions. 

        :param WorldObject obj: entity to be added to the engine.
        """
        if obj.model_file is None:
            return
        obj.physics_client = self
        pos = np.r_[obj.init_position.copy()] * 100 + 500 
        links, shapes = json_parser(obj.model_file, pos.tolist(), obj.init_orientation)
        cat_mask = self.groups[obj.group]['cat']
        mask = self.groups[obj.group]['last_mask']
        for sh in shapes:
            sh.filter = pymunk.ShapeFilter(categories=cat_mask, mask=mask)
            if hasattr(obj, 'color'):
                sh.color = THECOLORS[obj.color]
        if obj.tangible:
            self.groups[obj.group]['last_mask'] = mask << 1
        self.objects[obj.id] = {'bodies' : links, 'shapes' : shapes, 'mask' : mask, 'cat': cat_mask}
        self.engine.add(*links, *shapes)

    def add_objects(self, objects):
        """
        Iteratively add all the WorldObject entities to the engine so that its physics can be taken into account 
        during the simulation.

        :param iterable objects: iterable of WorldObjects whose physics have to be simulated.
        """
        for obj in objects:
            # Update group masks
            if obj.group not in self.groups:
                self.groups[obj.group] = {'cat' : obj.tangible and self.cat_pointer or 0b0, 'last_mask' : obj.tangible and 0b01 or 0b0}
                if obj.tangible:
                    self.cat_pointer = self.cat_pointer << 1
            self.add_physics(obj)
    
    def get_body_position(self, identifier, body_id):
        pos = self.objects[identifier]['bodies'][body_id].position
        return (np.array([pos.x, pos.y]) - 500) / 100 

    def reset_body_position(self, identifier, body_id, position):
        position = position * 100 + 500  
        self.objects[identifier]['bodies'][body_id].position = position

    def get_body_orientation(self, identifier, body_id):
        return self.objects[identifier]['bodies'][body_id].angle

    def reset_body_orientation(self, identifier, body_id, orientation):
        self.objects[identifier]['bodies'][body_id].angle = orientation

    def get_link_state(self, obj_id, link_idx):
        sh = self.objects[obj_id]['shapes']

    def disconnect(self):
        pass

    def initialize_render(self):
        pass

    def step_physics(self):
        # self.engine.step(1 / 60.0)
        self.engine.step(1 / 30.0)

    def step_render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                import sys; sys.exit(0)
        pygame.event.get()
        self.screen.fill((123,123,123))
        self.engine.debug_draw(self.draw_options)
        pygame.display.flip()
        self.clock.tick(60)



    def get_closest_point(self, obj_id):
        pass
        # p.getClosestPoints(self.sensor_owner.id, obj.id, 200,\
        #                     linkIndexA=-1, linkIndexB=-1, physicsClientId=self.sensor_owner.physics_client)


    def ray_cast(self, origin, destination):
        if len(origin) == 3:
            origin = origin[:2]
        if len(destination) == 3:
            destination = destination[:2]
        ray_res = self.engine.segment_query_first(origin, destination, 1, pymunk.ShapeFilter())
        if ray_res is not None:
            # Find ID of body
            ray_res.shape.body
        #* 
        return ray_res[0][0]