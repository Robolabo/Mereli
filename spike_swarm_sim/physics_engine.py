import time
import os
import json
import numpy as np
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc
import pygame
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pymunk
import pymunk.pygame_util
from pygame.color import THECOLORS
# from pymunk.vec2d import Vec2d
from spike_swarm_sim.globals import global_states

#! Inherit from Bullet??
class Engine3D:
    def __init__(self):
        self.connected = False
        self.render = global_states.RENDER
        self.engine = None
        self.gui_params = {}

    def step_physics(self):
        p.stepSimulation()

    def step_render(self):
        # if self.physics_client.readUserDebugParameter(self.gui_params['robot_focus']) == 1:
        #     self.physics_client.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=30,\
        #         cameraTargetPosition=self.robots['robotA_0'].position, cameraPitch=-70)#-60,)
        time.sleep(1/240.) # Fast mode
        # time.sleep(1/10) # Slow mode

    def connect(self, objects):
        self.engine = bc.BulletClient(connection_mode=p.GUI if self.render else p.DIRECT)
        self.engine.resetSimulation(physicsClientId=self.engine._client)
        self.engine.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.engine.setGravity(0, 0, -9.8)
        self.engine.setTimeStep(1/50.)
        # self.engine.setPhysicsEngineParameter(numSolverIterations=10)
        # self.engine.setPhysicsEngineParameter(fixedTimeStep=1000)
        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        # self.engine.changeDynamics(planeId, linkIndex=-1, lateralFriction=0.9)
        self.add_objects(objects)
        self.connected = True
        # self.gui_params = {}
        if self.render:
            self.gui_params['light_coverage'] = self.engine.addUserDebugParameter("Show lights' coverage", 1, -1, -1)
            # self.gui_params['robot_focus'] = self.physics_client.addUserDebugParameter('Robot focus', 1, -1, 1)
            self.engine.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=30,\
                    cameraPitch=-90, cameraTargetPosition=[0, 0, 0])

    def disconnect(self):
        # self.engine.resetSimulation(physicsClientId=self.engine._client)
        self.engine.disconnect()
        self.connected = False

    def add_physics(self, obj):
        # obj.physics_client = self.engine._client
        obj.physics_client = self
        obj.id = p.loadURDF(obj.urdf_file, obj.init_position,\
            p.getQuaternionFromEuler(obj.init_orientation),
            globalScaling=1., physicsClientId=self.client)
        #!
        for i in range(2):
            p.changeDynamics(obj.id, i, lateralFriction=0.9, physicsClientId=self.client,\
                activationState=p.ACTIVATION_STATE_DISABLE_WAKEUP)

    def add_objects(self, objects):
        for obj in objects:
            self.add_physics(obj)
            # obj.add_physics(self.engine._client)

    @property
    def client(self):
        return self.engine._client

    def get_body_position(self, identifier, body_id):
        pos = np.array(p.getBasePositionAndOrientation(identifier, physicsClientId=self.client)[0])        
        if np.isnan(pos).any():import pdb; pdb.set_trace()
        return pos
    
    def get_body_orientation(self, identifier, body_id):
        quaternion_orientation = p.getBasePositionAndOrientation(identifier, physicsClientId=self.client)[1]
        return np.array(p.getEulerFromQuaternion(quaternion_orientation, physicsClientId=self.client))

    def get_body_velocity(self, identifier, body_id):
        return p.getBaseVelocity(identifier, physicsClientId=self.client)[0]
    
    def get_body_angular_velocity(self, identifier, body_id):
        return p.getBaseVelocity(identifier, physicsClientId=self.client)[0]

    def set_body_state(self, identifier, body_id, position, orientation):
        if len(orientation) == 1:
            orientation = [0., 0., orientation]
        p.resetBasePositionAndOrientation(identifier, position,\
            p.getQuaternionFromEuler(orientation), physicsClientId=self.client)

    def ray_cast(self, origin, destination):
        """ Casts a ray between coordinates origin and destination and verifies if there is some 
        object in between. It returns the id of the first encountered object.
        """
        ray_res = p.rayTest(origin, destination, physicsClientId=self.client)
        return ray_res[0][0]
    
    def get_closest_point(self, idA, idB, linkA=-1, linkB=-1, max_dist=10):
        """ """
        closest_points = p.getClosestPoints(idA, idB, max_dist, linkIndexA=linkA, linkIndexB=linkB, physicsClientId=self.client)
        return np.array(closest_points[0][6])



    #! USELESS?
    def initialize_render(self):
        self.gui_params['light_coverage'] = self.engine.addUserDebugParameter("Show lights' coverage", 1, -1, 1)
        self.engine.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=30,\
                    cameraPitch=-60, cameraTargetPosition=[0, 0, 0])

    def get_link_state(self, obj_id, link_idx):
        return p.getLinkState(obj_id, link_idx, physicsClientId=self.engine._client)[:2]



def json_parser(file, position, orientation):
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

class Engine2D:
    def __init__(self, height=1000, width=1000):
        self.height = height
        self.width = width
        self.world_delay = 1 #! TO BE REMOVED
        self.render = global_states.RENDER
        self.screen = None
        self.engine = None
        self.draw_options = None
        self.objects = {}
        self.groups = {}
        self.cat_pointer = 0b01

    def connect(self, objects):
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
        for obj in objects:
            # Update group masks
            if obj.group not in self.groups:
                self.groups[obj.group] = {'cat' : obj.tangible and self.cat_pointer or 0b0, 'last_mask' : obj.tangible and 0b01 or 0b0}
                if obj.tangible:
                    self.cat_pointer = self.cat_pointer << 1
            self.add_physics(obj)
    

    def get_link_state(self, obj_id, link_idx):
        sh = self.objects[obj_id]['shapes']

    def get_body_position(self, identifier, body_id):
        return self.objects[identifier]['bodies'][body_id].position

    def reset_body_position(self, identifier, body_id, position):
        self.objects[identifier]['bodies'][body_id].position = position

    def get_body_orientation(self, identifier, body_id):
        return self.objects[identifier]['bodies'][body_id].angle

    def reset_body_orientation(self, identifier, body_id, orientation):
        self.objects[identifier]['bodies'][body_id].angle = orientation

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
