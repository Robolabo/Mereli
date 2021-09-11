import time
import os
import json
import numpy as np
import xml.etree.cElementTree as ET
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc
import pygame
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pymunk
import pymunk.pygame_util
from pygame.color import THECOLORS
from matplotlib import colors
from spike_swarm_sim.globals import global_states


    

class Engine3D:
    """ 3D Physics and Render Engine class. Its role in the simulation is to iterate the 
    3D physic simulations and collision detections of the entities in the environment and render 
    the 3D graphics. For these purposes it uses the `pybullet library <https://pybullet.org>`_  .

    :param float dt: time step of the physics simulation (in seconds).
    :param float T_control: period of the sensing+control+action loop. It cannot be lower than dt and 
        it is set to 5*dt by default. Essentially this means that the physics are updated 5 times 
        in between every executing of sensors, controllers and actuators.

    :var BulletClient engine: pybullet client engine.
    :var bool render: flag indicating if the simulation is run in visual or render mode.
    :var bool connected: whether the engine is connected or not.
    """
    def __init__(self, dt=0.02, T_control=0.1):
        self.dt = dt
        self.T_control = T_control
        assert T_control >= dt
        self.connected = False
        self.render = global_states.RENDER
        self.engine = None
        self.physical_sensors = {}
        self.physical_actuators = {}
        self.luminous_objects = {}
        self.gui_params = {}


    def connect(self, objects):
        """ Connects to the pybullet based physics and render engines. It starts the pybullet 
        client in either visual or direct mode, sets up all the physics constants (gravity, sampling period, etc.) 
        and adds all the ``WorldObjects`` to the engine.

        :param iterable objects: iterable of WorldObjects whose physics have to be simulated.
        """
        self.engine = bc.BulletClient(connection_mode=p.GUI if self.render else p.DIRECT)
        self.engine.resetSimulation(physicsClientId=self.engine._client)
        self.engine.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.engine.setGravity(0, 0, -9.8)
        self.engine.setTimeStep(self.dt)
        # self.engine.setPhysicsEngineParameter(numSolverIterations=10)
        # self.engine.setPhysicsEngineParameter(fixedTimeStep=1000)
        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        p.setCollisionFilterGroupMask(plane_id, -1, 0b0, 0b0)
        # self.engine.changeDynamics(planeId, linkIndex=-1, lateralFriction=0.9)
        self.add_objects(objects)
        self.connected = True
        # self.gui_params = {}
        if self.render:
            self.gui_params['light_coverage'] = self.engine.addUserDebugParameter("Show lights' coverage", 1, -1, -1)
            # self.gui_params['robot_focus'] = self.physics_client.addUserDebugParameter('Robot focus', 1, -1, 1)
            self.engine.resetDebugVisualizerCamera(cameraDistance=4, cameraYaw=30,\
                    cameraPitch=-90, cameraTargetPosition=[0, 0, 0])

    def disconnect(self):
        """ Disconnects the pybullet based physics and render engines. """
        # self.engine.resetSimulation(physicsClientId=self.engine._client)
        self.engine.disconnect()
        self.connected = False

    def step_physics(self):
        """ Iterates all the 3D physics of the world entities using pybullet. """
        for i in range(int(self.T_control//self.dt)):
            p.stepSimulation()

    def step_render(self):
        """ Iterates the graphics visualization at give FPS. """
        # if self.physics_client.readUserDebugParameter(self.gui_params['robot_focus']) == 1:
        #     self.physics_client.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=30,\
        #         cameraTargetPosition=self.robots['robotA_0'].position, cameraPitch=-70)#-60,)
        time.sleep(1/240.) # Fast mode
        # time.sleep(1/10) # Slow mode


    def add_objects(self, objects):
        """
        Iteratively add all the WorldObject entities to the engine so that its physics can be taken into account 
        during the simulation.

        :param iterable objects: iterable of WorldObjects whose physics have to be simulated.
        """
        for obj in objects:
            self.add_physics(obj)

    def add_physics(self, obj):
        """
        Adds the requested entity to the engine so that its physics can be taken into account 
        during the simulation. It uses the pybullet function ``loadURDF``, that creates the pybullet 
        entity described in the form of an URDF file. The precise file describing the entity is an 
        attribute of the corresponding WorldObject class. 

        :param WorldObject obj: entity to be added to the engine.
        """
        if obj.model_file is None:
            return
        obj.physics_client = self
        obj.id = p.loadURDF(obj.model_file, obj.init_position,\
            p.getQuaternionFromEuler(obj.init_orientation),
            globalScaling=obj.scaling if hasattr(obj, 'scaling') else 1, 
            physicsClientId=self.client)

        
        p.setCollisionFilterGroupMask(obj.id, -1, 0b001, 0b001, physicsClientId=self.client)
        p.setCollisionFilterPair(0, obj.id, -1, -1, 1, physicsClientId=self.client)
        for i in range(p.getNumJoints(obj.id, physicsClientId=self.client)):
            p.setCollisionFilterGroupMask(obj.id, i, 0b001, 0b01, physicsClientId=self.client)
            p.setCollisionFilterPair(0, obj.id, -1, i, 1, physicsClientId=self.client)


        if hasattr(obj, 'color'):
            color = list(colors.to_rgb(obj.color)) + [1.]
            p.changeVisualShape(obj.id, -1, rgbaColor=color, physicsClientId=self.client)
        if hasattr(obj, 'mass'):
            p.changeDynamics(obj.id, -1, mass=obj.mass, physicsClientId=self.client)
        #! Prov loop
        for i in range(2):
            p.changeDynamics(obj.id, i, lateralFriction=0.9, physicsClientId=self.client,\
                activationState=p.ACTIVATION_STATE_DISABLE_WAKEUP)
        
        self.parse_urdf(obj) #

    def parse_urdf(self, obj):
        link_names = np.array([p.getJointInfo(obj.id, i, physicsClientId=self.client)[12]\
                for i in range(p.getNumJoints(obj.id, physicsClientId=self.client))]).astype(str)
        tree = ET.parse(obj.model_file)
        root = tree.getroot()
        for sensor in root.findall(".//sensor"):
            sensor_name = sensor.get('name')
            self.physical_sensors[sensor_name] = {}
            for sector in sensor.findall("sector"):
                sector_idx = int(sector.get('index'))
                link = sector.find('parent').get('link')
                orientation = np.array(sector.find('origin').get('rpy').split(' ')).astype(float)
                link_idx = np.where(link_names == link)[0][0]
                ghost_link = sector.find('ghost').get('link') if sector.find('ghost') is not None else None
                ghost_link_idx = np.where(link_names == ghost_link)[0][0] if ghost_link is not None else None
                if ghost_link_idx is not None:
                    p.setCollisionFilterGroupMask(obj.id, ghost_link_idx, 0b00, 0b00, physicsClientId=self.client)
                    p.setCollisionFilterPair(0, obj.id, -1, ghost_link_idx, 0, physicsClientId=self.client)
                    self.set_color(obj.id, ghost_link_idx, [0,0,0], opacity=.0)
                self.physical_sensors[sensor_name][sector_idx] = {
                    'link' : link, 'ghost_link': ghost_link, 
                    'orientation' : orientation, 'idx' : link_idx, 'ghost_link_idx': ghost_link_idx,
                }
        for actuator in root.findall(".//actuator"):
            actuator_name = actuator.get('name')
            self.physical_actuators[actuator_name] = {}
            for sector in actuator.findall("sector"):
                sector_idx = int(sector.get('index'))
                link = sector.find('parent').get('link')
                link_idx = np.where(link_names == link)[0][0]
                self.physical_actuators[actuator_name][sector_idx] = {
                    'link' : link,  'idx' : link_idx
                }

        for ls in root.findall(".//lightsource"):
            link = ls.get('link')
            link_idx = np.where(link_names == link)[0][0] if len(link_names) else -1
            lum = int(ls.get('luminosity'))
            color = ls.get('color')
            if hasattr(obj, 'color'):
                color = obj.color
            self.luminous_objects[obj.id] = {'link' : link, 'link_idx': link_idx, 'color' : color, 'luminosity' : lum}

    @property
    def client(self):
        """ Pybullet engine client used in the simulation. """
        return self.engine._client


    def get_body_position(self, identifier, body_id, z_offset=0.0):
        """
        Getter method of the current position of the root link of an entity with the 
        given identifier.

        :param int identifier: identifier of the entity whose position is requested.
        :param int body_id: deprecated, to be removed

        :returns: numpy array of shape (3,) with the entities' position.
        """
        pos = np.array(p.getBasePositionAndOrientation(identifier, physicsClientId=self.client)[0])        
        if np.isnan(pos).any():import pdb; pdb.set_trace()
        return pos
    
    def get_body_orientation(self, identifier, body_id):
        """ 
        Getter method of the current Euler orientation of the root link of an entity with the 
        given identifier.

        :param int identifier: identifier of the entity whose position is requested.
        :param int body_id: deprecated, to be removed

        :returns: numpy array of shape (3,) with the entities' Euler orientation.
        """
        quaternion_orientation = p.getBasePositionAndOrientation(identifier, physicsClientId=self.client)[1]
        return np.array(p.getEulerFromQuaternion(quaternion_orientation, physicsClientId=self.client))
         

    def get_body_velocity(self, identifier, body_id):
        """ 
        Getter method of the current velocity of the root link of an entity with the 
        given identifier.

        :param int identifier: identifier of the entity whose position is requested.
        :param int body_id: deprecated, to be removed

        :returns: numpy array of shape (3,) with the entities' velocity.
        """
        return p.getBaseVelocity(identifier, physicsClientId=self.client)[0]
    
    def get_body_angular_velocity(self, identifier, body_id):
        """ 
        Getter method of the current angular velocity of the root link of an entity with the 
        given identifier.

        :param int identifier: identifier of the entity whose position is requested.
        :param int body_id: deprecated, to be removed

        :returns: numpy array of shape (3,) with the entities' angular velocity.
        """
        return p.getBaseVelocity(identifier, physicsClientId=self.client)[0]

    def set_body_state(self, identifier, body_id, position, orientation):
        """ 
        Sets the physics state (position and orientation) of a registered entity with the given
        identifier. 

        :param int identifier: identifier of the entity whose position is requested.
        :param int body_id: deprecated, to be removed
        :param np.ndarray position: new 3D position of the entity.
        :param np.ndarray orientation: new 3D Euler orientation of the entity. It is also possible to 
            introduce an angle scalar in radians so that orientation = [0,0,orientation].
        """
        if len(orientation) == 1:
            orientation = [0., 0., orientation]
        p.resetBasePositionAndOrientation(identifier, position,\
            p.getQuaternionFromEuler(orientation), physicsClientId=self.client)

    def ray_cast(self, origin, destination):
        """ Casts a ray between coordinates origin and destination and verifies if there is some 
        object in between. It returns the id of the first encountered object.
        #!UPDATE

        :param np.ndarray origin: 3D numpy array with the origin coordinates.
        :param np.ndarray destination: 3D numpy array with the destination coordinates.

        :returns: int identifier of the first intersected WorldObject by the casted ray.
        """
        origin, dest = zip(*[(o + 1.2 * (d - o), o + 0.1 * (d - o)) for o, d in zip(origin, destination)])
        ray_res = p.rayTestBatch(origin, dest, collisionFilterMask=0b001, physicsClientId=self.client)
        ray_res, ray_pos = zip(*[(ray[0], ray[3]) for ray in ray_res])
        if len(ray_res) == 1:
            ray_res = ray_res[0]
            ray_pos = ray_pos[0]

        # p.addUserDebugLine(origin[0], dest[0], lineColorRGB=[1, 0, 0], lineWidth=2.0, lifeTime=0.1, physicsClientId=self.client)
        # if ray_res[0][0] == -1 and np.linalg.norm(dest - origin) < 0.8: 
        #     p.addUserDebugLine(origin, dest, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0, physicsClientId=self.client)
        #     import pdb; pdb.set_trace()
        return ray_res, ray_pos
    
    def get_closest_point(self, idA, idB, linkA=-1, linkB=-1, max_dist=10):
        """ Computes the closest points between two links of two registered entities.

        :param int idA: identifier of the first WorldObject entity.
        :param int idB: identifier of the second WorldObject entity.
        :param int linkA: identifier of the link of the first WorldObject entity.
        :param int linkB: identifier of the link of the second WorldObject entity.
        :param float max_dist: maximum distance between the objects.

        :returns: 3D numpy array with the coordinates of the closest point in linkB of entity with idB. 
        """
        if linkB is None:
            closest_points = p.getClosestPoints(idA, idB, max_dist, linkIndexA=linkA, physicsClientId=self.client)
        else:
            closest_points = p.getClosestPoints(idA, idB, max_dist, linkIndexA=linkA, linkIndexB=linkB, physicsClientId=self.client)
        if len(closest_points) == 0:
            return closest_points
        return np.array(closest_points[np.argmin([v[8] for v in closest_points])][6])

    def get_contact_points(self, obj_id, ghost_ids=None):
        """ Computes the contact points between any link of the given entity and any other 
        entity. It steps the collision detection engine to perform the query. The identifier 
        of ghost links can be specified in order to momentarily activate collisions and detect 
        obstacles.

        :param int obj_id: identifier of the WorldObject entity.
        :param list ghost_ids: list of the identifiers of the entity ghost links.

        :returns: list of tuples, each composed by the following entries: (objB_id, linkA_id, linkB_id).
        """
        if ghost_ids is not None:
            for idx in ghost_ids:
                p.setCollisionFilterGroupMask(obj_id, idx, 0b01, 0b01,  physicsClientId=self.client)
        p.performCollisionDetection(physicsClientId=self.client)
        contact_points = p.getContactPoints(obj_id, physicsClientId=self.client)
        if ghost_ids is not None:
            for idx in ghost_ids:
                p.setCollisionFilterGroupMask(obj_id, idx, 0b0, 0b0,  physicsClientId=self.client)
        return [(pt[2], pt[3], pt[4]) for pt in contact_points]

    def get_link_state(self, obj_id, link_idx):
        """ Getter of the position and orientation of a given link in the specified entity.
        
        :param int obj_id: identifier of the WorldObject entity.
        :param int link_idx: identifier of the link of the WorldObject entity.

        :returns: ``tuple`` with the 3D numpy position and 3D orientation of the link.
        """
        pos, qt_ori =  p.getLinkState(obj_id, link_idx, physicsClientId=self.client)[:2]
        return (np.array(pos), np.array(p.getEulerFromQuaternion(qt_ori, physicsClientId=self.client)))

    def get_sensor_position(self, obj_id, sensor_name, sector=0):
        """ Getter of the physical position of a sensor within a robot. Sensors are attached to 
        robot links and, therefore, it returns the 3D coordinates of the corresponding link.

        .. note::
            For the moment only directional sensor positions can be queried.
        
        :param int obj_id: identifier of the robot owning the sensor.
        :param str sensor_name: reference name of the sensor.
        :para int sector: index of the sensor's sector requested.

        :returns: numpy array with the position. 
        """
        sensor_index = self.physical_sensors[sensor_name][sector]['idx']
        return np.array(self.get_link_state(obj_id, sensor_index)[0]), sensor_index

    def get_actuator_position(self, obj_id, actuator_name, sector=0):
        """TODO Getter of the physical position of a sensor within a robot. Sensors are attached to 
        robot links and, therefore, it returns the 3D coordinates of the corresponding link.

        .. note::
            For the moment only directional sensor positions can be queried.
        
        :param int obj_id: identifier of the robot owning the sensor.
        :param str sensor_name: reference name of the sensor.
        :para int sector: index of the sensor's sector requested.

        :returns: tuple with the numpy array with the position and the actuator identifier.
        """
        actuator_index = self.physical_actuators[actuator_name][sector]['idx']
        return np.array(self.get_link_state(obj_id, actuator_index)[0]), actuator_index

    def get_sensor_orientation(self, obj_id, sensor_name, sector=0):
        """ Getter of the physical position of a sensor within a robot. Sensors are attached to 
        robot links and, therefore, it returns the 3D coordinates of the corresponding link.

        .. note::
            For the moment only directional sensor positions can be queried.
        
        :param int obj_id: identifier of the robot owning the sensor.
        :param str sensor_name: reference name of the sensor.
        :para int sector: index of the sensor's sector requested.

        :returns: numpy array with the position. 
        """
        return self.physical_sensors[sensor_name][sector]['orientation'][-1] #!only yaw ftm

    
    def set_color(self, obj_id, link_id, color, opacity=1.0):
        """ Getter of the physical position of a sensor within a robot. Sensors are attached to 
        robot links and, therefore, it returns the 3D coordinates of the corresponding link.

        .. note::
            For the moment only directional sensor positions can be queried.
        
        :param int obj_id: identifier of the robot owning the sensor.
        :param int obj_id: identifier of the link of the robot owning the sensor whose color is changed.
        :param list color: ``list`` with the RGB code of the color or ``str`` with the color name.
        :para float opacity: opacity of the color.
        """
        if isinstance(color, str):
            color = list(colors.to_rgb(color))
        rgba_color = color + [opacity]
        p.changeVisualShape(obj_id, link_id, rgbaColor=rgba_color, physicsClientId=self.client)

    def set_camera_focus(self, position, distance):
        if self.render:
            self.engine.resetDebugVisualizerCamera(cameraDistance=distance, cameraYaw=0,\
                    cameraPitch=-90, cameraTargetPosition=tuple(position))


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

class Engine2D:
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
    def __init__(self, height=1000, width=1000):
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
