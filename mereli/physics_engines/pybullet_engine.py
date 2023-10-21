
import time
import logging
import numpy as np
import xml.etree.cElementTree as ET
import contextlib
with contextlib.redirect_stdout(None):
    import pybullet as p
    import pybullet_data
    import pybullet_utils.bullet_client as bc
from matplotlib import colors
from mereli.objects import Robot
from mereli.utils.utils import HidePrintf
from .base_engine import BaseEngine
from mereli.register import physics_engine_registry



@physics_engine_registry(name='pybullet')
class PybulletEngine(BaseEngine):
    """ 3D Physics and Render Engine class based on the `pybullet library <https://pybullet.org>`_. 
    Its role in the simulation is to iterate the 3D physic simulations and collision detections of the 
    entities in the environment and render the 3D graphics. 

    :param float dt: time step of the physics simulation (in seconds).
    :param float T_control: period of the sensing+control+action loop. It cannot be lower than dt and 
        it is set to 5*dt by default. Essentially this means that the physics are updated 5 times 
        in between every executing of sensors, controllers and actuators.

    :var BulletClient engine: pybullet client engine.
    :var bool render: flag indicating if the simulation is run in visual or render mode.
    :var bool connected: whether the engine is connected or not.
    :var dict physical_sensors: maps sensor reference names to actual physical links of the robot. Specifically, each key of the dict 
        corresponds to a sensor (`'distance_sensor'`, `'light_sensor'`, ...), which in turn have a subdict as value. The subdict of 
        each sensor has another int key per each of the available sectors (e.g. from 0 to 7). Finally, the value of each sector contains 
        information about the physical link of the corresponding sensor sector (link name, link idx, orientation, ghost link, etc.).
        For example, the value of the sector 0 of the `'distance sensor'` would be:

        Example::
        
        >>> print(self.physical_sensors['distance_sensor'][0])
        >>>     {'link': 'IR0', 'ghost_link': 'ghost_cone_DS0', 'orientation': array([0.     , 0.     , 0.26179]), 'idx': 33, 'ghost_link_idx': 34}

    :var dict physical_actuators: maps actuator reference names to actual physical links of the robot. 
    :var dict luminous_objects: dict that gathers all the entities with one or more links that emit light. 
        It maps entity identifiers to physical information, such as the luminous link id, the color of the light 
        or the luminosity. TODO: The complete integration of this feature and actual use of it in the simulation 
        is in process.  
    :var dict gui_params: unused ftm.  """
    def __init__(self, *args, **kwargs):
        super(PybulletEngine, self).__init__('3D', *args, **kwargs)
        self.physical_sensors = {}
        self.physical_actuators = {}
        self.luminous_objects = {}
        self.gui_params = {}
        self.robot_ids = []
        self.camera_options = {
            'focus' : True,
            'focus_coords' : [0,0,0],
            'focus_target' : 0,
            'distance' : 1.5,  
            'yaw' : 0,
            'pitch' : -89
        }
        self.fps = 240
        self.paused = False 
        self.debug = False 

    def connect(self, objects):
        """ Connects to the pybullet based physics and render engines. It starts the pybullet 
        client in either visual or direct mode, sets up all the physics constants (gravity, sampling period, etc.) 
        and adds all the ``WorldObjects`` to the engine.

        :param iterable objects: iterable of WorldObjects whose physics have to be simulated.
        """
        wpx = 1400#1920
        hpx = 1080
        options = f'--width={wpx} --height={hpx}' if self.render else ''
        with HidePrintf():
            self.engine = bc.BulletClient(connection_mode=p.GUI if self.render else p.DIRECT, options=options)
        # import os
        # plugin_fn = os.path.join(
        #     p.__file__.split("bullet3")[0],
        #     "bullet3/build/lib.linux-x86_64-3.5/eglRenderer.cpython-35m-x86_64-linux-gnu.so")
        # plugin = p.loadPlugin(plugin_fn, "_tinyRendererPlugin")
        p.setInternalSimFlags(0)
        self.engine.resetSimulation(physicsClientId=self.client)

        # p.resetSimulation(physicsClientId=self.client)
        self.engine.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.engine.setGravity(0, 0, -9.8)
        self.engine.setTimeStep(self.dt)
        # self.engine.setPhysicsEngineParameter(numSolverIterations=10)

        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)#, globalScaling=5)
        p.setCollisionFilterGroupMask(plane_id, -1, 0b0, 0b0, physicsClientId=self.client)

        # self.engine.changeDynamics(planeId, linkInde1, lateralFriction=0.9)
        # p.setPhysicsEngineParameter(enableConeFriction=0)
        if self.render:
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
        self.add_objects(objects)
        if self.render:
            if len(self.robot_ids) == 0:
                self.camera_options['focus'] = False
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        # p.setPhysicsEngineParameter(numSolverIterations=10)
        # p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
        self.connected = True

    def disconnect(self):
        """ Disconnects the pybullet based physics and render engines. """
        # self.engine.resetSimulation(physicsClientId=self.engine._client)
        self.engine.disconnect()
        self.connected = False

    def step_physics(self):
        """ Iterates all the 3D physics of the world entities using pybullet. """
        if self.connected:
            for i in range(int(self.T_control//self.dt)):
                p.stepSimulation(physicsClientId=self.client)


    def step_render(self):
        """ Iterates the graphics visualization at given FPS. """
        pKey = ord('p')
        rKey = ord('r')
        tabKey = ord('\t')
        plusKey = 93
        minusKey = 47 
        ctrlKey = 65307 
        delKey = 8 
        jKey = ord('j') 
        hKey = ord('h') 
        kKey = ord('k') 
        lKey = ord('l') 
        wKey = ord('w') 
        aKey = ord('a') 
        sKey = ord('s') 
        dKey = ord('d') 
        keys = p.getKeyboardEvents()
        if len(keys) > 0:
            if pKey in keys and keys[pKey]&p.KEY_WAS_TRIGGERED:
                self.paused = True
                print('KEY P PRESSED!')
            if rKey in keys and keys[rKey]&p.KEY_WAS_TRIGGERED:
                self.paused = False
                print('KEY R PRESSED!')
            if tabKey in keys and keys[tabKey]&p.KEY_WAS_TRIGGERED:
                print('KEY Tab PRESSED!')
                if self.debug and self.camera_options['focus']:
                    self.hide_ghost_objects(self.robot_ids[self.camera_options['focus_target']])
                self.camera_options['focus'] = True
                if self.camera_options['focus_target'] is None:
                    self.camera_options['focus_target']= 0
                else:
                    self.camera_options['focus_target'] = (self.camera_options['focus_target']+ 1) % len(self.robot_ids)            
                if self.debug and self.camera_options['focus']:
                    self.show_ghost_objects(self.robot_ids[self.camera_options['focus_target']])
            if delKey in keys and keys[delKey]&p.KEY_WAS_TRIGGERED:
                self.camera_options['focus'] = False 
                if self.debug:
                    self.hide_ghost_objects(self.robot_ids[self.camera_options['focus_target']])
            if plusKey in keys and keys[plusKey]&p.KEY_WAS_TRIGGERED:
                if ctrlKey in keys: 
                    self.fps = min(250, self.fps + 10)
                    print(f'Simulation FPS increased to {self.fps}')
                else:
                    self.camera_options['distance'] -= 0.25
            if minusKey in keys and keys[minusKey]&p.KEY_WAS_TRIGGERED:
                if ctrlKey in keys:
                    self.fps = max(10, self.fps - 10)
                    print(f'Simulation FPS decreased to {self.fps}')
                else:
                    self.camera_options['distance'] += 0.25
            # Control Pitch and yaw of camera using h, j, k and l
            if lKey in keys and keys[lKey]&p.KEY_WAS_TRIGGERED:
                self.camera_options['yaw'] += 5
                # print(f'Camera Yaw changed to {self.yaw}')
            if hKey in keys and keys[hKey]&p.KEY_WAS_TRIGGERED:
                self.camera_options['yaw'] -= 5
                # print(f'Camera Yaw changed to {self.yaw}')
            if kKey in keys and keys[kKey]&p.KEY_WAS_TRIGGERED:
                self.camera_options['pitch'] = min(self.camera_options['pitch'] + 5, 1)
                # print(f'Camera pitch changed to {self.pitch}')
            if jKey in keys and keys[jKey]&p.KEY_WAS_TRIGGERED:
                self.camera_options['pitch']= max(self.camera_options['pitch'] - 5, -89) 
                # print(f'Camera Pitch changed to {self.pitch}')
        if self.camera_options['focus'] and self.camera_options['focus_target'] is not None: 
            self.camera_options['focus_coords'] = self.get_body_position(self.robot_ids[self.camera_options['focus_target']], -1)
        else:
            # Manual free control using w, a, s and d (only if focus is disabled)
            if wKey in keys and keys[wKey]&p.KEY_WAS_TRIGGERED:
                self.camera_options['focus_coords'][1] += 0.25 
            if sKey in keys and keys[sKey]&p.KEY_WAS_TRIGGERED:
                self.camera_options['focus_coords'][1] -= 0.25 
            if aKey in keys and keys[aKey]&p.KEY_WAS_TRIGGERED:
                self.camera_options['focus_coords'][0] -= 0.25 
            if dKey in keys and keys[dKey]&p.KEY_WAS_TRIGGERED:
                self.camera_options['focus_coords'][0] += 0.25 
        #     camera_focus = (0,0,0)
        p.resetDebugVisualizerCamera(cameraDistance=self.camera_options['distance'], cameraYaw=self.camera_options['yaw'], \
                cameraTargetPosition=self.camera_options['focus_coords'], cameraPitch=self.camera_options['pitch'])#-60,)
        if self.fps < 250: 
            time.sleep(1 / self.fps)


    def add_objects(self, objects):
        """
        Iteratively add all the WorldObject entities to the engine so that its physics can be taken into account 
        during the simulation.

        :param iterable objects: iterable of WorldObjects whose physics have to be simulated.
        """
        import time
        t0= time.time()
        for obj in objects:
            # self.add_physics(obj)
            obj.physics_client = self
            obj.render()
            if issubclass(type(obj), Robot):
                self.robot_ids.append(obj.id)
                shapeId = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius = 0.4)
                p.changeVisualShape(obj.id, self.physical_sensors['distance_sensor'][0]['ghost_link_idx'],shapeIndex=shapeId)
                self.physical_sensors['distance_sensor'][0]['ghost_link_idx']
        # print('Render time ', time.time() - t0)

    def load_from_urdf(self, obj):
        # self.set_camera_focus([0,0,-2], 1)
        obj.id = p.loadURDF(obj.model_file, obj.init_position,\
            p.getQuaternionFromEuler(obj.init_orientation),
            globalScaling=1 * (obj.scaling if hasattr(obj, 'scaling') else 1), 
            # flags= p.URDF_ENABLE_SLEEPING | p.URDF_ENABLE_WAKEUP,# | p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES,
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
        self.parse_urdf(obj) #

    def create_box(self, A=1, B=1, H=0.2, mass=0):
        colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[A / 2, B / 2, H], )
        # textureId = p.loadTexture("mereli/models/textures/brick.jpeg")
        boxId = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=colBoxId)
        # p.changeVisualShape(boxId, 0, textureUniqueId=textureID)
        # textureId = p.loadTexture("floor_diffuse.jpg")
        # terrain  = p.createMultiBody(0, terrainShape)
        p.changeVisualShape(boxId, -1, rgbaColor=[0.,0.,.4,1])
        p.setCollisionFilterGroupMask(boxId, -1, 0b001, 0b001, physicsClientId=self.client)

        # p.setCollisionFilterPair(0, colBoxId, -1, -1, 1, physicsClientId=self.client)
        return boxId


    def add_physics(self, obj):
        """
        Adds the requested entity to the engine so that its physics can be taken into account 
        during the simulation. It uses the pybullet function ``loadURDF``, that creates the pybullet 
        entity described in the form of an URDF file. The precise file describing the entity is an 
        attribute of the corresponding WorldObject class. 

        :param WorldObject obj: entity to be added to the engine.
        """
        # print(obj)
        if obj.model_file is None:
            return
        obj.physics_client = self
        obj.id = p.loadURDF(obj.model_file, obj.init_position,\
            p.getQuaternionFromEuler(obj.init_orientation),
            globalScaling=1 * (obj.scaling if hasattr(obj, 'scaling') else 1), 
            physicsClientId=self.client)
        # print('Load: ', time.time() - t0)
        t0 = time.time()
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
        # for i in range(2):
        #     p.changeDynamics(obj.id, i, contactStiffness=.1, physicsClientId=self.client)
        # print('Config and colls: ', time.time() - t0)
        self.parse_urdf(obj) #
        if issubclass(type(obj), Robot):
            self.robot_ids.append(obj.id)


    def parse_urdf(self, obj):
        link_names = np.array([p.getJointInfo(obj.id, i, physicsClientId=self.client)[12]\
                for i in range(p.getNumJoints(obj.id, physicsClientId=self.client))]).astype(str)
        tree = ET.parse(obj.model_file)
        root = tree.getroot()
        for sensor in root.findall(".//sensor"):
            sensor_name = sensor.get('name')
            if sensor_name not in self.physical_sensors:
                self.physical_sensors[sensor_name] = {}
            for sector in sensor.findall("sector"):
                sector_idx = int(sector.get('index'))
                link = sector.find('parent').get('link')
                orientation = np.array(sector.find('origin').get('rpy').split(' ')).astype(float)
                link_idx = np.where(link_names == link)[0][0]
                ghost_link = sector.find('ghost').get('link') if sector.find('ghost') is not None else None
                ghost_link_idx = np.where(link_names == ghost_link)[0] if ghost_link is not None else None
                if ghost_link_idx.size == 0:
                    ghost_link_idx = None
                else:
                    ghost_link_idx = ghost_link_idx[0]
                if ghost_link_idx is not None:
                    p.setCollisionFilterGroupMask(obj.id, ghost_link_idx, 0b00, 0b00, physicsClientId=self.client)
                    p.setCollisionFilterPair(0, obj.id, -1, ghost_link_idx, 0, physicsClientId=self.client)
                    # if sensor_name == 'distance_sensor':
                    # self.set_color(obj.id, ghost_link_idx, [1,0,0], opacity=0.5)
                    # self.set_color(obj.id, ghost_link_idx, [1,0,0], opacity=0.0)
                # import pdb; pdb.set_trace()
                p.setCollisionFilterGroupMask(obj.id, link_idx, 0b00, 0b00)
                self.physical_sensors[sensor_name][sector_idx] = {
                    'link' : link, 'ghost_link': ghost_link, 
                    'orientation' : orientation, 'idx' : link_idx, 'ghost_link_idx': ghost_link_idx,
                }
        
        for actuator in root.findall(".//actuator"):
            actuator_name = actuator.get('name')
            if actuator_name in self.physical_actuators:
                continue
            else:
                self.physical_actuators[actuator_name] = {}
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
        if np.isnan(pos).any(): 
            print(pos, self.__dict__)
            import pdb; pdb.set_trace()
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
        """ Casts a batch of rays between pairwise coordinates in origin and destination lists
        and verifies if there is some object/obstacle in between. 
        It returns the id of the first encountered object.

        Example::
        
        >>> # Cast two rays, both of them starting at [0,0,0] and with destinations
        >>> # [1,0,0] and [1,1,0] respectively.  
        >>> origins = [np.array([0,0,0])] * 2
        >>> destinations = [np.array([1,0,0]), np.array([1,1,0])]
        >>> # The result is a list with 2 components, each storing the id of the first 
        >>> # obstacle detected in the ray trajectory (or -1 if no object was detected). 
        >>> ids_list = ray_cast(origins, destinations) 

        :param list origin: list of numpy arrays with the set of origin coordinates.
        :param list destination: list of numpy arrays with the set of destination coordinates.

        :returns: list of int identifiers of the first intersected WorldObject by each of the casted rays
            between pairwise origins and destinations. For each ray, if no obstacle was detected it returns 
            a -1.
        """
        dest = destination
        # origin, dest = zip(*[(o + 1.2 * (d - o), o + 0.1 * (d - o)) for o, d in zip(origin, destination)])
        ray_res = p.rayTestBatch(origin, dest, collisionFilterMask=0b001, physicsClientId=self.client)
        ray_responses = []
        ray_positions = []
        for i in range(len(ray_res)):
            ray_i = ray_res[i]
            ori = origin[i]
            dest = destination[i]
            ray_responses.append(ray_i[0])
            if ray_i[0] != -1:
                coll = ori + ray_i[2] * (dest - ori)
                # __import__('pdb').set_trace()
                # ray_positions.append(ray_i[2] * np.array(ray_i[3]))
                ray_positions.append(coll)
            else:
                ray_positions.append(ray_i[3])
        return ray_responses, ray_positions
        # ray_res, ray_pos = zip(*[(ray[0], ray[2]*ray[3]) for ray in ray_res])
        # if len(ray_res) == 1:
        #     ray_res = ray_res[0]
        #     ray_pos = ray_pos[0]
        # return ray_res, ray_pos
    
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
                # import pdb; pdb.set_trace()
                p.setCollisionFilterGroupMask(obj_id, idx, 0b01, 0b01, physicsClientId=self.client)
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

    def control_joints(self, obj_id, joints, actions, control_type='velocity'):
        """ Control a series of robot joints either by velocity or by position.
        
        :param int obj_id: identifier of the robot whose joints will be controlled.
        :param list joints: list of identifiers of the joints of the robot to be controlled.
        :param list actions: list or np.ndarray of actions to control each joint.
        :param str control_type: type of joint control (either "velocity" or "position").
        """
        assert len(joints) == len(actions)
        for action, joint in zip(actions, joints):
            if control_type == 'velocity':
                p.setJointMotorControl2(obj_id, joint, targetVelocity=action, velocityGain=1,
                    controlMode=p.VELOCITY_CONTROL, physicsClientId=self.client)
            elif control_type == 'position':
                p.setJointMotorControl2(obj_id, joint, targetPosition=action, controlMode=p.POSITION_CONTROL,
                    positionGain=1.1, velocityGain=1.1, physicsClientId=self.client)
            else:
                raise Exception(logging.error('Joints cannot be controlled by {}.'\
                     'Please Select either "velocity" or "position".'.format(control_type)))

    def read_joints(self, obj_id, joints):
        """ Reads the position (rad) and velocity (rad/s) of the requested joints of a robot. It returns 
        a tuple (joint_velocities, joint_positions) with the arrays of the measurements of each kind.

        :param int obj_id: identifier of the robot whose joints will be read.
        :param list joints: list of identifiers of the joints of the robot to be read.

        :returns: tuple of the form (joint_velocities, joint_positions) with the arrays of the measurements of each kind.
        """
        positions, velocities = map(np.array, zip(*[p.getJointState(obj_id, joint, 
                                    physicsClientId=self.client)[:2] for joint in joints]))
        positions = (positions + np.pi) % (2 * np.pi) - np.pi #! Check
        return (positions, velocities)

    def compute_inverse_kinematics(self, target_position, target_orientation, obj_id, end_effector_link=-1):
        joint_actions = p.calculateInverseKinematics(obj_id, end_effector_link, target_position, lowerLimits=[-5, -5], upperLimits=[5, 5])
        return joint_actions
    
    def change_dynamics(self, obj_id, **kwargs):
        p.changeDynamics(obj_id, -1,  physicsClientId=self.client, **kwargs)

    def set_color(self, obj_id, link_id, color, opacity=1.0):
        """ Sets the color and opacity of a link of an entity. 
        
        :param int obj_id: identifier of the robot owning the sensor.
        :param int obj_id: identifier of the link of the robot owning the sensor whose color is changed.
        :param list color: ``list`` with the RGB code of the color or ``str`` with the color name.
        :para float opacity: opacity of the color.
        """
        if isinstance(color, str):
            color = list(colors.to_rgb(color))
        rgba_color = color + [opacity]
        p.changeVisualShape(obj_id, link_id, rgbaColor=rgba_color, physicsClientId=self.client)

    def set_texture(self, obj_id, link_id, texture_file):
        texture = p.loadTexture('mereli/models/textures/bricks2.jpeg')
        p.changeVisualShape(obj_id, link_id, textureUniqueId=texture)

    def set_camera_options(self, **kwargs):
        self.camera_options.update(kwargs)

    def set_camera_focus(self, position, distance, yaw=0, pitch=-90):
        """ Sets of the camera target position and distance in the environment. 
        The camera spotlight is set to the given position and the camera it placed at the given 
        distance wrt to that position. Yaw and pitch in degrees can be also specified. 
        This method is only applied in render mode.

        :param np.ndarray position: new spotlight of the camera.
        :param float distance: distance of the camera wrt to the spotlight.
        :param float yaw: yaw angle of the camera.
        :param float pitch: pitch angle of the camera.
        """
        if self.render:
            self.engine.resetDebugVisualizerCamera(cameraDistance=distance, cameraYaw=yaw,\
                    cameraPitch=pitch, cameraTargetPosition=tuple(position))

    def show_ghost_objects(self, obj_id):
        for i in range(8):
            self.set_color(obj_id, self.physical_sensors['distance_sensor'][i]['ghost_link_idx'], [1,0,0], opacity=0.4)

    def hide_ghost_objects(self, obj_id):
        for i in range(8):
            self.set_color(obj_id, self.physical_sensors['distance_sensor'][i]['ghost_link_idx'], [1,0,0], opacity=0.0)
    
    def is_focused(self, obj_id):
        return self.camera_options['focus'] and self.robot_ids[self.camera_options['focus_target']] == obj_id


