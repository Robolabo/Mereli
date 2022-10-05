
import time
import logging
import numpy as np
from .base_engine import BaseEngine
from mereli.register import physics_engine_registry
try:
    import pygame as pg
    from pygame.locals import *
    from OpenGL.GL import *
    from OpenGL.GLUT import *
    from OpenGL.GLU import *
except:
    print("Running without graphic libs.")

@physics_engine_registry(name='particle')
class ParticleEngine(BaseEngine):
    def __init__(self, *args, **kwargs):
        super(ParticleEngine, self).__init__('2D', *args, **kwargs)
        self.physical_sensors = {}
        self.physical_actuators = {}
        self.luminous_objects = {}
        self.gui_params = {}
        self.entities = {}

    def connect(self, objects):
        """ Connects to the pybullet based physics and render engines. It starts the pybullet 
        client in either visual or direct mode, sets up all the physics constants (gravity, sampling period, etc.) 
        and adds all the ``WorldObjects`` to the engine.

        :param iterable objects: iterable of WorldObjects whose physics have to be simulated.
        """
        if self.render:
            pg.init()
            display = (1680, 1050)
            pg.display.set_mode(display, DOUBLEBUF|OPENGL)
            gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
            glTranslatef(0.0, 0.0, -5)
        self.add_objects(objects)
        self.connected = True

    def disconnect(self):
        """ Disconnects the pybullet based physics and render engines. """
        # self.engine.resetSimulation(physicsClientId=self.engine._client)
        self.engine.disconnect()
        self.connected = False

    def step_physics(self):
        pass

    def step_render(self):
         """ Iterates the graphics visualization at given FPS. """
         for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()
         glRotatef(1, 1, 1, 1)
         glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
         
         for ent_st in self.entitites.values():
             vertices = ent_st['vertices']
             glBegin(GL_QUADS)                                  # start drawing a rectangle
             # glBegin(GL_LINES)
             vertices = self.vertices
             glVertex2f(vertices[0][0], vertices[0][1])                                   # bottom left point
             glVertex2f(vertices[1][0], vertices[1][1])                                   # bottom left point
             glVertex2f(vertices[2][0], vertices[2][1])                                   # bottom left point
             glVertex2f(vertices[0][0], vertices[0][1])                                   # bottom left point
             glEnd()
         pg.display.flip()
         pg.time.wait(10)
       

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
        obj.physics_client = self
        idx = 1 if len(self.entities) == 0 else 1 + max([*self.entities.keys()])
        ori = obj.init_orientation[-1] if not isinstance(type(obj.init_orientation), list) else obj.init_orientation  
        self.entities[idx] = {'position' : obj.init_position, 'orientation' : ori, 'vertices' : obj.vertices}
        obj.id = idx

    def parse_urdf(self, obj):
        pass

    @property
    def client(self):
        """ Pybullet engine client used in the simulation. """
        return self

    def control_joints(self, obj_id, joints, actions, control_type='velocity'):
        """
        """
        max_speed = 1
        v_motors = max_speed * actions 
        current_pos = self.entities[obj_id]['position'] 
        current_theta = self.entities[obj_id]['orientation'] 
        # v_motors[np.abs(v_motors)] < min_thresh] = 0.0
        delta_t = 0.1
        robot_radius = 9
        R = .5 * robot_radius * v_motors.sum() / (v_motors[0] - v_motors[1] + 1e-3)
        w = (v_motors[0] - v_motors[1] + 1e-3) / (robot_radius * .5)
        icc = current_pos + R * np.array([-np.sin(current_theta), np.cos(current_theta)])
        transf_mat = lambda x: np.array([[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]])
        self.delta_pos = transf_mat(w * delta_t).dot(current_pos - icc) + icc - current_pos
        self.delta_theta = w * delta_t
        new_pos = current_pos + self.delta_pos.astype(float)
        self.entities[obj_id]['position'] = new_pos
        self.entities[obj_id]['orientation'] = (self.entities[obj_id]['orientation'] + self.delta_theta) % 2*np.pi  

    def read_joints(self, obj_id, joints):
        """ Reads the position (rad) and velocity (rad/s) of the requested joints of a robot. It returns 
        """
        pass

    def get_body_position(self, identifier, body_id, z_offset=0.0):
        """
        """
        return self.entities[identifier]['position']
    
    def get_body_orientation(self, identifier, body_id):
        """ 
        """
        return self.entities[identifier]['orientation']
         

    def get_body_velocity(self, identifier, body_id):
        """ 
        """
        return None 

    def get_body_angular_velocity(self, identifier, body_id):
        """ 
        """
        return None

    def set_body_state(self, identifier, body_id, position, orientation):
        """ 
        """
        self.entities[identifier]['position'] = position
        self.entities[identifier]['orientation'] = orientation

    def ray_cast(self, origin, destination):
        pass 

    def get_closest_point(self, idA, idB, linkA=-1, linkB=-1, max_dist=10):
        pass

    def get_contact_points(self, obj_id, ghost_ids=None):
        pass

    def get_link_state(self, obj_id, link_idx):
        """ Getter of the position and orientation of a given link in the specified entity.
        
        :param int obj_id: identifier of the WorldObject entity.
        :param int link_idx: identifier of the link of the WorldObject entity.

        :returns: ``tuple`` with the 3D numpy position and 3D orientation of the link.
        """
        return self.entities[obj_id]['position'], self.entities[obj_id]['orientation']

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
