from abc import ABC, abstractmethod
from enum import Enum
from spike_swarm_sim.globals import global_states

# class EngineDim(Enum):
#     Engine3D 

class BaseEngine(ABC):
    """ Base abstract class of physics and render Engines. 
    Its role in the simulation is to iterate the physic simulations and collision detections of the 
    entities in the environment and render the graphics. Precise engine implementations 
    must inherit from this class, properly overwritting the required abstract methods to cover the 
    engine's basic functionalities. The implementation can be either supported on an extern physics simulations
    library or coded from zero. 

    :param str engine_type: either 2D or 3D indicating the dimensions of the space where physics are simulated.
        TODO: change to Enum.
    :param float dt: time step of the physics simulation (in seconds).
    :param float T_control: period of the sensing+control+action loop. It cannot be lower than dt and 
        it is set to 5*dt by default. Essentially this means that the physics are updated 5 times 
        in between every executing of sensors, controllers and actuators.

    :var BulletClient engine: pybullet client engine.
    :var bool render: flag indicating if the simulation is run in visual or render mode.
    :var bool connected: whether the engine is connected or not.
    """
    def __init__(self, engine_type, dt=0.02, T_control=0.1):
        self._engine_type = engine_type
        self.dt = dt
        self.T_control = T_control
        assert T_control >= dt
        self.connected = False
        self.engine = None

    @property
    def engine_type(self):
        return self._engine_type

    @property
    def render(self):
        """ Flag indicating whether the simulation is running in visual or in compute mode."""
        return global_states.RENDER

    @abstractmethod
    def connect(self, objects):
        """ Abstract method for connecting to physics and render engines. 
        Some common actions to be taken, apart from creating the engine client, are to set 
        either visual or direct mode, set up all the physics constants (gravity, sampling period, etc.) 
        or adding all the ``WorldObjects`` to the engine.

        :param iterable objects: iterable of WorldObjects whose physics have to be simulated.
        """
        pass
    
    @abstractmethod
    def disconnect(self):
        """ Abstract method for disconnecting the physics and render engines. """
        pass
    
    @abstractmethod
    def step_physics(self):
        """ Abstract method for iterating all the 3D physics and dynamical states of the 
        world entities using the implemented engine. """
        pass
    
    @abstractmethod
    def step_render(self):
        """ Abstract method for iterating the graphics or renderings. """
        pass

    @abstractmethod
    def add_physics(self, obj):
        """
        Abstract method for adding the requested entity to the engine so that its physics can be taken into account 
        during the simulation. 

        :param WorldObject obj: entity to be added to the engine.
        """
        pass

    def add_objects(self, objects):
        """
        Iteratively add all the WorldObject entities to the engine so that its physics can be taken into account 
        during the simulation.

        :param iterable objects: iterable of WorldObjects whose physics have to be simulated.
        """
        for obj in objects:
            self.add_physics(obj)

    @abstractmethod
    def get_body_position(self, identifier):
        """ Abstract getter method of the current position of the root link of an entity with the 
        given identifier.

        :param int identifier: identifier of the entity whose position is requested.

        :returns: numpy array of shape N (N=2 if 2D or N=3 if 3D)  with the entities' position.
        """
        pass

    @abstractmethod
    def get_body_orientation(self, identifier):
        """ Abstract getter method of the current Euler orientation of the root link of an entity with the 
        given identifier.

        :param int identifier: identifier of the entity whose position is requested.

        :returns: numpy array with the entities' Euler orientation (if 2D then only yaw angle).
        """
        pass

    @abstractmethod
    def get_body_velocity(self, identifier, body_id):
        """ Abstract getter method of the current velocity of the root link of an entity with the 
        given identifier.

        :param int identifier: identifier of the entity whose position is requested.

        :returns: numpy array with the entities' velocity.
        """
        pass

    @abstractmethod
    def get_body_angular_velocity(self, identifier):
        """ Abstract getter method of the current angular velocity of the root link of an entity with the 
        given identifier.

        :param int identifier: identifier of the entity whose position is requested.

        :returns: numpy array  with the entities' angular velocity.
        """
        pass

    @abstractmethod
    def set_body_state(self, identifier, position, orientation):
        """ Abstract setter of the physics state (position and orientation) of a registered entity with the given
        identifier. 

        :param int identifier: identifier of the entity whose position is requested.
        :param np.ndarray position: new 3D or 2D position of the entity.
        :param np.ndarray orientation: new 3D or scalar (yaw) Euler orientation of the entity. It is also possible to 
            introduce an angle scalar in radians so that orientation = [0,0,orientation].
        """
        pass

    @abstractmethod
    def ray_cast(self, origin, destination):
        """ Abstract method for casting a batch of rays between pairwise coordinates in origin and destination lists
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
        pass

    @abstractmethod
    def get_link_state(self, obj_id, link_idx):
        """ Abstract getter of the position and orientation of a given link in the specified entity.
        
        :param int obj_id: identifier of the WorldObject entity.
        :param int link_idx: identifier of the link of the WorldObject entity.

        :returns: ``tuple`` with the numpy position and orientation of the link.
        """
        pass

    @abstractmethod
    def get_sensor_position(self, obj_id, sensor_name, sector=0):
        """ Abstract Getter of the physical position of a sensor within a robot. Sensors are attached to 
        robot links and, therefore, it returns the 3D coordinates of the corresponding link.
        
        :param int obj_id: identifier of the robot owning the sensor.
        :param str sensor_name: reference name of the sensor.
        :para int sector: index of the sensor's sector requested.

        :returns: numpy array with the position. 
        """
        pass

    @abstractmethod
    def get_actuator_position(self, obj_id, actuator_name, sector=0):
        """ TODO 
        """
        pass

    @abstractmethod
    def get_sensor_orientation(self, obj_id, sensor_name, sector=0):
        """ Abstract Getter of the physical position of a sensor within a robot. Sensors are attached to 
        robot links and, therefore, it returns the 3D coordinates of the corresponding link.
        
        :param int obj_id: identifier of the robot owning the sensor.
        :param str sensor_name: reference name of the sensor.
        :para int sector: index of the sensor's sector requested.

        :returns: numpy array with the position. 
        """
        pass

    @abstractmethod
    def control_joints(self, obj_id, joints, actions, control_type='velocity'):
        """ Abstract method for controlling a series of robot joints either by velocity or by position.
        
        :param int obj_id: identifier of the robot whose joints will be controlled.
        :param list joints: list of identifiers of the joints of the robot to be controlled.
        :param list actions: list or np.ndarray of actions to control each joint.
        :param str control_type: type of joint control (either "velocity" or "position").
        """
        pass

    @abstractmethod
    def read_joints(self, obj_id, joints):
        """ Reads the position (rad) and velocity (rad/s) of the requested joints of a robot. It returns 
        a tuple (joint_velocities, joint_positions) with the arrays of the measurements of each kind.

        :param int obj_id: identifier of the robot whose joints will be read.
        :param list joints: list of identifiers of the joints of the robot to be read.

        :returns: tuple of the form (joint_velocities, joint_positions) with the arrays of the measurements of each kind.
        """
        pass

    def get_closest_point(self, idA, idB, linkA=-1, linkB=-1, max_dist=10):
        """ Computes the closest points between two links of two registered entities.

        :param int idA: identifier of the first WorldObject entity.
        :param int idB: identifier of the second WorldObject entity.
        :param int linkA: identifier of the link of the first WorldObject entity.
        :param int linkB: identifier of the link of the second WorldObject entity.
        :param float max_dist: maximum distance between the objects.

        :returns: numpy array with the coordinates of the closest point in linkB of entity with idB. 
        """
        pass

    def get_contact_points(self, obj_id, ghost_ids=None):
        """ Computes the contact points between any link of the given entity and any other 
        entity. It steps the collision detection engine to perform the query. The identifier 
        of ghost links can be specified in order to momentarily activate collisions and detect 
        obstacles.

        :param int obj_id: identifier of the WorldObject entity.
        :param list ghost_ids: list of the identifiers of the entity ghost links.

        :returns: list of tuples, each composed by the following entries: (objB_id, linkA_id, linkB_id).
        """
        pass

    def set_color(self, obj_id, link_id, color, opacity=1.0):
        """ Abstract setter of the color and opacity of a link of an entity. 
        
        :param int obj_id: identifier of the robot owning the sensor.
        :param int obj_id: identifier of the link of the robot owning the sensor whose color is changed.
        :param list color: ``list`` with the RGB code of the color or ``str`` with the color name.
        :para float opacity: opacity of the color.
        """
        pass

    def set_camera_focus(self, position, distance, yaw=0, pitch=-90):
        """ Abstract setter of the camera target position and distance in the environment. 
        The camera spotlight is set to the given position and the camera it placed at the given 
        distance wrt to that position. Yaw and pitch in degrees can be also specified. 
        This method must only be applied in render mode.

        :param np.ndarray position: new spotlight of the camera.
        :param float distance: distance of the camera wrt to the spotlight.
        :param float yaw: yaw angle of the camera.
        :param float pitch: pitch angle of the camera.
        """
        pass
