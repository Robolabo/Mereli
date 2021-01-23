import numpy as np
import pybullet as p
# from shapely.geometry import LineString, Point, box, Polygon

class WorldObject(object):
    """ 
    Base class for world objects (robots, lights, walls, and so on). 
    This class must not be directly instantiated and all world objects have to
    inherit from it.
    ====================================================================================
    - Params:
        position [np.ndarray or list]: position vector of the object.
        static [bool]: whether the object is static or can move.
        shape []
        controller [Controller or None] : controller, if any, defining object behavior.
        tangible [bool]:
        luminous [bool]: whether the object emits light or not.
        trainable [bool]: whether the object controoler can be trained. (#!CHECK)
    ====================================================================================
    """
    def __init__(self, position, static, shape,
                controller=None, tangible=True, luminous=False,
                trainable=False):
        self._id = None
        self.position = position.astype(float) if isinstance(position, np.ndarray) else position
        self.init_pos = self.position.copy() if isinstance(position, np.ndarray) else position
        
        self.static = static
        self._shape = shape if tangible else None
        self.controller = controller
        self.tangible = tangible
        self.luminous = luminous
        self.trainable = trainable

    @property
    def id(self):
        """ Getter of the unique object id."""
        return self._id

    @id.setter
    def id(self, new_id):
        """ Setter of the unique object id."""
        self._id = new_id

    @property
    def controllable(self):
        """
        Getter of flag denoting whether the object 
        can be controlled or not.
        """
        return self.controller is not None

    def step(self):
        raise NotImplementedError

    def render(self, canvas):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError



class WorldObject3D(object):
    """ 
    Base class for world objects (robots, lights, walls, and so on). 
    This class must not be directly instantiated and all world objects have to
    inherit from it.
    ====================================================================================
    - Params:
        pos [np.ndarray or list]: position vector of the object.
        static [bool]: whether the object is static or can move.
        shape []
        controller [Controller or None] : controller, if any, defining object behavior.
        tangible [bool]:
        luminous [bool]: whether the object emits light or not.
        trainable [bool]: whether the object controoler can be trained. (#!CHECK)
    ====================================================================================
    """
    def __init__(self, urdf_file, position, orientation, static=True,
                controller=None, tangible=True, luminous=False,
                trainable=False):
        urdf_file = "spike_swarm_sim/objects/urdf/" + urdf_file + ".urdf"
        self._id = p.loadURDF(urdf_file, position, p.getQuaternionFromEuler(orientation))        
        self.init_position, self.init_orientation = p.getBasePositionAndOrientation(self._id) 
        self.static = static
        self.controller = controller
        self.tangible = tangible
        self.luminous = luminous
        self.trainable = trainable

    @property
    def position(self):
        return np.array(p.getBasePositionAndOrientation(self._id)[0])

    @property
    def orientation(self):
        return np.array(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self._id)[1]))

    @position.setter
    def position(self, new_position):
        """ Setter of the position. """
        p.resetBasePositionAndOrientation(self.id, new_position, p.getQuaternionFromEuler(self.orientation))
    
    @orientation.setter
    def orientation(self, new_orientation):
        """ Setter of the orientation. """
        if len(new_orientation) == 1:
            new_orientation = [0., 0., new_orientation]
        p.resetBasePositionAndOrientation(self.id, self.position, p.getQuaternionFromEuler(new_orientation))

    @property
    def id(self):
        """ Getter of the unique object id."""
        return self._id

    @id.setter
    def id(self, new_id):
        """ Setter of the unique object id."""
        self._id = new_id

    @property
    def controllable(self):
        """
        Getter of flag denoting whether the object 
        can be controlled or not.
        """
        return self.controller is not None

    def step(self):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError