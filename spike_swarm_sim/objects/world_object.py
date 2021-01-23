import numpy as np
# from shapely.geometry import LineString, Point, box, Polygon

class WorldObject(object):
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
    def __init__(self, pos, static, shape,
                controller=None, tangible=True, luminous=False,
                trainable=False):
        self._id = None
        
        self.pos = pos.astype(float) if isinstance(pos, np.ndarray) else pos
        self.init_pos = self.pos.copy() if isinstance(pos, np.ndarray) else pos
        
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