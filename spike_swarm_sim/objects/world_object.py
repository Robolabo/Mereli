from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import pybullet as p

class WorldObject(ABC):
    """ 
    Base class for abstract world objects. This class is the most basic class of
    world entities and only implements abstract properties of objects. It does not 
    implement positions, orientations, and so on. 
    This class must not be directly instantiated and all world objects have to
    inherit from it indirectly.
    ====================================================================================
    - Params:
        static [bool]: whether the object is static or can move.
        controller [Controller or None] : controller, if any, defining object behavior.
        tangible [bool]: whether the object could have collisions or not. #!(usefulness to be checked).
        luminous [bool]: whether the object emits light or not.
        trainable [bool]: whether the object controoler can be trained. (#!CHECK)
    ====================================================================================
    """
    def __init__(self, static=True, controller=None, tangible=True,\
                    luminous=False, trainable=False):
        self._id = None
        self.group = None
        self.static = static
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
        """ Getter of flag denoting whether the object 
        can be controlled or not.
        """
        return self.controller is not None

    @abstractmethod
    def step(self):
        raise NotImplementedError
    
    @abstractmethod
    def reset(self, seed=None):
        raise NotImplementedError

    @abstractproperty
    def position(self):
        raise NotImplementedError
    
    @abstractproperty
    def orientation(self):
        raise NotImplementedError
    
  
class WorldObject2D(WorldObject):
    """
    Base class for 2D world objects (robots, lights, walls, and so on). 
    This class must not be directly instantiated and all 2D world objects have to
    inherit from it.
    ====================================================================================
    - Params:
        position [np.ndarray or list]: position vector of the object.
    ====================================================================================
    """
    def __init__(self, model_file, position, orientation, *args, **kwargs):
        super(WorldObject2D, self).__init__(*args, **kwargs)
        self.model_file = model_file
        self.init_position = position.astype(float) if isinstance(position, np.ndarray) else position
        self.init_orientation = orientation
        self.physics_client = None

    def step(self):
        raise NotImplementedError

    @property
    def position(self):
        pos = self.physics_client.get_body_position(self.id, 0)
        return (np.array([pos.x, pos.y]) - 500) / 100 
    
    @property
    def orientation(self):
        return self.physics_client.get_body_orientation(self.id, 0)

    @position.setter
    def position(self, new_position):
        """ Setter of the position. """
        new_position = new_position * 100 + 500        
        # if any(new_position > 1000):import pdb; pdb.set_trace()
        self.physics_client.reset_body_position(self.id, 0, tuple(new_position))
    
    @orientation.setter
    def orientation(self, new_orientation):
        """ Setter of the orientation. """
        self.physics_client.reset_body_orientation(self.id, 0, new_orientation)


class WorldObject3D(WorldObject):
    """ Base class for 3D world objects (robots, lights, walls, and so on). 
    This class must not be directly instantiated and all 3D world objects have to
    inherit from it.
    ====================================================================================
    - Params:
        urdf_file [str]: extension less name of the URDF file defining the object.
        position [np.ndarray or list]: position 3D vector of the object.
        orientation [np.ndarray or list]: 3D Euler orientation vector.
        physics_client [int]: identifier of the pybullet physics server.
        z_offset [float]
    ====================================================================================
    """
    def __init__(self, urdf_file, position, orientation, *args, z_offset=0, **kwargs):
        super(WorldObject3D, self).__init__(*args, **kwargs)
        self.urdf_file = urdf_file + ".urdf"
        if len(urdf_file.split('/')) < 2 or 'tmp' in urdf_file:
            self.urdf_file = "spike_swarm_sim/objects/urdf/" + self.urdf_file
        self.init_position = position
        self.init_orientation = orientation
        self.z_offset = z_offset
        self._id = None
        self.physics_client = None

    def step(self):
        raise NotImplementedError
    
    def reset(self, seed=None):
        raise NotImplementedError

    @property
    def position(self):
        #TODO: move z_offset to physics_engine 3D. 
        pos = self.physics_client.get_body_position(self.id, 0)
        pos[-1] = self.init_position[-1] + self.z_offset
        return pos
 
    @property
    def orientation(self):
        return self.physics_client.get_body_orientation(self.id, 0)
        
    @property
    def velocity(self):
        return self.physics_client.get_body_velocity(self.id, 0)

    @property
    def angular_velocity(self):
        return self.physics_client.get_body_angular_velocity(self.id, 0)

    @position.setter
    def position(self, new_position):
        """ Setter of the position. """
        self.physics_client.set_body_state(self.id, 0, new_position, self.orientation)
        
    @orientation.setter
    def orientation(self, new_orientation):
        """ Setter of the orientation. """
        self.physics_client.set_body_state(self.id, 0, self.position, new_orientation)