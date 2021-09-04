from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import pybullet as p
from spike_swarm_sim.utils import isinstance_of_any

class WorldObject(ABC):
    """ 
    Base class for abstract world objects or entities. This class is the most basic class of
    world entities and only implements abstract properties of objects. It does not 
    implement positions, orientations, and so on. This class must not be directly instantiated 
    and all world objects have to inherit from it indirectly.

    :param str model_file: name of the file where the model of the entity is stored.
    :param position:
    :param orientation: 
    :param bool static: whether the object is static or can move.
    :param Controller controller: controller, if any, defining object behavior. 
        If the object cannot be controlled, then controller=None. 
    :param bool tangible: whether the object has collisions or not.
    :param bool luminous: whether the object emits light or not.
    :param bool trainable: whether the object controller can be trained (not really used yet).
    :param float z_offset: z-axis offset of the entity position. It is only valid for 3D entities. 
        If entity is not 3D, then it should be left as ``None``.

    :var Engine physics_client: physics and render client used in the simulations.
    :var int id: unique identifier of the entity (should not be changed unless strictly needed). 
    """
    def __init__(self, model_file, position, orientation, 
                    static=True, controller=None, tangible=True,
                    luminous=False, trainable=False, z_offset=None):
        self.model_file = model_file
        if model_file is not None:
            self.model_file = model_file + ".urdf"  if '.urdf' not in model_file else model_file # Hcer algo con esto            
            self.model_file = "spike_swarm_sim/models/" + self.model_file
        self.init_position = position
        self.init_orientation = orientation
        if isinstance_of_any(orientation, [float, int]):
            self.init_orientation = [0,0,orientation]
        self.z_offset = z_offset
        self.static = static
        self.controller = controller
        self.tangible = tangible
        self.luminous = luminous
        self.trainable = trainable
        self.physics_client = None
        self._id = None
        self.group = None

    @abstractmethod
    def step(self):
        """ Abstract step method that is executed in every simulation cycle. The clearest function concerns 
        robots, that read sensors and execute the controller. 
        """
        raise NotImplementedError
    
    @abstractmethod
    def reset(self, seed=None):
        """ Abstract reset method that is executed at the begginning of every simulation. It normally resets all 
        the dynamical variables of the entity (position, orientation, controller, etc.). It can receive a seed in 
        order to be reset at a precise random state.

        :param int seed: seed to reset at a precise random state. None if no seed is used.
        """
        raise NotImplementedError

    @property 
    def position(self):
        """ 
        Getter method of the current position of the entity.

        :returns: numpy array with the entities' position.
        """
        pos = self.physics_client.get_body_position(self.id, 0)
        if self.z_offset is not None:
            pos[-1] = self.init_position[-1] + self.z_offset
        return pos
    
    @property
    def orientation(self):
        """ 
        Getter method of the current orientation of the entity.

        :returns: numpy array with the entities' orientation if 3D or
            ``float`` if 2D.
        """
        return self.physics_client.get_body_orientation(self.id, 0)

        
    @property
    def velocity(self):
        """ 
        Getter method of the current velocity of the entity.

        :returns: numpy array with the entities' velocity.
        """
        return self.physics_client.get_body_velocity(self.id, 0)

    @property
    def angular_velocity(self):
        """ 
        Getter method of the current angular velocity of the entity.

        :returns: numpy array with the entity angular velocity.
        """
        return self.physics_client.get_body_angular_velocity(self.id, 0)

    @position.setter
    def position(self, new_position):
        """ Setter of the position of the entity. 
        
        :param np.ndarray new_position: numpy array with the new position of the entity.
        """
        self.physics_client.set_body_state(self.id, 0, new_position, self.orientation)
        
    @orientation.setter
    def orientation(self, new_orientation):
        """ Setter of the orientation of the entity. 
        
        :param np.ndarray new_orientation: numpy array with the new orientation of the entity.
        """
        self.physics_client.set_body_state(self.id, 0, self.position, new_orientation)

    @property
    def id(self):
        """ Getter of the unique object's id.
        
        :returns: int id of the entity.
        """
        return self._id

    @id.setter
    def id(self, new_id):
        """ Setter of the unique object id. Do not use outside the program or during a simulation.

        :param int new_id: new id of the object.
        """
        self._id = new_id

    @property
    def controllable(self):
        """ Getter of a flag denoting whether the object 
        can be controlled or not.

        :returns: bool stating if the entity is controllable.
        """
        return self.controller is not None


    
  
# class WorldObject2D(WorldObject):
#     """
#     Base class for 2D world objects (robots, lights, walls, and so on). 
#     This class must not be directly instantiated and all 2D world objects have to
#     inherit from it.

#     :param str model_file: name of the json file stored at spike_swarm_sim/objects/urdf describing 
#         the entity 2D model.
#     :param np.ndarray position: 2D position vector of the entity.
#     :param float position: orientation in radians of the entity.
#     """
#     def __init__(self, model_file, position, orientation, *args, **kwargs):
#         super(WorldObject2D, self).__init__(*args, **kwargs)
#         self.model_file = model_file
#         self.init_position = position.astype(float) if isinstance(position, np.ndarray) else position
#         self.init_orientation = orientation
#         self.physics_client = None

#     def step(self):
#         raise NotImplementedError

#     @property
#     def position(self):
#         pos = self.physics_client.get_body_position(self.id, 0)
#         return (np.array([pos.x, pos.y]) - 500) / 100 
    
#     @property
#     def orientation(self):
#         return self.physics_client.get_body_orientation(self.id, 0)

#     @position.setter
#     def position(self, new_position):
#         """ Setter of the position. """
#         new_position = new_position * 100 + 500        
#         # if any(new_position > 1000):import pdb; pdb.set_trace()
#         self.physics_client.reset_body_position(self.id, 0, tuple(new_position))
    
#     @orientation.setter
#     def orientation(self, new_orientation):
#         """ Setter of the orientation. """
#         self.physics_client.reset_body_orientation(self.id, 0, new_orientation)


# class WorldObject3D(WorldObject):
#     """ Base class for 3D world objects (robots, lights, walls, and so on). 
#     This class must not be directly instantiated and all 3D world objects have to
#     inherit from it.
#     ====================================================================================
#     - Params:
#         urdf_file [str]: extension less name of the URDF file defining the object.
#         position [np.ndarray or list]: position 3D vector of the object.
#         orientation [np.ndarray or list]: 3D Euler orientation vector.
#         physics_client [int]: identifier of the pybullet physics server.
#         z_offset [float]
#     ====================================================================================
#     """
#     def __init__(self, model_file, position, orientation, *args, z_offset=0, **kwargs):
#         super(WorldObject3D, self).__init__(*args, **kwargs)
#         self.model_file = model_file + ".urdf"
#         if len(model_file.split('/')) < 2 or 'tmp' in model_file:
#             self.model_file = "spike_swarm_sim/objects/urdf/" + self.model_file
#         self.init_position = position
#         self.init_orientation = orientation
#         self.z_offset = z_offset
#         self._id = None
#         self.physics_client = None

#     def step(self):
#         raise NotImplementedError
    
#     def reset(self, seed=None):
#         raise NotImplementedError

#     @property
#     def position(self):
#         #TODO: move z_offset to physics_engine 3D. 
#         pos = self.physics_client.get_body_position(self.id, 0)
#         pos[-1] = self.init_position[-1] + self.z_offset
#         return pos
 
#     @property
#     def orientation(self):
#         return self.physics_client.get_body_orientation(self.id, 0)
        
#     @property
#     def velocity(self):
#         return self.physics_client.get_body_velocity(self.id, 0)

#     @property
#     def angular_velocity(self):
#         return self.physics_client.get_body_angular_velocity(self.id, 0)

#     @position.setter
#     def position(self, new_position):
#         """ Setter of the position. """
#         self.physics_client.set_body_state(self.id, 0, new_position, self.orientation)
        
#     @orientation.setter
#     def orientation(self, new_orientation):
#         """ Setter of the orientation. """
#         self.physics_client.set_body_state(self.id, 0, self.position, new_orientation)