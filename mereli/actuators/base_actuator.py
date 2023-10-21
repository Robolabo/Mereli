
class Actuator:
    """ Base class for actuators. Any actuator to be implemented in the simulator must 
    inherit (directly or indirectly) from this base class.

    :param Robot actuator_owner: instance of the agent executing the actuator.
    """
    def __init__(self, actuator_owner):
        self.actuator_owner = actuator_owner
        self.action = None
    

    def step(self):
        """ Method to execute an iteration of the actuator. In this base class it is empty and 
        must be overwritten by actuators inheriting from it in order to particularize their 
        functionality. Essentially, this method transforms actions planned by the robot controller 
        into real interactions with the environment.

        :param np.ndarray action: vector or scalar action to be executed.
        """
        raise NotImplementedError

    @property
    def physics_client(self):
        """ Returns the physics engine client of the robot owning the actuator. """
        return self.actuator_owner.physics_client

    @property
    def owner_id(self):
        """ Returns the identifier of the robot owning the actuator. """
        return self.actuator_owner.id
    
    @property
    def robot(self):
        return self.actuator_owner

class HighLevelActuator(Actuator):
    """ Base class for high level actuators. A high level actuator is an actuator that either 
    makes use of global information that would not be available in reality or that simplifies 
    the implementation of an more complex actuator for easing simulation. For the moment this 
    class is completely void and it is merely used to keep track of which actuators are high 
    level and which are not. 

    :param Robot actuator_owner: instance of the agent executing the actuator.
    """
    def __init__(self, *args, **kwargs):
        super(HighLevelActuator, self).__init__(*args, **kwargs)


    def step(self, action, neighborhood):
        """ Method to execute an iteration of the actuator. In this base class it is empty and 
        must be overwritten by high level actuators inheriting from it in order to particularize their 
        functionality. Essentially, this method transforms actions planned by the robot controller 
        into real interactions with the environment.

        :param np.ndarray action: vector or scalar action to be executed.
        """
        raise NotImplementedError
