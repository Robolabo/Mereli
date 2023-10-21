import numpy as np
import pybullet as p
from mereli.register import sensor_registry
from mereli.utils import compute_angle
from mereli.sensors import Sensor

@sensor_registry(name='joint_position_sensor')
class JointPositionSensor(Sensor):
    """ Sensor of the position (in radians) of the joints of the robot. 

    :param list joints: list of joints to be read.
    """
    def __init__(self, *args, joints=[0], **kwargs):
        super(JointPositionSensor, self).__init__(*args, **kwargs)
        self.joints = joints
        self.reading = np.zeros(len(self.joints))
        
    def step(self):
        """ Reads the current position of the requested joints. It queries this operation 
        to the physics engine by calling the method ``Engine.read_joints``.

        :returns: numpy array with the position reading of each joint.
        
        """
        joint_states = self.physics_client.read_joints(self.owner_id, self.joints)[0]
        self.reading = np.array(joint_states)

    def reset(self):
        self.reading = np.zeros(len(self.joints))

@sensor_registry(name='encoder')
class Encoder(Sensor):
    """ Sensor of the position (in radians/s) of the joints of the robot.

    :param list joints: list of joints to be read.
    """
    def __init__(self, *args, joints=[0,1], **kwargs):
        super(Encoder, self).__init__(*args, **kwargs)
        self.joints = joints
        
    def step(self):
        """ Reads the current velocity of the requested joints. It queries this operation 
        to the physics engine by calling the method ``Engine.read_joints``.
        
        :returns: numpy array with the velocity reading of each joint.
        """
        joint_states = self.physics_client.read_joints(self.owner_id, self.joints)[1]
        self.reading = np.array(joint_states)

    def reset(self):
        self.reading = np.zeros(len(self.joints))
