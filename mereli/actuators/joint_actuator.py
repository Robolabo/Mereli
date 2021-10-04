import logging
import pybullet as p
import numpy as np
from .base_actuator import Actuator
from mereli.register import actuator_registry

@actuator_registry(name='joint_velocity_actuator')
class JointVelocityActuator(Actuator):
    """ The joint velocity actuator controls the velocity of the joints of a robot. 
    Given a reference velocity (action), the actuator drives the velocity of the joint towards 
    the desired target value.

    :param list joint ids: list of the joint ids to be controlled.
    :param float max_velocity: maximum velocity of the joint (CHECK).
    :param bool inverse_mirrored: Whether to control two joints with the same velocity reference 
        but with opposite sign. For example, this feature is useful when controlling two-wheeled 
        mobile robots that only rotate. In this case, the robot controller would only plan an scalar 
        action with the velocity and sign.
    """
    def __init__(self, *args, joint_ids=[0], inverse_mirrored=None, max_velocity=10., **kwargs):
        super(JointVelocityActuator, self).__init__(*args, **kwargs)
        self.joint_ids = joint_ids
        self.max_velocity = max_velocity
        self.inverse_mirrored = inverse_mirrored

    def step(self, action):
        """ Steps the actuator by transforming the action planed by the robot controller into an actual 
        joint rotation speed. The action argument is a vector settling the desired reference velocity of 
        each joint to be controlled. The action references are contrained within [-1, 1], meaning that an 
        action of 1 means maximum velocity in one sense and -1 represents maximum velocity in the opposite 
        sense. An action of 0 means no rotation at all.

        :param np.ndarray action: numpy array collection the normalized reference velocities of each joint 
            to be controlled.
        """
        if len(action) != len(self.joint_ids):
            raise Exception(logging.error('Size of the action in Joint Actuator differs from '\
            	'the number of controllable joints.'))
        action *= self.max_velocity # Convert range [-1,1] to [-w_max, w_max].
        self.physics_client.control_joints(self.owner_id, self.joint_ids, action, control_type='velocity')
        if self.inverse_mirrored is not None and len(action) == 1: #! mejorar
            self.physics_client.control_joints(self.owner_id, [self.inverse_mirrored], 
                    [-action[0]], control_type='velocity')
            
    def reset(self,):
        """ Resets the actuator."""
        if self.physics_client is not None:
            self.physics_client.control_joints(self.owner_id, self.joint_ids, np.zeros(len(self.joint_ids)), control_type='velocity')


@actuator_registry(name='joint_position_actuator')
class JointPositionActuator(Actuator):
    """ The joint position actuator controls the position of the joints of a robot. 
    Given a reference angle position (action), the actuator drives the position of the joint towards 
    the desired target value.

    :param list joint ids: list of the joint ids to be controlled.
    :param float max_velocity: maximum velocity of the joint (CHECK).
    :param bool inverse_mirrored: Whether to control two joints with the same velocity reference 
        but with opposite sign. For example, this feature is useful when controlling two-wheeled 
        mobile robots that only rotate. In this case, the robot controller would only plan an scalar 
        action with the velocity and sign.
    """
    def __init__(self, *args, joint_ids=[0], **kwargs):
        super(JointPositionActuator, self).__init__(*args, **kwargs)
        self.joint_ids = joint_ids
        self.max_velocity = 1.
         
    def step(self, action):
        """ Steps the actuator by transforming the action planed by the robot controller into an actual 
        joint angle position. The action argument is a vector settling the desired reference positions of 
        each joint to be controlled. The action references are contrained within [-1, 1], which are mapped 
        into [-pi, pi] inside the method.

        :param np.ndarray action: numpy array collection the normalized reference velocities of each joint 
            to be controlled.            
        """
        if len(action) != len(self.joint_ids):
            raise Exception(logging.error('Size of the action in Joint Actuator differs from '\
            	'the number of controllable joints.'))
        action *= np.pi # Convert range [-1,1] to [-pi, pi].
        self.physics_client.control_joints(self.owner_id, self.joint_ids, action, control_type='position')
        
    # def reset(self,):
    #     for joint in self.joint_ids:
    #         p.setJointMotorControl2(self.actuator_owner.id, joint, targetVelocity=0, velocityGain=0,\
    #             controlMode=p.VELOCITY_CONTROL, physicsClientId=self.actuator_owner.physics_client)
