import logging
import pybullet as p
import numpy as np
from .base_actuator import Actuator
from spike_swarm_sim.register import actuator_registry

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

        .. todo::

            Quitar llamada explicita a pybullet y usar request a physics_client.

        """
        if len(action) != len(self.joint_ids):
            raise Exception(logging.error('Size of the action in Joint Actuator differs from '\
            	'the number of controllable joints.'))
        for ac, joint in zip(action, self.joint_ids):
            p.setJointMotorControl2(self.owner_id, joint, targetVelocity=ac * self.max_velocity,\
                controlMode=p.VELOCITY_CONTROL, physicsClientId=self.physics_client.client, velocityGain=1.1)
        if self.inverse_mirrored is not None and len(action) == 1: #! mejorar
            p.setJointMotorControl2(self.owner_id, self.inverse_mirrored,\
                targetVelocity=-action[0] * self.max_velocity, controlMode=p.VELOCITY_CONTROL,\
                physicsClientId=self.physics_client.client, velocityGain=1.1)

    def reset(self,):
        """ Resets the actuator."""
        if self.physics_client is not None:
            for joint in self.joint_ids:
                p.setJointMotorControl2(self.actuator_owner.id, joint, targetVelocity=0, velocityGain=0,\
                    controlMode=p.VELOCITY_CONTROL, physicsClientId=self.physics_client.client)

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

        .. todo::

            Quitar llamada explicita a pybullet y usar request a physics_client.
            
        """
        if len(action) != len(self.joint_ids):
            raise Exception(logging.error('Size of the action in Joint Actuator differs from '\
            	'the number of controllable joints.'))

        for ac, joint in zip(action, self.joint_ids):
            p.setJointMotorControl2(self.actuator_owner.id, joint, targetPosition=ac * np.pi,\
                controlMode=p.POSITION_CONTROL, physicsClientId=self.actuator_owner.physics_client.client,\
                positionGain=1.1, velocityGain=1.1, maxVelocity=self.max_velocity)
    
    # def reset(self,):
    #     for joint in self.joint_ids:
    #         p.setJointMotorControl2(self.actuator_owner.id, joint, targetVelocity=0, velocityGain=0,\
    #             controlMode=p.VELOCITY_CONTROL, physicsClientId=self.actuator_owner.physics_client)
