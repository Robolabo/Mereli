import logging
import pybullet as p
import numpy as np
from .base_actuator import Actuator
from spike_swarm_sim.register import actuator_registry

@actuator_registry(name='joint_actuator')
class JointActuator(Actuator):
    """ Robot wheel actuator using a differential drive system. 
    """
    def __init__(self, *args, joint_ids=[0], control='velocity', **kwargs):
        super(JointActuator, self).__init__(*args, **kwargs)
        self.joint_ids = joint_ids
        self.control = {
            'position' : p.POSITION_CONTROL,
            'velocity' : p.VELOCITY_CONTROL
        }[control]
        self.max_velocity = 30.0
        #for j in range(2):
        #      p.changeDynamics(self.actuator_owner.id, j,\
        #         mass=0.01, physicsClientId=self.actuator_owner.physics_client)
         
    def step(self, action):
        if len(action) != len(self.joint_ids):
            raise Exception(logging.error('Size of the action in Joint Actuator differs from '\
            	'the number of controllable joints.'))
        #action = np.random.choice([-1, -0.5, 0, 0.5, 1], size=2)
        action = [1, 1.]
        for ac, joint in zip(action, self.joint_ids):
            p.setJointMotorControl2(self.actuator_owner.id,joint, targetVelocity=ac * self.max_velocity,\
                controlMode=p.VELOCITY_CONTROL, physicsClientId=self.actuator_owner.physics_client, )
