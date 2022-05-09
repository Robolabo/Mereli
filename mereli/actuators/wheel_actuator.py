import numpy as np
from .base_actuator import Actuator
from mereli.register import actuator_registry

@actuator_registry(name='wheel_actuator')
class WheelActuator(Actuator):
    """ Robot wheel actuator using a differential drive system. 
    """
    def __init__(self, *args, robot_radius=0.11, dt=0.01, min_thresh=0.0, **kwargs):
        super(WheelActuator, self).__init__(*args, **kwargs)
        self.robot_radius = robot_radius
        self.dt = 5e-3
        self.delta_pos = np.zeros(2)
        self.delta_theta = 0
        self.min_thresh = min_thresh
    
    def step(self, v_motors):
        if isinstance(v_motors, list):
            v_motors = np.array(v_motors)
        current_pos = self.actuator_owner.position
        current_theta = self.actuator_owner.orientation
        v_motors[np.abs(v_motors) < self.min_thresh] = 0.0
        delta_t = self.dt
        R = .5 * self.robot_radius * v_motors.sum() / (v_motors[0] - v_motors[1] + 1e-3)
        w = (v_motors[0] - v_motors[1] + 1e-3) / (self.robot_radius * .5)
        icc = current_pos + R * np.array([-np.sin(current_theta), np.cos(current_theta)])
        transf_mat = lambda x: np.array([[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]])
        self.delta_pos = transf_mat(w * delta_t).dot(current_pos - icc) + icc - current_pos
        self.delta_theta = w * delta_t
        new_pos = self.actuator_owner.position + self.delta_pos.astype(float)
        print(self.actuator_owner.position, new_pos)
        self.actuator_owner.position = new_pos
        self.actuator_owner.orientation = self.actuator_owner.orientation + self.delta_theta
        self.actuator_owner.orientation = self.actuator_owner.orientation % (2 * np.pi)
        self.delta_pos = np.zeros(2)
        self.delta_theta = 0.0


@actuator_registry(name='disc_wheel_actuator')
class DiscreteWheelActuator(Actuator):
    """ 
    """
    def __init__(self, *args, max_velocity=8, **kwargs):
        super(DiscreteWheelActuator, self).__init__(*args, **kwargs)
        self.max_velocity = max_velocity
        self.joint_ids = [0,1]
    
    def step(self, action):
        v_mot = np.array({
            0 : [0,0], # Stop
            1 : [1,1], # Go straight
            2 : [1,-1], # Turn Right
            3 : [-1,1], # Turn Left
            4 : [-1,-1], # Go back
        }.get(action[0], [0,0]))
        v_mot *= self.max_velocity # Convert range [-1,1] to [-w_max, w_max].
        self.physics_client.control_joints(self.owner_id, self.joint_ids, v_mot, control_type='velocity')

            
    def reset(self,):
        """ Resets the actuator."""
        if self.physics_client is not None:
            self.physics_client.control_joints(self.owner_id, self.joint_ids, np.zeros(len(self.joint_ids)), control_type='velocity')
        