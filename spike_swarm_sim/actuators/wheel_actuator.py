import numpy as np
from spike_swarm_sim.register import actuator_registry

@actuator_registry(name='wheel_actuator')
class WheelActuator:
    """ Robot wheel actuator using a differential drive system. 
    """
    def __init__(self, robot_radius, dt=1.65, min_thresh=0.0):
        self.robot_radius = robot_radius
        self.dt = dt
        self.delta_pos = np.zeros(2)
        self.delta_theta = 0
        self.min_thresh = min_thresh
    
    def step(self, v_motors, current_pos, current_theta, ):
        if isinstance(v_motors, list):
            v_motors = np.array(v_motors)
        v_motors[np.abs(v_motors) < self.min_thresh] = 0.0
        delta_t = self.dt
        R = .5 * self.robot_radius * v_motors.sum() / (v_motors[0] - v_motors[1] + 1e-3)
        w = (v_motors[0] - v_motors[1] +1e-3) / (self.robot_radius * .5)
        icc = current_pos + R * np.array([-np.sin(current_theta), np.cos(current_theta)])
        transf_mat = lambda x: np.array([[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]])
        self.delta_pos = transf_mat(w * delta_t).dot(current_pos - icc) + icc - current_pos
        self.delta_theta = w * delta_t