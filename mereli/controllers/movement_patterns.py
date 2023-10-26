import numpy as np
from mereli.controllers import RobotController, BasicObstacleAvoider
from mereli.register import controller_registry


@controller_registry(name='snake_movement')
class SnakeMovement(RobotController):
    """
    """
    def step(self, state, reward=0.0):
        f =  0.2
        action_wheels = 0.5 * (1 + np.array([np.cos(2*np.pi*f*self.t*0.01), np.sin(2*np.pi*f*self.t*0.01)]))
        self.get_actuator('joint_velocity_actuator').action = action_wheels
