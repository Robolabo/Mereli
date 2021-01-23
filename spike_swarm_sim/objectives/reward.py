import numpy as np
import numpy.linalg as LA
from spike_swarm_sim.utils import angle_mean, angle_diff


class AlignmentReward:
    def __init__(self):
        self.required_info = ("robot_positions", "robot_orientations",)
    
    def __call__(self, actions, states, info=None):
        thetas = info['robot_orientations']
        angle_errs = np.mean([angle_diff(th1, th2)\
                            for j, th1 in enumerate(thetas)\
                            for i, th2 in enumerate(thetas) if i != j])
        rA = 1 - (angle_errs / np.pi) ** 0.7
        rB = 1 - np.mean([np.abs(ac['wheel_actuator'][0]) for ac in actions])
        return 0.7 * rA + 0.3 * rB



class GoToLightReward:
    def __init__(self):
        self.required_info = ("robot_positions", "light_positions")

    def __call__(self, actions, states, info=None):
        # positions = info['robot_positions']
        # light_pos = info['light_positions']
        # distances = [LA.norm(robot_pos - light_pos) for robot_pos in positions]

        # rew = np.mean([np.clip(1 - (dist / 100), a_min=0, a_max=1) for dist in distances])
        # return rew
        if np.max(states['light_sensor']) > 0:
            return np.max(states['light_sensor']) ** 0.5
        else:
            return -.5
