import numpy as np
import numpy.linalg as LA
from spike_swarm_sim.utils import angle_mean, angle_diff
from spike_swarm_sim.register import reward_registry


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


@reward_registry(name='goto_light')
class GoToLightReward:
    def __init__(self):
        self.required_info = ("generation", "robot_positions", "light_positions")

    def __call__(self, actions, states, info=None):
        # positions = info['robot_positions']
        # light_pos = info['light_positions']
        # distances = [LA.norm(robot_pos - light_pos) for robot_pos in positions]

        # rew = np.mean([np.clip(1 - (dist / 100), a_min=0, a_max=1) for dist in distances])
        # return rew
        
        rew_obst = -1. if np.max(states['distance_sensor3D']) > 0.4 else 0.0
        rew_ls = 1. if np.max(states['light_sensor3D']) > 0.4 else 0.0
        return  rew_obst + rew_ls


@reward_registry(name='transport_cube')
class TransportCubeReward:
    def __init__(self):
        self.t = 0
        self.required_info = ("generation", "robot_positions", "light_positions")

    def __call__(self, actions, states, entity_name=None, info=None):
        # positions = info['robot_positions']
        # light_pos = info['light_positions']
        # distances = [LA.norm(robot_pos - light_pos) for robot_pos in positions]

        # rew = np.mean([np.clip(1 - (dist / 100), a_min=0, a_max=1) for dist in distances])
        # return rew
        cubes_pos = np.array([obj.position for obj in info.values() if type(obj).__name__ == 'Cube'])
        cubes_vel = np.array([obj.velocity for obj in info.values() if type(obj).__name__ == 'Cube'])
        cubes_vel[cubes_vel < 1e-2] = 0
        ground_area_pos = np.array([obj.position for obj in info.values() if type(obj).__name__ == 'GroundArea'])
        ground_area_rad = np.array([obj.radius for obj in info.values() if type(obj).__name__ == 'GroundArea'])
        robot_pos = info[entity_name].position
        # robot_vel = info[entity_name].velocity
        if cubes_vel[:, :2].sum() > 0.1: import pdb; pdb.set_trace()

        mask = np.linalg.norm(cubes_pos[:,:2] - robot_pos[:2], axis=1) < 1
        cubes_vel[mask]
        return 0