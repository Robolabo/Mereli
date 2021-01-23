import numpy as np
from spike_swarm_sim.controllers import Controller
from spike_swarm_sim.register import controller_registry
from spike_swarm_sim.utils import compute_angle, normalize, toroidal_difference, increase_time

@controller_registry(name='light_orbit_controller')
class LightOrbitController(Controller):
    """ Class for the control of light sources as orbits
    around a central position.
    There is a small probability of rotation sense inversion. 
    """
    def __init__(self):
        self.t = 1
        self.dir = 1

    @increase_time
    def step(self, pos):
        new_pos = pos.copy()
        self.dir = np.random.choice([self.dir, -self.dir], p=[0.99, 0.01])
        current_angle = compute_angle(new_pos - np.array([500, 500]))
        new_angle = current_angle + self.dir * 0.013 #0.012 #0.01
        new_rad = min(np.linalg.norm(new_pos - np.array([500, 500])) + 1, 200)
        new_pos = new_rad*np.r_[np.cos(new_angle), np.sin(new_angle)] + np.array([500, 500])
        return new_pos

    def reset(self):
        self.t = 1
        self.dir = 1

@controller_registry(name='light_rnd_pos_controller')
class LightRndPositionController(Controller):
    """ Class for the control of light sources as straight 
    trajectories to randomly sampled goal locations. Every 
    200 time steps the goal location is resampled.
    """
    def __init__(self):
        self.t = 1
        self.tar_pos = np.random.uniform(400, 600, size=2)

    @increase_time
    def step(self, pos):
        new_pos = pos.copy()
        if self.t % 50 == 0:
            self.tar_pos = np.random.uniform(400, 600, size=2)
        new_pos = pos + 3 * normalize(self.tar_pos - pos)
        return new_pos

    def reset(self):
        self.t = 1
        self.tar_pos = np.random.uniform(400, 600, size=2)

@controller_registry(name='light_prey_controller')
class PreyController(Controller):
    """ Class for the control of light sources mimicking prey escape. 
    The controller deterministically computes the escape direction (steering) 
    based on the known positions of the predators. If a predator is at a distance 
    lower than 30 (3cm), then the prey is hunted and stops its motion.
    """
    def __init__(self):
        self.t = 1
        self.direction = np.r_[np.cos(np.pi/4), np.sin(np.pi/4)]
        self.hunted = 0

    @increase_time
    def step(self, my_pos, robot_positions):
        new_pos = my_pos.copy()
        robot_light_vecs = np.stack([toroidal_difference(robot_pos, my_pos) for robot_pos in robot_positions])
        distances = np.array([np.linalg.norm(v) for v in robot_light_vecs])
        near_robots = [d < 150 for d in distances]
        if any(near_robots):
            weights = (150 - distances[near_robots]) / 150
            weights /= sum(weights)
            self.direction = -normalize(np.dot(weights, robot_light_vecs[near_robots]))
        if not self.hunted:
            new_pos += 3. * self.direction
            self.hunted = any([np.linalg.norm(v) < 30 for v in robot_light_vecs])
        return new_pos

    def reset(self):
        self.t = 1
        self.hunted = 0