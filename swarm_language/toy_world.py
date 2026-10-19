""" Minimal 2D world to run :class:`LanguageAgent` swarms without a physics simulator.

It is meant for quick tests, demos and regression checks. It is NOT the setup of the paper
(no e-puck physics, no obstacle avoidance, no staggered control loop); use the Mereli
configs in ``mereli/config/PaperLanguageEmergence`` for that.

The world uses its own random generator, so the global ``np.random`` stream is consumed
only by the agents.
"""
import numpy as np
from .agent import LanguageAgent


class ToyWorld:
    """ Agents move straight in a square arena, turning randomly and at the walls.

    :param int n_agents: swarm size.
    :param list objects: object descriptions, dicts with ``'color'``, ``'shape'``, ``'size'``
        and optionally ``'position'`` (random if missing).
    :param float arena_size: side of the square arena (centered at the origin).
    :param float comm_range: agents closer than this hear each other.
    :param float sensing_range: an object is perceived if its surface is closer than this.
    :param float speed: displacement per step.
    :param float turn_prob: probability of a random heading change per step.
    :param int seed: seed of the world generator (positions, headings, movement).
    :param callable agent_factory: callable returning a new agent.
    """
    def __init__(self, n_agents, objects, arena_size=6.0, comm_range=1.0, sensing_range=0.6,
                 speed=0.05, turn_prob=0.05, seed=None, agent_factory=LanguageAgent):
        self.rng = np.random.RandomState(seed)
        self.arena_size = arena_size
        self.comm_range = comm_range
        self.sensing_range = sensing_range
        self.speed = speed
        self.turn_prob = turn_prob
        half = arena_size / 2
        self.objects = []
        for obj in objects:
            obj = dict(obj)
            if 'position' not in obj:
                obj['position'] = self.rng.uniform(-half, half, size=2)
            obj['position'] = np.asarray(obj['position'], dtype=float)
            self.objects.append(obj)
        self.agents = [agent_factory() for _ in range(n_agents)]
        self.positions = self.rng.uniform(-half, half, size=(n_agents, 2))
        self.headings = self.rng.uniform(0, 2 * np.pi, size=n_agents)
        self.t = 0

    def observed_object(self, i):
        """ Closest object within sensing range of agent ``i``, or None. """
        closest, closest_dist = None, 0
        for obj in self.objects:
            dist = np.linalg.norm(obj['position'] - self.positions[i]) - obj['size'] / 2
            if dist < self.sensing_range and (closest is None or dist < closest_dist):
                closest, closest_dist = obj, dist
        return closest

    def neighbors(self, i):
        """ Indices of the agents within communication range of agent ``i``. """
        dist = np.linalg.norm(self.positions - self.positions[i], axis=1)
        return np.where((dist < self.comm_range) & (dist > 0))[0]

    def step(self):
        """ Steps every agent sequentially, then moves them. """
        for i, agent in enumerate(self.agents):
            messages = [self.agents[j].message for j in self.neighbors(i)]
            agent.step(self.observed_object(i), messages)
        self._move()
        self.t += 1

    def _move(self):
        half = self.arena_size / 2
        turn = self.rng.random_sample(len(self.agents)) < self.turn_prob
        self.headings[turn] = self.rng.uniform(0, 2 * np.pi, size=turn.sum())
        step = self.speed * np.c_[np.cos(self.headings), np.sin(self.headings)]
        self.positions += step
        out = np.any(np.abs(self.positions) > half, axis=1)
        self.positions = np.clip(self.positions, -half, half)
        self.headings[out] = self.rng.uniform(0, 2 * np.pi, size=out.sum())
