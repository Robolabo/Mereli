""" Runs a swarm of language agents in the minimal :class:`ToyWorld` and prints the
evolution of the swarm metrics. Only numpy and networkx are required.

Example (from the folder that contains ``swarm_language``)::

    python -m swarm_language.demo --agents 20 --steps 3000 --seed 0

The referents are those of Experiment 2 of the paper. Note that this toy world is not the
e-puck/pybullet setup of the paper, so convergence times differ.
"""
import argparse
import numpy as np
from .toy_world import ToyWorld
from .metrics import swarm_metrics, report_line, REPORT_HEADER

EXPERIMENT_2_OBJECTS = [{'color': c, 'shape': s, 'size': 0.2} for c, s in
                        [('green', 'cube'), ('red', 'cube'), ('green', 'ball'), ('red', 'ball'),
                         ('red', 'pyramid')] for _ in range(4)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--agents', type=int, default=20)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--arena', type=float, default=6.0, help='side of the square arena (m)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--every', type=int, default=250, help='print period (steps)')
    args = parser.parse_args()

    np.random.seed(args.seed)
    world = ToyWorld(args.agents, EXPERIMENT_2_OBJECTS, arena_size=args.arena, seed=args.seed)
    print(REPORT_HEADER)
    for t in range(1, args.steps + 1):
        world.step()
        if t % args.every == 0:
            print(report_line(t, swarm_metrics([agent.lexicon for agent in world.agents])))


if __name__ == '__main__':
    main()
