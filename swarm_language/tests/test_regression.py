""" Regression test: the language-game dynamics must stay bit-identical to the implementation
used for the published experiments.

The reference digests in ``data/reference_digests.json`` were produced by the original
``build_lexicon_B`` controller (Mereli commit cc09e39, branch Thesis_RS) driven through the
same :class:`ToyWorld` scenario. Every 100 steps the test hashes the full state of all agents
(words, meanings, MAN, traces, association weights, prototypes, messages) and the state of
the global ``np.random`` generator, so any change in the dynamics or in the consumption of
random numbers is detected.

Run with ``python -m unittest swarm_language.tests.test_regression`` from the repo root.
"""
import hashlib
import json
import os
import unittest
import numpy as np
import networkx as nx
from swarm_language import ToyWorld

REFERENCE_FILE = os.path.join(os.path.dirname(__file__), 'data', 'reference_digests.json')

SCENARIO = {
    'np_seed': 2026,
    'world_seed': 115879,
    'n_agents': 10,
    'arena_size': 4.0,
    'steps': 1500,
    'check_every': 100,
    # Referents of Experiment 2 of the paper, 4 copies each.
    'objects': [{'color': c, 'shape': s, 'size': 0.2} for c, s in
                [('green', 'cube'), ('red', 'cube'), ('green', 'ball'), ('red', 'ball'),
                 ('red', 'pyramid')] for _ in range(4)],
}


def _canonical(x):
    """ Platform-independent representation (numpy scalar types and integer widths vary). """
    if isinstance(x, np.ndarray):
        return ['array', list(x.shape), hashlib.sha256(np.ascontiguousarray(x, dtype=np.float64).tobytes()).hexdigest()]
    if isinstance(x, (list, tuple)):
        return [_canonical(v) for v in x]
    if isinstance(x, dict):
        return [[_canonical(k), _canonical(v)] for k, v in x.items()]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x).hex()
    return x


def agent_state(agent):
    lex, pd = agent.lexicon, agent.pattern_detector
    msg = agent.message
    return _canonical([
        lex.words, lex.meanings, lex.active, lex.traces, lex.assoc_weights, lex.man,
        lex.prototype_to_meaning, lex.word_to_meaning,
        [[n, d] for n, d in lex.nx_tree.nodes(data=True)], [list(e) for e in lex.nx_tree.edges],
        lex.elementary_words, lex.composed_words, lex.assoc_list,
        pd.pattern_mat, pd.names,
        None if msg is None else [msg['pattern'], msg['word'], msg['parents']],
    ])


def rng_state():
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return _canonical([name, keys, pos, has_gauss, cached])


def digest(world):
    payload = json.dumps([[agent_state(a) for a in world.agents], rng_state()], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def run_scenario(scenario=SCENARIO):
    """ Returns ``{step: digest}`` for the regression scenario. """
    np.random.seed(scenario['np_seed'])
    world = ToyWorld(scenario['n_agents'], scenario['objects'], arena_size=scenario['arena_size'],
                     seed=scenario['world_seed'])
    digests = {}
    for t in range(1, scenario['steps'] + 1):
        world.step()
        if t % scenario['check_every'] == 0:
            digests[str(t)] = digest(world)
    return digests, world


class TestRegression(unittest.TestCase):
    def test_dynamics_unchanged(self):
        with open(REFERENCE_FILE) as f:
            reference = json.load(f)
        digests, world = run_scenario()
        # The scenario must exercise compositionality, otherwise the test is weak.
        self.assertTrue(all(len(a.lexicon.composed_words) > 0 for a in world.agents))
        for step, ref in reference['digests'].items():
            self.assertEqual(digests[step], ref, 'Dynamics diverged before step {}'.format(step))


if __name__ == '__main__':
    unittest.main()
