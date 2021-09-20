import numpy as np
import matplotlib.pyplot as plt

from spike_swarm_sim.globals import global_states
from spike_swarm_sim.neural_networks import NeuralNetwork
from spike_swarm_sim.neural_networks.utils import *

global_states.set_states(debug=True)

dt = 0.1

ann = NeuralNetwork(dt, neuron_model='rate_model', synapse_model='static_synapse')

ann.add_stimuli('I1', 10)
ann.add_ensemble('H1', 100)
ann.add_ensemble('H2', 50)
ann.add_ensemble('O', 10)
ann.set_motor('O')

ann.add_synapse('I1-H1', 'I1', 'H1', weight='random', p=0.5)
ann.add_synapse('H1-H2', 'H1', 'H2', weight='random', p=0.7)
ann.add_synapse('H2-O',  'H2', 'O',  weight='random', p=0.6)
ann.add_synapse('H1-H1', 'H1', 'H1', weight='random', p=0.4)
ann.add_synapse('H2-H2', 'H2', 'H2', weight='random', p=0.4)
ann.add_decoder('IdentityDecoding', 'O', 'A')
# ann.build()
ann.reset()

for _ in range(1000):
    stim = np.random.random(10)
    ann.step({'I1' : stim})

outputs = np.stack(tuple(ann.monitor.get('outputs').values()))
import pdb; pdb.set_trace()