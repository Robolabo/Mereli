import numpy as np
import matplotlib.pyplot as plt

from mereli.globals import global_states
from mereli.neural_networks import NeuralNetwork
from mereli.neural_networks.utils import *

global_states.set_states(debug=True)

dt = 0.1

ann = NeuralNetwork(dt, neuron_model='rate_model', synapse_model='static_synapse')

ann.add_stimuli('I1', 1)
ann.add_ensemble('H1', 300, tau=np.random.randint(10, 500, 300)*dt, bias=-3)#np.random.randint(10, 500, 3)*dt)
# ann.add_ensemble('H2', 1, tau=100*dt, bias=-3)#np.random.randint(10, 500, 3)*dt)
# ann.add_ensemble('H3', 1, tau=100*dt, bias=-3)#np.random.randint(10, 500, 3)*dt)
# ann.add_ensemble('H4', 1, tau=200*dt, bias=-3)#np.random.randint(10, 500, 3)*dt)
# ann.add_ensemble('H5', 1, tau=200*dt, bias=-3)#np.random.randint(10, 500, 3)*dt)
# ann.add_ensemble('H6', 1, tau=200*dt, bias=-3)#np.random.randint(10, 500, 3)*dt)

ann.add_ensemble('O', 1, tau=10*dt)
ann.set_motor('O')

ann.add_synapse('I1-H1', 'I1', 'H1', weight='random', p=0.6)
ann.add_synapse('H1-H1', 'H1', 'H1', weight='random', p=0.3)
# ann.add_synapse('H1-H2', 'H1', 'H2', weight=5, p=1)
# ann.add_synapse('H2-H3', 'H2', 'H3', weight=7, p=1)
# ann.add_synapse('H3-H4', 'H3', 'H4', weight=10, p=1)
# ann.add_synapse('H4-H5', 'H4', 'H5', weight=4, p=1)
# ann.add_synapse('H5-H6', 'H5', 'H6', weight=4, p=1)
# ann.add_synapse('H6-H1', 'H6', 'H1', weight=4, p=1)
# ann.add_synapse('H3-H1', 'H3', 'H1', weight=4, p=1)

ann.add_synapse('H1-O', 'H1', 'O', weight=1, p=1)
ann.add_decoder('IdentityDecoding', 'O', 'A')
ann.build()
ann.reset()
ann.synapses.weights *= 10

stimuli = np.array([100 < t < 200 for t in range(1000)]).astype(int)
for stim in stimuli:
    ann.step({'I1' : stim})

outputs = np.stack(tuple(ann.monitor.get('outputs').values()))
currents = np.stack(tuple(ann.monitor.get('currents').values()))

import pdb; pdb.set_trace()