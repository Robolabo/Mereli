import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from mereli.globals import global_states
from mereli.neural_networks import NeuralNetwork
from mereli.neural_networks.utils import plot_state_plane
from mereli.utils import activations

# global_states.set_states(debug=True)

def sigm_inv(x):
    return np.log(x/(1-x))

dt = 0.1
ann = NeuralNetwork(dt, neuron_model='rate_model', synapse_model='static_synapse')


ann.add_ensemble('1', 1, tau=5*dt, bias=0, gain=100, activation='sigmoid')
ann.add_ensemble('2', 1, tau=5*dt, bias=0, gain=100, activation='sigmoid')
ann.add_ensemble('3', 1, tau=5*dt, bias=0, gain=100, activation='sigmoid')

ann.set_motor('1')

patterns = np.array([
    [0, 0., 1.]
])


w12 = np.sum([pat[0]*pat[1] for pat in patterns])
w13 = np.sum([pat[0]*pat[2] for pat in patterns])
w23 = np.sum([pat[1]*pat[2] for pat in patterns])

w12 = np.sum([(2*pat[0]-1)*(2*pat[1]-1) for pat in patterns])
w13 = np.sum([(2*pat[0]-1)*(2*pat[2]-1) for pat in patterns])
w23 = np.sum([(2*pat[1]-1)*(2*pat[2]-1) for pat in patterns])

ann.add_synapse('1-2', '1', '2', weight=w12)
ann.add_synapse('2-1', '2', '1', weight=w12)
ann.add_synapse('1-3', '1', '3', weight=w13)
ann.add_synapse('3-1', '3', '1', weight=w13)
ann.add_synapse('2-3', '2', '3', weight=w23)
ann.add_synapse('3-2', '3', '2', weight=w23)

# No self connections
ann.add_synapse('1-1', '1', '1', weight=0)
ann.add_synapse('2-2', '2', '2', weight=0)
ann.add_synapse('3-3', '3', '3', weight=0)



ann.add_decoder('IdentityDecoding', '1', 'A1')

ann.build()
ann.reset()


ann.laplacian

print('\nANN weight adjacency matrix: \n', ann.weights)
print('Neurons Voltages: ', ann.voltages)
print('Neurons Time Constants (tau): ', ann.neurons.tau)
print('Neurons biases or offsets: ', ann.neurons.bias)
print('Neurons gains: ', ann.neurons.gain)
print()


w11 = []
ann.neurons.voltages = np.random.randn(3)*0.2

for t in range(100):
    output = ann.step({})

print(output)
import pdb; pdb.set_trace()