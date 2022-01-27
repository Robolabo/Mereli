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


ann.add_ensemble('1', 1, tau=30*dt, bias=0, gain=1, activation='sigmoid')
ann.add_ensemble('2', 1, tau=30*dt, bias=0, gain=1, activation='sigmoid')
ann.add_ensemble('3', 1, tau=30*dt, bias=0, gain=1, activation='sigmoid')

ann.set_motor('1')


pattern = np.array([0.2, 0.2, 0.2])
pattern2 = np.array([0.8, 0.8, 0.8])


ann.add_synapse('1-2', '1', '2', weight=0.5*(pattern[0] * sigm_inv(pattern[1]) + pattern2[0] * sigm_inv(pattern2[1])))
ann.add_synapse('2-1', '2', '1', weight=0.5*(pattern[1] * sigm_inv(pattern[0]) + pattern2[1] * sigm_inv(pattern2[0])))
ann.add_synapse('1-3', '1', '3', weight=0.5*(pattern[0] * sigm_inv(pattern[2]) + pattern2[0] * sigm_inv(pattern2[2])))
ann.add_synapse('3-1', '3', '1', weight=0.5*(pattern[2] * sigm_inv(pattern[0]) + pattern2[2] * sigm_inv(pattern2[0])))
ann.add_synapse('2-3', '2', '3', weight=0.5*(pattern[1] * sigm_inv(pattern[2]) + pattern2[1] * sigm_inv(pattern2[2])))
ann.add_synapse('3-2', '3', '2', weight=0.5*(pattern[2] * sigm_inv(pattern[1]) + pattern2[2] * sigm_inv(pattern2[1])))

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
ann.neurons.voltages = [0,0,0]#np.random.uniform(-3, 3, size=ann.num_neurons)
print('V_0(0)=', ann.neurons.voltages[0])


for t in range(300):
    output = ann.step({})

print(output)
import pdb; pdb.set_trace()