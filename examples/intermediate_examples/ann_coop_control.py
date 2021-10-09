import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from mereli.globals import global_states
from mereli.neural_networks import NeuralNetwork
from mereli.neural_networks.utils import plot_state_plane
from mereli.utils import activations

# global_states.set_states(debug=True)

dt = 0.1
ann = NeuralNetwork(dt, neuron_model='rate_model', synapse_model='static_synapse')

ann.add_stimuli('I', 1)
ann.add_ensemble('1', 1, tau=50*dt, bias=0, activation='linear')
ann.add_ensemble('2', 1, tau=50*dt, bias=0, activation='linear')
ann.add_ensemble('3', 1, tau=50*dt, bias=0, activation='linear')
ann.add_ensemble('4', 1, tau=50*dt, bias=0, activation='linear')
ann.add_ensemble('5', 1, tau=50*dt, bias=0, activation='linear')
ann.add_ensemble('6', 1, tau=50*dt, bias=0, activation='linear')
ann.set_motor('5')

# ann.add_synapse('1-2', '1', '2', weight=1/2)
# ann.add_synapse('1-5', '1', '5', weight=1)
# ann.add_synapse('2-3', '2', '3', weight=1)
# ann.add_synapse('3-4', '3', '4', weight=1/2)
# ann.add_synapse('4-1', '4', '1', weight=1)
# ann.add_synapse('4-2', '4', '2', weight=1/2)
# ann.add_synapse('5-4', '5', '4', weight=1/2)

# TREE
ann.add_synapse('I-1', 'I', '1', weight=1, learning_rule='simple_hebb')
ann.add_synapse('I-2', 'I', '2', weight=1, learning_rule='stdp')
ann.add_synapse('1-3', '1', '3', weight=1)
ann.add_synapse('1-4', '1', '4', weight=1)
ann.add_synapse('2-5', '2', '5', weight=1)
ann.add_synapse('2-6', '2', '6', weight=1)



# PERIODIC 6
# ann.add_synapse('1-2', '1', '2', weight=1)
# ann.add_synapse('2-3', '2', '3', weight=1)
# ann.add_synapse('3-4', '3', '4', weight=1)
# ann.add_synapse('4-5', '4', '5', weight=1)
# ann.add_synapse('5-6', '5', '6', weight=1)
# ann.add_synapse('6-1', '6', '1', weight=1)

ann.add_decoder('IdentityDecoding', '5', 'A1')


ann.build()
ann.reset()


ann.laplacian

print('\nANN weight adjacency matrix: \n', ann.weights)
print('Neurons Voltages: ', ann.voltages)
print('Neurons Time Constants (tau): ', ann.neurons.tau)
print('Neurons biases or offsets: ', ann.neurons.bias)
print('Neurons gains: ', ann.neurons.gain)
print()


ann.neurons.voltages = np.random.uniform(-3,3, size=ann.num_neurons)
print('V_0(0)=', ann.neurons.voltages[0])
outputs = []
for t in range(1000):
    if t> 500:
        ii =  np.array([1])
    else: 
        ii = np.array([2])
    output = ann.step({'I' : ii})['A1']
    outputs.append(ann.voltages)
outputs = np.vstack(outputs)
plt.plot(outputs)
plt.legend([f'Node {i+1}' for i in range(6)])
plt.show()
import pdb; pdb.set_trace()