import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from mereli.globals import global_states
from mereli.neural_networks import NeuralNetwork
from mereli.neural_networks.utils import plot_state_plane

global_states.set_states(debug=True)

dt = 0.1
ann = NeuralNetwork(dt, neuron_model='rate_model', synapse_model='static_synapse')


ann.add_ensemble('H1', 1, tau=10*dt, bias=0)
ann.add_ensemble('H2', 1, tau=10*dt, bias=-5)
ann.set_motor('H1')
ann.set_motor('H2')

ann.add_synapse('H1-H2', 'H1', 'H2', weight=4)
ann.add_synapse('H2-H1', 'H2', 'H1', weight=-4)
ann.add_synapse('H1-H1', 'H1', 'H1', weight=5)
ann.add_synapse('H2-H2', 'H2', 'H2', weight=5)

ann.add_decoder('IdentityDecoding', 'H1', 'A1')
ann.add_decoder('IdentityDecoding', 'H2', 'A2')
# ann.build()
ann.reset()


print('\nANN weight adjacency matrix: \n', ann.weights)
print('Neurons Voltages: ', ann.voltages)
print('Neurons Time Constants (tau): ', ann.neurons.tau)
print('Neurons biases or offsets: ', ann.neurons.bias)
print('Neurons gains: ', ann.neurons.gain)
print()


outputs_1 = []
outputs_2 = []
for _ in range(1000):
    output = ann.step({})
    out1 = output['A1']
    out2 = output['A2']
    outputs_1.append(out1)
    outputs_2.append(out2)


#* Plot Neuron's outputs
plt.plot(np.arange(len(outputs_1)) * dt, outputs_1, color='b')
plt.plot(np.arange(len(outputs_1)) * dt, outputs_2, color='g')
plt.xlabel('Time (s)')
plt.ylabel('Measure')
plt.legend([ 'Output/Action', 'Voltage/Neuron State'])
plt.show()

#* Plot State Space

# import pdb; pdb.set_trace()
plot_state_plane(ann, t_start=0, t_end=len(outputs_1), n_pc=2)
plt.xlabel('Output Neuron 1')
plt.ylabel('Output Neuron 2')
plt.title('Phase plane of the CTRNN.')
plt.show()