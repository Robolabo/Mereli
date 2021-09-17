import numpy as np
import matplotlib.pyplot as plt
from spike_swarm_sim.globals import global_states
from spike_swarm_sim.neural_networks import NeuralNetwork

dt = 0.1
ann = NeuralNetwork(dt, neuron_model='rate_model', synapse_model='static_synapse')


# ann.add_stimuli('Input_1', 1)
ann.add_ensemble('H1', 1, tau=1., bias=-2.75)
ann.add_ensemble('H2', 1, tau=1., bias=-1.75)
ann.set_motor('H1')
ann.set_motor('H2')

ann.add_synapse('H1-H2', 'H1', 'H2', weight=-1)
ann.add_synapse('H2-H1', 'H2', 'H1', weight=1)
ann.add_synapse('H1-H1', 'H1', 'H1', weight=4.5)
ann.add_synapse('H2-H2', 'H2', 'H2', weight=-4.5)

ann.add_decoder('IdentityDecoding', 'H1', 'A1')
ann.add_decoder('IdentityDecoding', 'H2', 'A2')
# ann.build()
ann.reset()

print('\nANN weight adjacency matrix: ', ann.weights)
print('Neurons Voltages: ', ann.voltages)
print('Neurons Time Constants (tau): ', ann.neurons.tau)
print('Neurons biases or offsets: ', ann.neurons.bias)
print('Neurons gains: ', ann.neurons.gain)
print()


outputs_1 = []
outputs_2 = []
for _ in range(500):
    output = ann.step({})
    out1 = output['A1']
    out2 = output['A2']
    outputs_1.append(out1)
    outputs_2.append(out2)

plt.plot(np.arange(len(outputs_1)) * dt, outputs_1, color='b')
plt.plot(np.arange(len(outputs_1)) * dt, outputs_2, color='g')
plt.xlabel('Time (s)')
plt.ylabel('Measure')
plt.legend([ 'Output/Action', 'Voltage/Neuron State'])
plt.show()
