import numpy as np
import matplotlib.pyplot as plt
from spike_swarm_sim.globals import global_states
from spike_swarm_sim.neural_networks import NeuralNetwork

dt = 0.1
ann = NeuralNetwork(dt, neuron_model='rate_model', synapse_model='static_synapse')


ann.add_stimuli('Input_1', 1)
# ann.add_ensemble('Hidden', 3, tau=5*dt)
ann.add_ensemble('Motor', 1, tau=5*dt)
ann.set_motor('Motor')

# ann.add_synapse('I-H', 'Input_1', 'Hidden', weight=0.5)
ann.add_synapse('I-M', 'Input_1', 'Motor', weight=1)

ann.add_decoder('IdentityDecoding', 'Motor', 'Action1')
# ann.build()
ann.reset()

print('\nANN weight adjacency matrix: ', ann.weights)
print('Neurons Voltages: ', ann.voltages)
print('Neurons Time Constants (tau): ', ann.neurons.tau)
print('Neurons biases or offsets: ', ann.neurons.bias)
print('Neurons gains: ', ann.neurons.gain)
print()


stimuli = np.array([t % 50 < 25  for t in range(500)])
outputs = []
voltages = []
for stim in stimuli:
    output = ann.step({'Input_1' : stim})['Action1']
    outputs.append(output)
    voltages.append(ann.voltage_of('Motor', 0))

plt.plot(np.arange(len(outputs)) * dt, stimuli, color='b')
plt.plot(np.arange(len(outputs)) * dt, outputs, color='g')
plt.plot(np.arange(len(outputs)) * dt, voltages, color='r')
plt.xlabel('Time (s)')
plt.ylabel('Measure')
plt.legend(['Stimuli', 'Output/Action', 'Voltage/Neuron State'])
plt.show()



