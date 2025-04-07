import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mereli.globals import global_states
from mereli.neural_networks import NeuralNetwork


dt = 0.1
ann = NeuralNetwork(dt, neuron_model='rate_model', synapse_model='static_synapse')


ann.add_stimuli('Input_1', 1)
ann.add_stimuli('Input_2', 1)
ann.add_ensemble('Hidden_1', 1, tau=5*dt)
ann.add_ensemble('Hidden_2', 1, tau=5*dt)
ann.add_ensemble('Motor', 1, tau=5*dt)
ann.set_motor('Motor')

ann.add_synapse('I1-H1', 'Input_1', 'Hidden_1', weight=-0.25)
ann.add_synapse('I2-H1', 'Input_2', 'Hidden_1', weight=0.5)
ann.add_synapse('I1-H2', 'Input_1', 'Hidden_2', weight=0.5)
ann.add_synapse('I2-H2', 'Input_2', 'Hidden_2', weight=-0.25)

ann.add_synapse('H1-H2', 'Hidden_1', 'Hidden_2', weight=0.5)
ann.add_synapse('H2-H1', 'Hidden_2', 'Hidden_1', weight=-0.5)
ann.add_synapse('H1-H1', 'Hidden_1', 'Hidden_1', weight=-0.1)
ann.add_synapse('H2-H2', 'Hidden_2', 'Hidden_2', weight=0.1)

ann.add_synapse('H1-M', 'Hidden_1', 'Motor', weight=0.7)
ann.add_synapse('H2-M', 'Hidden_2', 'Motor', weight=0.7)

ann.add_decoder('IdentityDecoding', 'Motor', 'Action1')
ann.build()
ann.reset()

print('\nANN weight adjacency matrix: ', ann.weights)
print('Neurons Voltages: ', ann.voltages)
print('Neurons Time Constants (tau): ', ann.neurons.tau)
print('Neurons biases or offsets: ', ann.neurons.bias)
print('Neurons gains: ', ann.neurons.gain)
print()

# stimuli = np.array([t % 50 < 25  for t in range(500)])
input1 = np.array([np.sin(2*np.pi*0.2*t*ann.dt) for t in range(1000)])
input2 = np.array([np.sin(2*np.pi*0.05*t*ann.dt) for t in range(1000)])
outputs = []
voltages = {'h1' : [], 'h2' : [], 'm' : []}
for in1, in2 in zip(input1, input2):
    output = ann.step({'Input_1' : in1, 'Input_2' : in2})['Action1']
    outputs.append(output)
    voltages['m'].append(ann.voltage_of('Motor', 0))
    voltages['h1'].append(ann.voltage_of('Hidden_1', 0))
    voltages['h2'].append(ann.voltage_of('Hidden_2', 0))
T = len(outputs)
plt.figure(figsize=(14,7))

# plt.plot(np.arange(T) * 0.01, input1, color='k', lw=4)
# plt.plot(np.arange(T) * 0.01, input2, color='r', lw=4)

# plt.plot(np.arange(T) * dt, voltages['m'], color='k', lw=2.5)
plt.plot(np.arange(T) * 0.01, voltages['h1'], color='k', lw=4)
plt.plot(np.arange(T) * 0.01, voltages['h2'], color='r', lw=4)
plt.plot(np.arange(T) * 0.01, voltages['m'], color='b', lw=4)
plt.ylabel('Voltage', fontsize=30)
# plt.ylabel('Input', fontsize=30)
plt.xlabel('Time', fontsize=30)

plt.yticks(fontsize=30)
plt.xticks(fontsize=30)
# plt.legend(['Input 1', 'Input 2'])
# plt.legend(['v1', 'v2'], fontsize=20)

plt.xlim([0, T*0.01])
plt.ylim([-1.2, 1.2])
plt.yticks(fontsize=30)
plt.xticks(fontsize=30)
plt.subplots_adjust(left=0.2, right=0.95, top=0.98, bottom=0.2)
plt.legend(['Hidden 1', 'Hidden 2', 'Output'], fontsize=25, loc='upper right',  framealpha=1) 
# plt.legend(['Input 1', 'Input 2'], fontsize=25, loc='upper right',  framealpha=1) 

# plt.title('Input Layer', fontsize=30)
# plt.title('Neuron Actgcc', fontsize=30)
plt.savefig('/Users/rsendra/mereli_scripts/example.png', dpi=800)
plt.show()

# axs[1].plot(np.arange(T) * dt, voltages['h1'], color='k')
# axs[1].plot(np.arange(T) * dt, voltages['h2'], color='r')
# axs[1].set_ylabel('Voltage')
# axs[1].set_xlabel('Time')
# axs[1].set_title('Hidden Layer')
# axs[1].legend(['v1', 'v2'], loc="upper right")


# axs[2].plot(np.arange(T) * dt, voltages['m'], color='k')
# axs[2].set_ylabel('Voltage')
# axs[2].set_xlabel('Time')
# axs[2].set_title('Output Layer')

# plt.xlabel('Time (s)')
# plt.ylabel('Measure')
# plt.legend(['Stimuli', 'Output/Action', 'Voltage/Neuron State'])
# plt.show()



