import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from mereli.globals import global_states
from mereli.neural_networks import NeuralNetwork
from mereli.neural_networks.utils import plot_state_plane
from mereli.utils import activations
from mereli.neural_networks.update_rules import BaseLearningRule
from mereli.register import learning_rule_registry

@learning_rule_registry(name='vin_square')
class VinSquareRule(BaseLearningRule):
    def __init__(self):
        super(VinSquareRule, self).__init__()
        self.modulated = False #!
        self.learning_rate = 1e-1

    def step(self, weights, activities, stimuli, reward=None):
        all_vars = np.r_[stimuli, activities]
        # delta_w = 0.9 * np.outer( all_vars, np.ones_like(stimuli)) - weights\
        delta_w = -0.9  *activities[0] ** 2 - weights[1][1]
        # delta_w = activities[0] ** 2 - weights[1][1]
        return self.learning_rate * self.mask * delta_w



@learning_rule_registry(name='sin_rule')
class SinRule(BaseLearningRule):
    def __init__(self):
        super(SinRule, self).__init__()
        self.modulated = False #!
        self.learning_rate = 1e-2

    def step(self, weights, activities, stimuli, reward=None):
        all_vars = np.r_[stimuli, activities]
        # delta_w = 0.9 * np.outer( all_vars, np.ones_like(stimuli)) - weights\
        delta_w = 0.9 * np.sin(stimuli[1]) - weights[1][0]
        # import pdb; pdb.set_trace()
        return self.learning_rate * self.mask * delta_w

@learning_rule_registry(name='cos_rule')
class CosRule(BaseLearningRule):
    def __init__(self):
        super(CosRule, self).__init__()
        self.modulated = False #!
        self.learning_rate = 1e-2

    def step(self, weights, activities, stimuli, reward=None):
        all_vars = np.r_[stimuli, activities]
        # delta_w = 0.9 * np.outer( all_vars, np.ones_like(stimuli)) - weights\
        delta_w = 0.9 * np.cos(stimuli[1]) - weights[0][0]
        return self.learning_rate * self.mask * delta_w

# global_states.set_states(debug=True)

dt = 0.1
ann = NeuralNetwork(dt, neuron_model='rate_model', synapse_model='static_synapse')

ann.add_stimuli('I1', 1)
ann.add_stimuli('I2', 1)
ann.add_ensemble('1', 1, tau=30*dt, bias=0, gain=1, activation='linear')
ann.add_ensemble('2', 1, tau=30*dt, bias=0, gain=1, activation='linear')

ann.set_motor('1')



# PARABOLA
# ann.add_synapse('I1-1', 'I1', '1', weight=0.9)
# ann.add_synapse('I1-2', 'I1', '2', weight=0)
# ann.add_synapse('1-2', '1', '2', weight=1, learning_rule='vin_square')
# ann.add_synapse('2-1', '2', '1', weight=0)
# ann.add_synapse('2-2', '2', '2', weight=0.1)
# ann.add_synapse('1-1', '1', '1', weight=0.1)

# ann.add_synapse('I1-1', 'I1', '1', weight=0.9)
# ann.add_synapse('I1-2', 'I1', '2', weight=0)
# ann.add_synapse('1-2', '1', '2', weight=1, learning_rule='vin_square')
# ann.add_synapse('2-1', '2', '1', weight=0)
# ann.add_synapse('2-2', '2', '2', weight=0)
# ann.add_synapse('1-1', '1', '1', weight=0)

# SPIRAL
ann.add_synapse('I1-1', 'I1', '1', weight=1, learning_rule='cos_rule')
ann.add_synapse('I2-1', 'I2', '1', weight=0)
ann.add_synapse('I1-2', 'I1', '2', weight=1, learning_rule='sin_rule')
ann.add_synapse('I2-2', 'I2', '2', weight=0)

ann.add_synapse('1-2', '1', '2', weight=0)
ann.add_synapse('2-1', '2', '1', weight=0)
ann.add_synapse('2-2', '2', '2', weight=0.1)
ann.add_synapse('1-1', '1', '1', weight=0.1)

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
ann.neurons.voltages = np.random.uniform(-3, 3, size=ann.num_neurons)
print('V_0(0)=', ann.neurons.voltages[0])


n_points = 50
cmap=plt.get_cmap("Blues")
attractor = []
# for i, phi in enumerate(np.linspace(0, 1, num=n_points)):

for i, (rho, phi) in enumerate(zip(np.linspace(0, 10, num=100), np.linspace(0, 2*2*np.pi, num=100))):
    ann.reset()
    trajectory = [ann.voltages]
    for t in range(100):
        output = ann.step({'I1' : np.array([rho]), 'I2' : np.array([phi])})
        # output = ann.step({'I1' : np.array([phi])})
        trajectory.append(ann.voltages)
    trajectory = np.stack(trajectory)
    plt.plot(trajectory[:,0 ], trajectory[:,1], color=cmap(i/n_points))
    attractor.append(ann.voltages)

# plt.colorbar()    

attractor = np.stack(attractor)
plt.scatter(attractor[:,0], attractor[:,1], color='r')
plt.plot(attractor[:,0], attractor[:,1], color='r')
plt.show()
import pdb; pdb.set_trace()