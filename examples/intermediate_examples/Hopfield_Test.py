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
N = 50

for i in range(N):
    ann.add_ensemble(str(i), 1, tau=2*dt, bias=0, gain=100, activation='sigmoid')


ann.set_motor('0')


# patterns = np.array([
#     [[1,1,1,1,1],
#     [ 0,0,1,0,0],
#     [ 0,0,1,0,0],
#     [ 1,1,1,1,1]],
#     [[1,1,1,1,1],
#     [ 1,0,0,0,1],
#     [ 1,1,1,1,1],
#     [ 1,0,0,0,1]],
#     # [[1,1,1,1,1],
#     # [ 1,0,0,0,0],
#     # [ 1,0,0,0,0],
#     # [ 1,1,1,1,1]],
# ])
np.random.seed(2)
patterns = np.random.choice(2, size=[2,N])
np.random.seed()
# patterns = np.random.random(N).reshape(-1, N)
for i in range(N):
    for j in range(N):
        if i != j:
            # ww = np.sum([pat.flatten()[i]*pat.flatten()[j] for pat in patterns])
            ww = np.sum([(2*pat.flatten()[i]-1)*(2*pat.flatten()[j]-1) for pat in patterns])
            ann.add_synapse('{}-{}'.format(i,j), str(i), str(j), weight=ww)
 

# w12 = np.sum([(2*pat[0]-1)*(2*pat[1]-1) for pat in patterns])
# w13 = np.sum([(2*pat[0]-1)*(2*pat[2]-1) for pat in patterns])
# w23 = np.sum([(2*pat[1]-1)*(2*pat[2]-1) for pat in patterns])






ann.add_decoder('IdentityDecoding', '1', 'A1')

ann.build()
ann.reset()




eval_pat = patterns[0].copy()

noisy_pattern = eval_pat.copy()
noisy_pattern[N//2:] = 1
ann.neurons.voltages = noisy_pattern.astype(float) * 20 - 10 # np.random.randn(N) *.2 #

for t in range(1000):
    output = ann.step({})

reconstr = ann.spikes.round().reshape(eval_pat.shape)
error = np.sum(np.abs(reconstr - eval_pat))
print('Error: ', error)
import pdb; pdb.set_trace()