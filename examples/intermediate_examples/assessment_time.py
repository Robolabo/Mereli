import time
import numpy as np
import matplotlib.pyplot as plt
from mereli.globals import global_states
from mereli.neural_networks import NeuralNetwork

dt = 0.1
times_build = []
times_sim = []
neurons = [10, 100, 500, 1000, 5000]
for n in neurons:
    t0 = time.time()
    ann = NeuralNetwork(dt, neuron_model='rate_model', synapse_model='static_synapse')
    
    ann.add_stimuli('I1', 10)
    ann.add_ensemble('H1', n)
    ann.add_ensemble('H2', 10)
    ann.set_motor('H2')

    ann.add_synapse('I1-H1', 'I1', 'H1', weight=1, p=0.5)
    ann.add_synapse('H1-H2', 'H1', 'H2', weight=0.3, p=0.4)
    ann.add_synapse('H1-H1', 'H1', 'H1', weight=-0.3, p=0.4)

    ann.add_decoder('IdentityDecoding', 'H2', 'A')

    ann.build()
    ann.reset()

    time_to_build = time.time() - t0
    t0 = time.time()
    for _ in range(1000):
        stim = np.random.random(10)
        ann.step({'I1' : stim})
    time_simulation = time.time() - t0
    times_build.append(time_to_build)
    times_sim.append(time_simulation)
    print('TOTAL TIME ELAPSED: ', np.round(time_to_build, 4), np.round(time_simulation, 4))

plt.plot(neurons, times_build)
plt.plot(neurons, times_sim)
plt.legend(['Time to build ANN.', 'Time to simulate (1000 steps).'])
plt.show()