
import sys
import os
sys.path.append('D:\subversion\SpikeSwarmSim')
import numpy as np
import matplotlib.pyplot as plt
from mereli.neural_networks import LIFModel

dt = 1.
n_neurons = 1
ensembles_dict = None
lif = LIFModel(dt, n_neurons, ensembles_dict)

frequencies = []
currents = []
for stim in np.arange(200):
    lif.reset()
    freq = np.sum([lif.step(stim)[1] for _ in range(1000)])
    print(stim, freq)
    frequencies.append(freq)
    currents.append(stim)
plt.plot(currents, frequencies)
plt.show()