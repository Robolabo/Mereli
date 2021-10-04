import sys
import os
sys.path.append('D:\subversion\SpikeSwarmSim')
import numpy as np
import matplotlib.pyplot as plt
from mereli.neural_networks import AdExModel 

dt = 1.
n_neurons = 1
ensembles_dict = None
lif = AdExModel(dt, n_neurons, ensembles_dict)

#!!!!! TODOOO
def phase_plane(self):
    self.reset()
    I_vals = []
    trajectory_v = []
    trajectory_u = []
    for t in range(1000):
        # I = 3 * (t > 100 and t < 1000) 
        I = int(t%100 == 0)
        self.step(I)
        trajectory_v.append(self.v)
        trajectory_u.append(self.u)
        I_vals.append(I)
    v_space = np.linspace(-100, 30, 100)
    u_space = np.linspace(-30, 10, 100)
    nullcline1 = lambda v: 0.04*v**2 + 5*v + 140
    nullcline2 = lambda v: v*self.b
    f, axes = plot.subplots(1,3)
    axes[2].plot(v_space, nullcline1(v_space), color='k')
    axes[2].plot(v_space, nullcline2(v_space), color='k')
    axes[2].set_ylim(-20,10)
    axes[0].plot(I_vals)
    axes[1].plot(trajectory_v)
    axes[2].plot(trajectory_v, trajectory_u, color='r')
    self.reset()
