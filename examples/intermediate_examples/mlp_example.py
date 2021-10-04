
import time
import numpy as np
import matplotlib.pyplot as plt
from mereli.globals import global_states
from mereli.neural_networks import MLP
from mereli.neural_networks.utils import plot_weights


ann = MLP()

ann.add_stimuli('I1', 1)
ann.add_ensemble('H1', 1, bias=1)
ann.add_ensemble('H2', 1, bias=1)
ann.add_ensemble('H3', 1, bias=1)
ann.add_ensemble('H4', 1, bias=1)

ann.set_motor('O')

ann.add_synapse('I1-O', 'I1', 'O', weight=1, p=1)
ann.add_synapse('I2-O', 'I2', 'O', weight=1, p=1)
ann.add_synapse('I3-O', 'I3', 'O', weight=1, p=1)

ann.build()
ann.reset()




# xx, yy = np.meshgrid(np.linspace(-1,1,1000), np.linspace(-1,1,1000))
# zz = []
# for x, y in zip(xx.ravel(), yy.ravel()):
#     inputs = np.array([np.sin(10*x), np.sin(10*y), np.linalg.norm([x,y])])
#     zz.append(ann.step(inputs))
# zz = np.stack(zz).reshape(xx.shape)
# import pdb; pdb.set_trace()