import numpy as np
import matplotlib.pyplot as plt

from spike_swarm_sim.globals import global_states
from spike_swarm_sim.neural_networks import NeuralNetwork
from spike_swarm_sim.neural_networks.utils import *


dt = 0.1
ann = NeuralNetwork(dt, neuron_model='rate_model', synapse_model='static_synapse')

ann.add_stimuli('I1', 1)
ann.add_stimuli('I2', 1)
ann.add_ensemble('H1', 10, tau=dt*5)
ann.add_ensemble('O', 1, tau=dt*5)
ann.set_motor('O')


ann.add_learning_rule()
ann.add_synapse('I1-H1', 'I1', 'H1', weight='random', p=1)
ann.add_synapse('I2-H1', 'I2', 'H1', weight='random', p=1)
ann.add_synapse('H1-O',  'H1', 'O',  weight='random', p=1)
ann.add_synapse('H1-H1', 'H1', 'H1', weight='random', p=0.5)
ann.add_decoder('IdentityDecoding', 'O', 'A')


#! OJO ESTA LINEA ES TEMPORAL HASTA MEJORAR CODIGO LEARNING RULES
ann.build()
ann.reset()
ann.learning_rule.A += 1
ann.learning_rule.modulated = True
#!


def random_function(duration):
    n = np.random.randint(3)
    n = 0
    if n == 0:
        tt = np.linspace(0,1,duration)
        freq = np.random.uniform(1, 15)
        phase = np.random.uniform(0, np.pi)
        return .5 + np.sin(2 * np.pi * freq * tt + phase) / 2
    elif n == 1: # Gaussian 
        tt = np.linspace(0,1,duration)
        center =  np.random.uniform(0,1)
        return np.exp(-100 * (tt - center) ** 2)
    elif n == 2: # Single pulse
        width = 100
        start = np.random.uniform(1, duration-width)
        end = start + width
        return np.array([ start < t < end  for t in range(duration)]).astype(int)
    elif n == 3: # Pulse train
        width = 100
        duty_cycle = 0.5
        phase = np.random.uniform(0, width)
        return np.array([(t + phase) % width < int(duty_cycle * width)  for t in range(duration)]).astype(int)
    else:
        tt = np.linspace(0, 1, duration)
        return np.sin(2 * np.pi * 10 * tt)      


ann.build()#!
D = 50
done = False
trial = 0
while not done:
    #! TEMPORAL
    ann.reset()#! OJO BUILD
    ann.learning_rule.A += 1# np.random.randn(*ann.learning_rule.A.shape)
    ann.learning_rule.modulated = True
    ann.learning_rule.learning_rate = 1e-2
    #!

    stimuli1 = random_function(500)
    stimuli2 = random_function(500)
    reward = 0
    mse = []
    ws = []
    for t, (stim1, stim2) in enumerate(zip(stimuli1, stimuli2)):
        y_true = (stim1 + stim2) / 2 
        y_hat = np.array(ann.step({'I1' : stim1, 'I2' : stim2}, reward=reward)['A'])
        squared_error = np.abs(y_true - y_hat) ** 2
        # reward = 1 - squared_error
        reward = y_true - y_hat
        reward = squared_error
        # if squared_error > 0.4:
        #     reward = - squared_error

        # if reward > 1 or reward < -1:
        #     print(reward)
        mse.append(squared_error)
        ws.append(ann.weights.flatten())
    mse = np.mean(mse)
    # print('TRIAL=',trial)
    if trial % 5 == 0:
        print('Trial=',trial,' MSE=',mse,' W_sum=', ann.weights.sum())
    # if ann.weights.sum()>1000:import pdb; pdb.set_trace()
    trial += 1
    done = mse < 1e-3 or trial >= 1000
    ws = np.stack(ws)
    # import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    # print('W_end ',ann.weights.sum())

plt.plot(0.5*(stimuli1 + stimuli2))
plt.plot(np.array([ann.step({'I1' : stim1, 'I2' : stim2}, reward=reward)['A'] for stim1, stim2 in zip(stimuli1,stimuli2)] ))
plt.show()
import pdb; pdb.set_trace()
# outputs = ann.monitor.get('outputs', 'O_0')
# plt.plot(outputs)
# plt.plot(stimuli)
# plt.show()


