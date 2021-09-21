import time
import numpy as np
import matplotlib.pyplot as plt

from spike_swarm_sim.globals import global_states
from spike_swarm_sim.neural_networks import NeuralNetwork
from spike_swarm_sim.neural_networks.utils import *
from spike_swarm_sim.utils import sigmoid
from spike_swarm_sim.neural_networks.utils import plot_weights 

global_states.set_states(debug=True, info=True)



def iterative_LS(x, y, beta_prev, n, alpha=0.5):
    # x = np.r_[x,1]
    delta_beta = (1/n)**(alpha) * x * (y - x.dot(beta_prev))
    return beta_prev + delta_beta

def random_function_generator(duration):
    n = np.random.randint(4)
    n = 1
    if n == 0:
        tt = np.linspace(0,1,duration)
        freq = np.random.uniform(1, 15)
        phase = np.random.uniform(0, np.pi)
        return .5 + np.sin(2 * np.pi * freq * tt + phase) / 2
    elif n == 1: # Gaussian 
        tt = np.linspace(0,1,duration)
        center =  np.random.uniform(0,1)
        return np.exp(-200 * (tt - center) ** 2)
    elif n == 2: # Single pulse
        width = 70
        start = np.random.uniform(1, duration-width)
        end = start + width
        return np.array([ start < t < end  for t in range(duration)]).astype(int)
    elif n == 3: # Pulse train
        width = 70
        duty_cycle = 0.5
        phase = np.random.uniform(0, width)
        return np.array([(t + phase) % width < int(duty_cycle * width)  for t in range(duration)]).astype(int)
    else:
        tt = np.linspace(0, 1, duration)
        return np.sin(2 * np.pi * 10 * tt) 



D = 100
np.random.seed(1223)
dt = 0.1
ann = NeuralNetwork(dt, neuron_model='rate_model', synapse_model='static_synapse')

ann.add_stimuli('I1', 1)
# ann.add_stimuli('I2', 1)
ann.add_ensemble('H1', 50, tau=np.array([5] * 10 + [10] * 20 + [50] * 10 + [100] * 10) * dt, bias=-3)#np.random.rayleigh(60, 50)#
ann.add_ensemble('H2', 50, tau=np.array([10] * 10 + [50] * 10 + [100] * 20 + [200] * 10) * dt, bias=-3)
ann.set_motor('H1')


ann.add_synapse('I1-H1', 'I1', 'H1', weight='random', conn_prob=0.8)
ann.add_synapse('I1-H2', 'I1', 'H2', weight='random', conn_prob=0.8)
# ann.add_synapse('I2-H1', 'I2', 'H1', weight='random', p=0.8)
ann.add_synapse('H1-H1', 'H1', 'H1', weight='random', conn_prob=0.4)
ann.add_synapse('H1-H2', 'H1', 'H2', weight='random', conn_prob=0.6)
ann.add_synapse('H2-H2', 'H2', 'H2', weight='random', conn_prob=0.4)
ann.add_synapse('H2-H1', 'H2', 'H1', weight='random', conn_prob=0.6)
ann.add_decoder('IdentityDecoding', 'H1', 'A')


#! OJO ESTA LINEA ES TEMPORAL HASTA MEJORAR CODIGO LEARNING RULES
ann.build()
ann.reset()
#!
ann.synapses.weights *= 10


done = False
trial = 0
beta = np.zeros(50)
n = 1
while not done:
    ann.reset()#! OJO BUILD

    stimuli1 = random_function_generator(500)
    stimuli2 = random_function_generator(500)
    mse = []
    step_time = []
    for t, (stim1, stim2) in enumerate(zip(stimuli1[D:], stimuli2),start=D):
        # y_true = (stim1 + stim2) / 2 
        y_true = stimuli1[t-D]
        t0 = time.time()
        activities = np.array(ann.step({'I1' : stim1})['A'])
        step_time.append(time.time() - t0)
        # activities = np.array(ann.step({'I1' : stim1, 'I2' : stim2})['A'])
        y_hat = activities.dot(beta)
        # beta = iterative_LS(activities, np.log((y_true/(1-y_true+1e-3))), beta, n, alpha=0.5)
        beta = iterative_LS(activities, y_true, beta, n, alpha=0.1)
        n += 1
        squared_error = np.abs(y_true - y_hat) ** 2
        # if squared_error == np.nan: import pdb; pdb.set_trace()
        
        mse.append(squared_error)
    # import pdb; pdb.set_trace()
    mse = np.mean(mse)
    # print('TRIAL=',trial)
    if trial % 5 == 0:
        print('Trial=',trial,' MSE=',mse, ' Step time: ', np.mean(step_time))
        # print('BETA=', beta)
    # if ann.weights.sum()>1000:import pdb; pdb.set_trace()
    trial += 1
    done = trial >= 1500 #or mse < 1e-3
    # import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    # print('W_end ',ann.weights.sum())

def evaluate():
    stimuli1 = random_function_generator(500)
    activs = np.array([ann.step({'I1' : stim1})['A'] for stim1 in stimuli1] )
    plt.plot(activs.dot(beta))
    plt.plot(stimuli1)
    plt.show()
evaluate()
outputs = np.stack(tuple(ann.monitor.get('outputs').values()))
currents = np.stack(tuple(ann.monitor.get('currents').values()))
import pdb; pdb.set_trace()
# outputs = ann.monitor.get('outputs', 'O_0')
# plt.plot(outputs)
# plt.plot(stimuli)
# plt.show()


