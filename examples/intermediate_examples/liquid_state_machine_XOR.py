import numpy as np
import matplotlib.pyplot as plt

from mereli.globals import global_states
from mereli.neural_networks import NeuralNetwork
from mereli.neural_networks.utils import *
from mereli.utils import sigmoid
from mereli.neural_networks.utils import plot_weights 




def iterative_LS(x, y, beta_prev, n, alpha=0.5):
    # x = np.r_[x,1]
    delta_beta = (1/n)**(alpha) * x * (y - x.dot(beta_prev))
    return beta_prev + delta_beta

def data_generator(N=1000):
    X_0 = np.random.normal([0,0], .1, [N//4, 2]) 
    X_1 = np.random.normal([1,1], .1, [N//4, 2]) 
    X_2 = np.random.normal([1,0], .1, [N//4, 2]) 
    X_3 = np.random.normal([0,1], .1, [N//4, 2]) 
    X = np.vstack((X_0, X_1, X_2, X_3))
    y = np.array([1] * int(N//2) + [0] * int(N//2))
    reindx = np.arange(N)
    np.random.shuffle(reindx)
    return X[reindx], y[reindx]

X, y = data_generator()
# plt.scatter(X[:,0], X[:,1], color=['r' if c else 'b' for c in y]); plt.show()

dt = 0.1
ann = NeuralNetwork(dt, neuron_model='rate_model', synapse_model='static_synapse')
ann.add_stimuli('I1', 2)
# ann.add_stimuli('I2', 1)
ann.add_ensemble('H1', 50, tau=10*dt)
ann.set_motor('H1')

ann.add_synapse('I1-H1', 'I1', 'H1', weight='random', p=0.8)
# ann.add_synapse('I1-H2', 'I1', 'H2', weight='random', p=0.8)
ann.add_synapse('H1-H1', 'H1', 'H1', weight='random', p=0.6)
# ann.add_synapse('H1-H2', 'H1', 'H2', weight='random', p=0.7)
# ann.add_synapse('H2-H2', 'H2', 'H2', weight='random', p=0.6)
# ann.add_synapse('H2-H1', 'H2', 'H1', weight='random', p=0.7)
ann.add_decoder('IdentityDecoding', 'H1', 'A')

#! OJO ESTA LINEA ES TEMPORAL HASTA MEJORAR CODIGO LEARNING RULES
ann.build()
ann.reset()

done = False
trial = 0
beta = np.zeros(50)
n = 1
for trial in range(50):
    ann.reset()#! OJO BUILD
    acc = []
    for x_i, y_i in zip(X, y):
        acc = []
        ann.reset()
        for t in range(20):
            try:
                activities = np.array(ann.step({'I1' : 3*x_i})['A'])
                y_hat = activities.dot(beta)
                beta = iterative_LS(activities, y_i, beta, n, alpha=0.5)
            except:
                import pdb; pdb.set_trace() 
            n += 1
        y_cls = int(y_hat >= 0.5)
        correct = y_i == y_cls
        acc.append(correct)
    acc = np.mean(acc)
    print('Trial=',trial,' ACC=',acc)



#* TEST
ann.reset()
global_states.set_states(debug=True, info=True)
for x_i, y_i in zip(X, y):
    # ann.reset()
    try:
        activities = [np.array(ann.step({'I1' : 3*x_i})['A']) for _ in range(60)]
    except:
        import pdb; pdb.set_trace()
    y_hat = activities[-1].dot(beta)
    y_hat = int(y_hat >= 0.5)
    print(y_i, y_hat)
    plt.scatter(x_i[0], x_i[1], color='r' if y_hat  else 'b')
plt.show()
import pdb; pdb.set_trace()


oo = np.stack(tuple(ann.monitor.get('outputs').values()))
# plt.plot(outputs)
# plt.plot(stimuli)
# plt.show()


