
import sys
import os
sys.path.append('D:\subversion\Mereli')
import numpy as np
import matplotlib.pyplot as plot


from mereli import World 
from mereli.objects import Robot
from mereli.controllers import NeuralController
from mereli.algorithms.evolutionary import xNES


class LorentzAttractor:
    def __init__(self, state_0):
        self.dt = 1e-3
        self.a = 10
        self.b = 99.6
        self.c = 8/3

        self.x = state_0[0]
        self.y = state_0[1]
        self.z = state_0[2]
        self.t = 0

    def step(self):
        self.x += self.dt * self.a * (self.y - self.x)
        self.y += self.dt * (self.x * (self.b - self.z) - self.y)
        self.z += self.dt * (self.x * self.y - self.c * self.z)
        self.t += 1

    def reset(self):
        self.t = 0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

    def state(self):
        return np.r_[self.x, self.y, self.z] / 100

class fitness:
    def __init__(self,):
        self.required_info = ()

    def __call__(self, actions, states, info=None):
        actions = np.stack([ac[0]['IR_transmitter']['msg'] for ac in actions])
        fitness = 0.0
        lorentz_sys = LorentzAttractor(actions[0])
        for ac in actions:
            lorentz_sys.step()
            fitness += (1 - np.mean((lorentz_sys.state() - ac)**2)**0.5)
        return fitness / len(actions)


ctrnn = {
    "dt" : 0.1, "time_scale" : 1, "stimuli" : {}, "encoding" : {},
    "stimuli": {"W1" : {"n" : 2, "sensor" : "wireless_receiver:msg"}},
    "encoding" : {"W1" :{"scheme" : "IdentityEncoding", "receptive_field" : {}}},
    "neurons" : {"model": "ctrnn", "params": {"tau":0.35, "gain" : 1.0, "bias" : 0}},
    "ensembles": {
        "H" : {"n" : 2, "tau" : 10, "gain" : 1, "bias": 0, "activation":"sigmoid"},
    },
    "outputs" : {"OUT" : {"ensemble" : ["H"], "actuator" : "IR_transmitter", "enc": "real"}},
    "synapses" :  {
        "i-h": {"pre":"W1", "post":"H", "trainable":True, "p":1},
        "h-h": {"pre":"H", "post":"H", "trainable":True, "p":1},
    },
    "decoding" : {"OUT" : {"scheme" : "IdentityDecoding", "params" : {"is_cat" : False}}}
}


debug = True
world = World(render=False, render_connections=False)
sensors = {"wireless_receiver" : {"msg_length" : 2}}
actuators = {"IR_transmitter" : {"msg_length" : 2}}
neural_controller = NeuralController(ctrnn, sensors, actuators, debug_options=debug)
abstract_robot = Robot(np.zeros(2), controller=neural_controller, trainable=True)
world.add('abstract_robot', abstract_robot)


pop_info = {'p1' : {"objects" : ["synapses:weights:all"], "min_vals" : -1, "max_vals":1,\
                    "encoding" : "real", "selection_operator" : "tournament",\
                    "crossover_operator" : "blxalpha", "mutation_operator" : "gaussian",
                    "mating_operator" : "random", "mutation_prob" : 0.05, "crossover_prob" : 0.9,
                    "num_elite" : 5}}
alg = xNES(pop_info, world, [None], fitness_fn=fitness(), population_size=10, n_generations=50,\
                eval_steps=10000, num_evaluations=1, checkpoint_name=None)
alg.run()


#* ---- TEST RESULTS --- #
neural_controller.reset()
st_0 = np.array(neural_controller.neural_network.step({})['OUT'])
lorentz_sys = LorentzAttractor(st_0)
true_values = lorentz_sys.state()
pred_values = st_0.copy()
for t in range(10000):
    lorentz_sys.step()
    pred = np.array(neural_controller.neural_network.step({})['OUT'])
    true_values = np.vstack((true_values, lorentz_sys.state()))
    pred_values = np.vstack((pred_values, pred))

# import pdb; pdb.set_trace()
plot.plot(true_values[:, 0], true_values[:, 1])
plot.plot(pred_values[:, 0], pred_values[:, 1])
plot.legend(['TRUE', 'PRED'])
plot.show()
