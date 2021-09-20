import logging
from itertools import product
import numpy as np


from spike_swarm_sim.neural_networks.neural_net import NeuralNetwork 
from .synapses import StaticSynapses
from .neuron_models import Perceptron
from spike_swarm_sim.register import neuron_models, synapse_models
from spike_swarm_sim.utils import merge_dicts

class MLP:

    def __init__(self, ):
        self.synapses = StaticSynapses()
        self.neurons = Perceptron(None)
        #* Overall ANN directed graph description.
        self.graph = {'inputs' : {}, 'neurons' : {}, 'synapses' : {}}
        self.ensemble_names = []
        self.input_ensemble_names = []
        self.motor_ensemble_names = []

        #* Ordered list of stimuli names (not input nodes)
        self.stimuli_names = []

    def step(self, inputs):
        ready = [True] * len(inputs) + [False] * len(self.voltages)
        while not all(ready):
            currents = self.synapses.step(np.r_[inputs, self.voltages], None)
            outputs = self.neurons.step(currents)[0]
            ready[len(inputs):] = np.logical_or(ready[len(inputs):], self.weights[:, ready].sum(1) != 0.0)
        #! No decoders ftm
        return outputs[np.array([self.ensemble_indices(out) for out in self.motor_ensemble_names]).flatten()]



    def build_from_dict(self, topology):
        #* Add neurons
        for name, ensemble in topology['ensembles'].items():
            self.add_ensemble(name, ensemble['n'], **ensemble['params'])
        #* Add stimuli
        for name, stim in topology['stimuli'].items():
            self.add_stimuli(name, stim['n'], stim['sensor'])
        #* Add motor ensembles
        for out in topology['outputs'].values():
            self.set_motor(out['ensemble'])
        #* Add Synapses
        for name, syn in topology['synapses'].items():
            syn_params = {key : val for key, val in syn.items()\
                    if key not in ['pre', 'post', 'p', 'trainable']}
            self.add_synapse(name, syn['pre'], syn['post'], conn_prob=syn['p'], use_seed=True, **syn_params)
        #* Build ANN
        self.build()

    def build(self):
        #! BUILD NEURONS?
        #* Build synapses
        self.synapses.build(self.graph)
        assert len(self.stimuli_names) > 0

    def set_motor(self, ensemble_name):
        if ensemble_name not in self.ensemble_names:
            raise Exception(logging.error('Ensemble "{}" does not exist').format(ensemble_name))
        if ensemble_name in self.motor_ensemble_names:
            return
        self.motor_ensemble_names.append(ensemble_name)
        for neuron in self.graph['neurons'].values():
            if neuron['ensemble'] == ensemble_name:
                neuron['is_motor'] = True

    def add_stimuli(self, name, num_nodes, sensor=None):
        for n in range(num_nodes):
            self.graph['inputs'].update({
                '{}_{}'.format(name, n) : {'ensemble' : name, 'sensor' : sensor, 'idx': len(self.graph['inputs'])}
            })
        self.input_ensemble_names.append(name)
        if sensor is None:
            sensor = name
        self.stimuli_names.append(sensor)

    def add_ensemble(self, name, num_neurons, **kwargs):
        self.ensemble_names.append(name)
        for n in range(num_neurons):
            self.add_neuron('{}_{}'.format(name, n), ensemble=name, **kwargs)
        
    def add_neuron(self, name, ensemble=None, **kwargs):
        self.neurons.add(**kwargs)#!
        ensemble = ensemble if ensemble is not None else name
        if ensemble not in self.ensemble_names:
            self.ensemble_names.append(ensemble)
        self.graph['neurons'].update({name : merge_dicts([{'ensemble' : ensemble,
                'idx' : len(self.neurons)-1, 'is_motor' : False}, kwargs])})

    def delete_neuron(self, name):
        neuron_index = self.graph['neurons'][name]['idx']
        ensemble = self.graph['neurons'][name]['ensemble']
        self.neurons.delete(neuron_index)
        self.graph['neurons'].pop(name, None)
        #! Ojo index of other neurons?
        for neuron in self.graph['neurons'].values():
            if neuron['idx'] >= neuron_index:
                neuron['idx'] -= 1
        #* Remove ensemble if neuron was the only unit.
        if not any([neuron['ensemble'] == ensemble for neuron in self.graph['neurons'].values()]):
            self.ensemble_names.remove(ensemble)
        #* Remove any synapse with the neuron as pre or post
        for syn_name, syn in [*self.graph['synapses'].items()]:
            if syn['pre'] == name or syn['post'] == name:
                self.delete_synapse(syn_name)

    def add_synapse(self, name, pre, post, weight=1., conn_prob=1., 
            trainable=True, use_seed=False, **kwargs):
        """ Adds synapses between pre and post ensembles. """
        if post in self.graph['inputs'] or post in self.input_ensemble_names:
            raise Exception(logging.error('An input node or ensemble cannot be '\
                'a postsynaptic neuron or ensemble.'))
        #* Check if pre is neuron or ensemble.
        if pre not in merge_dicts([self.graph['inputs'], self.graph['neurons']]):
            if pre not in self.input_ensemble_names + self.ensemble_names:
                raise Exception(logging.error('Connection presynaptic neuron or ensemble '\
                    '"{}" does not exist').format(pre))
            pre = [name for name, node in merge_dicts([self.graph['inputs'], self.graph['neurons']]).items() if node['ensemble'] == pre]
        else:
            pre = [pre]
        #* Check if post is neuron or ensemble.
        if post not in merge_dicts([self.graph['inputs'], self.graph['neurons']]):
            if post not in self.input_ensemble_names + self.ensemble_names:
                raise Exception(logging.error('Connection postsynaptic neuron or ensemble '\
                    '"{}" does not exist').format(post))
            post = [name for name, node in merge_dicts([self.graph['inputs'], self.graph['neurons']]).items() if node['ensemble'] == post]
        else:
            post = [post]
        #* Add connections (note: not compatible with previous implementation checkpoints).
        #! REVISAR SEED
        if use_seed:
            np.random.seed(44 + len(self.graph['synapses']))
        for i, (pre_node, post_node) in enumerate(product(pre, post)):
            if np.random.random() < conn_prob:
                synapse_config = merge_dicts([{
                    'pre' : pre_node, 'post' : post_node,
                    'weight': weight if weight is not 'random' else  0.2*np.random.randn(), 
                    'trainable' : trainable,
                    'group' : name, 'idx' : len(self.graph['synapses']), 'enabled' : True}, kwargs])
                syn_name = "{}_{}".format(name, i) if len(pre + post) > 2 else name
                self.graph['synapses'].update({syn_name : synapse_config})
        if use_seed:
            np.random.seed(None)

    def delete_synapse(self, name):
        self.graph['synapses'].pop(name, None)
    
    def reset(self):
        """ Reset process of all the neural network dynamics. """
        self.t = 0
        #! self.build()
        self.neurons.reset()
        self.synapses.reset()


    def ensemble_indices(self, ens_name, consider_inputs=False):
        """ Indices of the neurons of the requested ensemble. """
        if ens_name not in self.ensemble_names:
            raise Exception(logging.error('Requested ensemble "{}" does not exist.'.format(ens_name)))
        
        indices = np.array([neuron['idx'] for neuron in self.graph['neurons'].values() if neuron['ensemble'] == ens_name])
        if consider_inputs:
            indices += self.num_inputs #!
        return indices

    @property
    def voltages(self):
        """Getter instantaneous voltage vector (membrane voltage of each neuron membrane)
        at current simulation timestep."""
        return self.neurons.voltages
    
    @property
    def weights(self):
        "Getter of the numpy weight matrix."
        return self.synapses.weights

    @property
    def num_neurons(self):
        """ Number of neurons in the ANN (non-input). """
        return len(self.neurons)

    @property
    def num_inputs(self):
        return len(self.graph['inputs'])
    
    @property
    def num_motor(self):
        return len(self.motor_neurons)
    
    @property
    def num_hidden(self):
        return self.num_neurons - self.num_motor

