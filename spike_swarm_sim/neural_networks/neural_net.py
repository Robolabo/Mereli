import copy
import logging
from functools import wraps
import numpy as np
import matplotlib.pyplot as plt
# Own imports
from spike_swarm_sim.register import neuron_models, synapse_models
from spike_swarm_sim.utils import increase_time, merge_dicts, remove_duplicates
from .neuron_models import NonSpikingNeuronModel, SpikingNeuronModel
from .decoding import DecodingWrapper
from .encoding import EncodingWrapper
from .update_rules import GeneralizedABCDHebbian, BufferedHebb
from .utils.monitor import NeuralNetMonitor
from .utils.visualization import *

def monitor(func):
    """ Decorator for recording and monitoring the relevant neuronal variables. 
    The records are stored in the monitor attribute of the NeuralNetwork class.
    """
    @wraps(func)
    def wrapper(self, encoded_stimuli, **kwargs):
        spikes, Isynapses, voltages = func(self, encoded_stimuli, **kwargs)
        #* Debugging and Monitoring (debug option must be enabled)
        if self.monitor is not None:
            monitor_vars = {
                # 'encoded_inputs' : encoded_stimuli.copy(),
                'stimuli' : np.hstack(tuple(self.stimuli.values())).copy(),
                'voltages' : voltages.copy(),
                'currents' : Isynapses.copy(),
                'outputs' : spikes.copy()
            }
            if issubclass(type(self.neurons), SpikingNeuronModel):
                monitor_vars.update({
                    'encoded_inputs' : encoded_stimuli.copy(),
                    'spikes' : spikes.copy(),
                    'recovery' : self.neurons.recovery.copy(),
                    'neuron_theta' : self.neurons.theta.copy(),
                    'activities' : tuple([v.activities.copy() for v in self.decoders.all.values()][0])#!
                })
            self.monitor.update(**monitor_vars)
        return spikes, Isynapses, voltages
    return wrapper

class NeuralNetwork:
    """ Class for the artificial neural networks. This class is mainly a wrapper that creates and executes 
    the main building blocks of ANNs. These blocks are encoding, synapses, neurons and decoding, albeit there 
    are other functionalities such as learning rules, monitors, and so on. This class encompasses any kind 
    of neural network, the precise architecture and dynamics will be fixed by the neuron and synapses models 
    throughout the topology dictionary.
    ==========================================================================================================
    - Params:
        topology [dict] : dictionary specifying the ANN architecture (see configuration files for more details).
    - Attributes:
        dt [float] : Euler step of the ANN.
        t [int] : time counter.
        time_scale [int] : ratio between neuronal and environment dynamics. This means that every time step of 
                    the env., the ANN performs time_scale updates.
        synapses [Synapses] : object storing the synapse models.
        stim_encoding [dict] : dict of sensor_name : Encoding object storing all the neural encoders.
        pointers [dict] : dict mapping ensembles to the index in the ANN adjacency matrix. The index is only 
                    the index of the last neuron of the ensemble.
        subpop_neurons [dict] : dict mapping ensembles to number of neurons per ensemble.
        n_inputs [int] : number of ANN inputs (after decoding). 
        stimuli_order [list of str]: ordered list with the sensor order as specified in the ANN config.
        neurons [SpikingNeuronModel or NonSpikingNeuronModel] : object storing the neurons of the ANN.
        update_rule : #TODO
        output_neurons [list] : list with the name of the motor/output ensembles.
        monitor [NeuralNetMonitor or None]: Monitor to record neuronal variables if mode is DEBUG.
        spikes [np.ndarray of shape=num_neurons]: current generated spikes.
        stimuli [dict] : current supplied stimuli. Dict mapping sensor name to stimuli values. 
        action_decoding [dict] :  dict of action_name : Decoding object storing all the neural decoders.
    ==========================================================================================================
    """
    def __init__(self, topology):
        self.dt = topology['dt']
        self.t = 0
        self.time_scale = topology['time_scale'] #* ANN steps per world step.

        #* --- Create and build Synapses ---
        if topology['synapse_model'] == 'dynamic_synapse' and \
            issubclass(neuron_models[topology['neuron_model']], NonSpikingNeuronModel):
            raise Exception(logging.error('The combination of dynamic synapses and '\
                'non-spiking neuron models is not currently implemented.'))
        self.synapses = synapse_models[topology['synapse_model']](self.dt)

        #* --- Create and build Encoders ---
        self.encoders = EncodingWrapper(topology)
      
        #* --- Create and Build Topology ---
        #* pointers point to the index of the last neuron of subpopulations
        #* in the weight matrix (includes inputs)
        self.pointers, self.subpop_neurons, self.n_inputs = self.build(topology.copy())
        self.stimuli_order = [v['sensor'] for v in topology['stimuli'].values()]

        #* --- Create and build Neurons ---
        self.neurons = neuron_models[topology['neuron_model']](self.dt,\
                sum([ens['n'] for ens in topology['ensembles'].values()]),\
                **merge_dicts([{param : ens['n'] * [val]\
                for param, val in ens['params'].items()}\
                for ens in topology['ensembles'].values()]))

        #TODO --- Create Learning Rule ---
        self.update_rule = BufferedHebb()

        #* --- Create Monitor (DEBUG MODE) ---
        self.output_neurons = remove_duplicates([out['ensemble'] for out in topology['outputs'].values()])
        if logging.root.level == logging.DEBUG:
            self.monitor = NeuralNetMonitor({key : val['n'] for key, val in topology['ensembles'].items()},\
                        {val['sensor'] : val['n'] for val in topology['stimuli'].values()},\
                        {key : self.subpop_neurons[key] for key in topology['stimuli'].keys()}, self.output_neurons)
        else:
            self.monitor = None

        #* --- Create and build Decoders ---
        self.decoders = DecodingWrapper(topology)
        #* Aux vars of current stim. and spikes.
        self.stimuli, self.spikes = None, None
        #* --- Reset dynamics ---
        self.reset()

    @increase_time
    @monitor
    def _step(self, stimuli):
        """
        Private method devoted to step the synapses and neurons sequentially. 
        ======================================
        - Args:
            stimuli [dict]: dict mapping stimuli name and numpy array containing its values.
        - Returns:
            spikes [np.ndarray]: boolean vector with the generated spikes.
            soma_currents [np.ndarray]: vector of currents injected to the neurons.
            voltages [np.ndarray]: vector of membrane voltages after neurons step.
        ======================================
        """
        soma_currents = self.synapses.step(np.r_[stimuli, self.spikes], self.voltages)
        spikes, voltages = self.neurons.step(soma_currents)
        return spikes, soma_currents, voltages

    def step(self, stimuli, reward=None):
        """
        Simulation step of the neural network.
        It is composed by four main steps:
            1) Encoding of stimuli to spikes (if SNN used).
            2) Synapses step.
            3) Neurons step.
            4) Decoding of spikes or activities into actions.
        ======================================
        - Args:
            stimuli [dict]: dict mapping stimuli name and numpy array containing its values.
        - Returns:
            actions [dict]: dict mapping output names and actions.
        ======================================
        """
        if hasattr(self.neurons, 'tau'):
            self.neurons.tau[-np.sum([self.subpop_neurons[kk] for kk in self.output_neurons]):] = 0.5 #!

        #* --- Convert stimuli into spikes (Encoders Step) ---
        if len(stimuli) == 0:
            raise Exception(logging.error('The ANN received empty stimuli.'))
        stimuli = {s : stimuli[s].copy() for s in self.stimuli_order}
        self.stimuli = stimuli.copy()
        inputs = self.encoders.step(stimuli)
        if self.time_scale == 1:
            inputs = inputs[np.newaxis]
        
        #TODO --- Apply update rules to synapses ---
        # if self.update_rule is not None and reward is not None:
        #     self.synapses.weights += self.update_rule.step(inputs[-1], self.spikes, reward=0.01)

        #* --- Step synapses and neurons ---
        spikes_window = []
        for tt, stim in enumerate(inputs):
            spikes, _, _ = self._step(stim)
            self.spikes = spikes.copy()
            spikes_window.append(spikes.copy())
        spikes_window = np.stack(spikes_window)

        #* --- Convert spikes into actions (Decoding Step) ---
        actions = self.decoders.step(spikes_window[:, self.motor_neurons])

        #* --- Debugging stuff (DEBUG MODE) --- #
        if self.t == self.time_scale * 1000 and self.monitor is not None:
            vv = np.stack(tuple(self.monitor.get('voltages').values()))
            ii = np.stack(tuple(self.monitor.get('stimuli').values()))
            # plot_spikes(self)
            import pdb; pdb.set_trace()
        return actions
    
    @property
    def num_neurons(self):
        """ Number of neurons in the ANN (non-input). """
        return self.voltages.shape[0]

    @property
    def motor_neurons(self):
        """ Indices of motor neurons without counting input nodes. When addressing the 
        weight matrix or any kind of ANN adj. mat., the number of inputs MUST be added.
        """
        return np.hstack([self.ensemble_indices(out) for out in self.output_neurons])
    
    def ensemble_indices(self, ens_name):
        """ Indices of the neurons of the requested ensemble. """
        return np.arange(self.pointers[ens_name] - self.subpop_neurons[ens_name] - self.n_inputs,\
                self.pointers[ens_name] - self.n_inputs)

    @property
    def is_spiking(self):
        """ Whether the neural network is a spiking neural network or not. """
        return issubclass(type(self.neurons), SpikingNeuronModel)

    @property
    def voltages(self):
        """Getter instantaneous voltage vector (membrane voltage of each neuron membrane)
        at current simulation timestep."""
        return self.neurons.voltages

    @property
    def weights(self):
        "Getter of the numpy weight matrix."
        return self.synapses.weights

    def build(self, topology):
        """ Builder of the ANN topology. """
        return self.synapses.build(copy.deepcopy(topology))

    def reset(self):
        """ Reset process of all the neural network dynamics. """
        self.t = 0
        self.neurons.reset()
        self.synapses.reset()
        self.encoders.reset()
        self.decoders.reset()
        self.update_rule.reset()
        if self.monitor is not None:
            self.monitor.reset()
        self.spikes = np.zeros(self.weights.shape[0])
        self.stimuli = None