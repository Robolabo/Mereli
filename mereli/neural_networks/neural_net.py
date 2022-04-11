import logging
from functools import wraps
import numpy as np

# Own imports
from mereli.register import neuron_models, synapse_models
from mereli.utils import increase_time
from mereli.neural_networks.base_neural_net import BaseNeuralNet
from .neuron_models import  SpikingNeuronModel
from .encoding import EncodingWrapper
try:
    from .utils.visualization import *
except:
    pass


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
                    'activities' : np.hstack([v.activities.copy() for v in self.decoders.all.values()])#!
                })
            self.monitor.update(**monitor_vars)
        return spikes, Isynapses, voltages
    return wrapper





class NeuralNetwork(BaseNeuralNet):
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
    def __init__(self, dt, neuron_model='rate_model', synapse_model='static_synapse', time_scale=1):
        self.t = 0
        self.dt = dt #* Euler Step
        self.time_scale = time_scale #* ANN steps per world step.
        self.neuron_model = neuron_model
        self.synapse_model = synapse_model
        neurons = neuron_models[self.neuron_model](self.dt)
        synapses = synapse_models[self.synapse_model](self.dt)
        super(NeuralNetwork, self).__init__(neurons, synapses, encoders=EncodingWrapper(self.time_scale))
        self.learning_rule = None #! OJO: provisional
        self.ww_buffer = []

    @increase_time
    @monitor
    def _step(self, stimuli):
        """ Private method devoted to step the synapses and neurons sequentially. 
        ====================================================================================
        - Args:
            stimuli [dict]: dict mapping stimuli name and numpy array containing its values.
        - Returns:
            spikes [np.ndarray]: boolean vector with the generated spikes.
            soma_currents [np.ndarray]: vector of currents injected to the neurons.
            voltages [np.ndarray]: vector of membrane voltages after neurons step.
        ====================================================================================
        """
        soma_currents = self.synapses.step(np.r_[stimuli, self.spikes], self.voltages)
        # import pdb; pdb.set_trace()
        spikes, voltages = self.neurons.step(soma_currents)
        return spikes, soma_currents, voltages

    def step(self, stimuli, reward=None):
        """ Simulation step of the neural network.
        It is composed by four main steps:
            1) Encoding of stimuli to spikes (if SNN used).
            2) Synapses step.
            3) Neurons step.
            4) Decoding of spikes or activities into actions.
        ===============================================================
        - Args: 
            stimuli [dict]: dict mapping stimuli name and numpy array 
                    containing its values.
        - Returns:
            actions [dict]: dict mapping output names and actions.
        ===============================================================
        """
        task = stimuli['task']
        # stimuli['distance_sensor'] *= 0.
        #* --- Convert stimuli into spikes (Encoders Step) ---
        if len(stimuli) == 0 or stimuli is None:
            stimuli = {'dummy_input' : np.array([])}
            # raise Exception(logging.error('The ANN received empty stimuli.'))
        stimuli = {s : stimuli[s].copy() for s in self.stimuli_names}
        inputs = self.encoders.step(stimuli)
        self.stimuli = stimuli.copy()
        if self.time_scale == 1:
            inputs = inputs[np.newaxis]

        #* --- Apply update rules to synapses ---
        if self.learning_rule is not None and self.t > 1:
            # If reward is None  while learning rule is not, then 
            # assume that it is a non modulated learning rule.
            if reward is None:
                reward = 1.
            # Use inputs and neuron outputs of previous time step.
            self.synapses = self.learning_rule.step(self.synapses, self.spikes, self.prev_input, reward=reward)

        #* --- Step synapses and neurons ---
        spikes_window = []
        for tt, stim in enumerate(inputs):
            spikes, _, _ = self._step(stim)
            self.spikes = spikes.copy()
            spikes_window.append(spikes.copy())
        spikes_window = np.stack(spikes_window)
        #* --- Convert spikes into actions (Decoding Step) ---
        actions = self.decoders.step(spikes_window[:, self.motor_neurons])
        self.prev_input = inputs[-1].copy()
        #* --- Debugging stuff (DEBUG MODE) --- #
        if self.t == self.time_scale * 2999 and self.monitor is not None:
            oo = np.stack(tuple(self.monitor.get('outputs').values()))
            ii = np.stack(tuple(self.monitor.get('stimuli').values()))
            II = np.stack(tuple(self.monitor.get('currents').values()))
            vv = np.stack(tuple(self.monitor.get('voltages').values()))
            # grasp0 = self.monitor.get('outputs')['OUT_GRASP_0']
            # plot_spikes(self)
            # ww = np.stack(self.ww_buffer)
            import pdb; pdb.set_trace()
        # actions['outB'] = [np.sin(2*np.pi*self.t*0.01)]
        # self.ww_buffer.append(self.weights)
        # if stimuli['reward'] > 0.01:
        # actions['outA'] = [1, -1]
        # if task == 1:
        #     actions['outB'] = [1]
        # else: 
        #     actions['outB'] = [0]
        # else:
        #     # if any(stimuli['mean_neigh_state'] == 1):
        #     actions['outB'] = [0,0]
            # else: 
            #     actions['outB'] = [0, 0]
        # import pdb; pdb.set_trace()
        return actions
    
    def reset(self):
        """ Reset process of all the neural network dynamics. """
        self.t = 0
        #! self.build()
        self.neurons.reset()
        self.synapses.reset()
        self.encoders.reset()
        self.decoders.reset()
        if self.learning_rule is not None:
            self.learning_rule.reset()
        if self.monitor is not None:
            self.monitor.reset()
        self.spikes = np.zeros(self.weights.shape[0])
        self.stimuli = None
        self.prev_input = None
        
