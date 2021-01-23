from itertools import chain, product
from collections import deque
from functools import wraps
from abc import ABC, abstractmethod, abstractproperty
import numpy as np
from spike_swarm_sim.neural_networks.utils.builder import SynapsesBuilder
from spike_swarm_sim.algorithms.interfaces import GET, SET, LEN, INIT
from spike_swarm_sim.utils import merge_dicts
from spike_swarm_sim.register import synapse_registry

def delay(func):
    @wraps(func)
    def wrapper(self, spikes, voltages, **kwargs):
        spikes = spikes.astype(bool)
        delayed_spikes = np.array([v.pop() for v in self.spike_buffer])
        any([v.appendleft(bool(s)) for s, v in zip(spikes, self.spike_buffer)])
        ret = func(self, delayed_spikes.copy(), voltages, **kwargs)
        return ret
    return wrapper  

class Synapses(ABC):
    def __init__(self, dt):
        self.dt = dt
        #* Weighted adjacency matrix.
        self.weights = None
        #* Unweighted boolean adjacency matrix (w=0 != no connection)
        self.mask = None
        #* Mask of connection that can be optimized.
        self.trainable_mask = None
        #* Dict mapping the name of each synapse group to the indices in
        #* The adjacency matrix.
        self.connection_dict = {}
        #* List of connections that are from sensory inputs.
        self.sensory_connections = []
        #* List of connections that feed motor neurons.
        self.motor_connections = []
        #* List of hidden connections (neither sensory nor motor connections).
        self.hidden_connections = []
        #*
        self.recurrent_connections = []

    @abstractmethod
    def step(self, spikes, voltages):
        pass
    @abstractmethod
    def build(self, topology):
        pass
    @abstractmethod
    def reset(self):
        pass
    
    @GET("synapses:weights")
    def get_weights(self, conn_name, min_val=0., max_val=1., only_trainable=True):
        """
        Given a connection name the method returns the flattened array of synapse strengths in
        that connection. If only_trainable is active, only weights in train mode are returned.
        """
        #* Return scaled in [0,1]
        if conn_name == 'all':
            return (self.weights[self.trainable_mask].copy() - min_val) / (max_val - min_val)
        #* Special queries of synapses
        conn_name = {
            'sensory' : self.sensory_connections,
            'hidden' : self.hidden_connections,
            'motor' : self.motor_connections
        }.get(conn_name, [conn_name])
        conn_subdict = merge_dicts([self.connection_dict[k] for k in conn_name])
        weights_string = np.hstack([self.weights[post[0]:post[1], pre[0]:pre[1]].flatten().copy() \
                         for pre, post in zip(conn_subdict['pre'], conn_subdict['post'])])
        if only_trainable:
            mask = np.hstack([self.trainable_mask[post[0]:post[1], pre[0]:pre[1]].flatten().copy()\
                        for pre, post in zip(conn_subdict['pre'], conn_subdict['post'])])
            return (weights_string[mask].flatten() - min_val) / (max_val - min_val)
        else:
            return (weights_string.flatten() - min_val) / (max_val - min_val)

    @SET("synapses:weights")
    def set_weights(self, conn_name, data, min_val=0., max_val=1.,):
        """
        """
        #* rescale genotype segment to weight range
        data = min_val + data * (max_val - min_val)
        if conn_name == 'all':
            self.weights[self.trainable_mask] = data[:self.trainable_mask.sum()].copy()
            # self.weights = np.abs(self.weights)
        else:
            #* Special queries of synapses
            conn_name = {
                'sensory' : self.sensory_connections,
                'hidden' : self.hidden_connections,
                'motor' : self.motor_connections
            }.get(conn_name, [conn_name])
            conn_subdict = merge_dicts([self.connection_dict[k] for k in conn_name])
            # conn_subdict = self.connection_dict[conn_name]
            counter = 0
            for pre, post in zip(conn_subdict['pre'], conn_subdict['post']):
                mask = self.trainable_mask[post[0] : post[1], pre[0] : pre[1]].copy()
                subconn_len = mask.sum()
                self.weights[post[0]:post[1], pre[0]:pre[1]][mask] = data[counter:counter+subconn_len].copy()
                counter += subconn_len
            # self.weights = np.abs(self.weights)
            # import pdb; pdb.set_trace()

    @INIT('synapses:weights')
    def init_weights(self, conn_name, min_val=0., max_val=1., only_trainable=True):
        """
        """
        weights_len = self.len_weights(conn_name)
        random_weights = 0.5 + np.random.randn(weights_len)*0.25 #between 0 and 1 (denormalized in set)
        random_weights = np.clip(random_weights, a_min=0, a_max=1)
        self.set_weights(conn_name, random_weights, min_val=min_val, max_val=max_val)

    @LEN('synapses:weights')
    def len_weights(self, conn_name, only_trainable=True):
        """
        """
        return self.get_weights(conn_name, only_trainable=True).shape[0]

@synapse_registry(name='static_synapse')
class StaticSynapses(Synapses):
    def step(self, spikes, voltages):
        return self.weights.dot(spikes)

    def build(self, topology):
        np.random.seed(821)
        # Instantiate builder and guide it
        builder = SynapsesBuilder(topology)
        # build connection mask and weights skeleton (not initialized)
        self.weights = builder.build_connections()
        self.mask = self.weights.copy().astype(bool)
        # Build trainable mask, indicating whether the connection can be optimized or not.
        # Non trainable weights are initialized randomly.
        self.trainable_mask, self.weights = builder.build_trainable_mask(self.mask, self.weights)
        np.random.seed()
        self.connection_dict = builder.connection_dict
        self.sensory_connections = [syn_name for syn_name, syn in topology['synapses'].items()\
                                            if syn['pre'] in topology['stimuli'].keys()]
        motor_neurons = [mot_neuron for output in topology['outputs'].values() for mot_neuron in output['ensemble']]
        self.motor_connections = [syn_name for syn_name, syn in topology['synapses'].items()\
                                    if syn['post'] in motor_neurons]
        self.hidden_connections = [syn for syn in topology['synapses'].keys() \
                                       if syn not in self.sensory_connections + self.motor_connections]
        return builder.ensemble_pointers, {n : u['n'] for n, u in builder.overall_topology.items()}, builder.n_inputs
    
    def reset(self):
        pass

    #! Remove (after test)
    def initialize(self):
        self.weights[self.trainable_mask] = (np.random.random(size=np.sum(self.trainable_mask))) * 1

@synapse_registry(name='dynamic_synapse')
class DynamicSynapses(Synapses):
    def __init__(self, dt):
        super(DynamicSynapses, self).__init__(dt)
        # masks and matrices
        self.gaba_mask = None
        self.ampa_mask = None
        self.ndma_mask = None
        self.delays = None
        self.spike_buffer = None
        
        # synapse variables (initialized in reset)
        self.s_ampa, self.s_gaba, self.s_ndma = None, None, None
        self.x_ndma = None
        
        #* Synapse Parameters (default)
        #* Changed in build method
        self.ampa_gain = 0.6
        self.ampa_tau = 5.
        self.ampa_E = 0.0
        self.gaba_gain = 1.
        self.gaba_tau = 10
        self.gaba_E = -70.
        self.ndma_gain = 0.1
        self.ndma_tau_rise = 6
        self.ndma_tau_decay = 100
        self.ndma_Mg2 = 0.5
        self.ndma_E = 0.0

    @delay
    def step(self, spikes, voltages):
        Iampa = self.step_ampa(spikes, voltages)
        Igaba = self.step_gaba(spikes, voltages)
        Indma = self.step_nmda(spikes, voltages)
        I = Iampa + Igaba + Indma
        return I

    def step_ampa(self, spikes, voltages):
        self.s_ampa[spikes] = 1
        self.s_ampa[~spikes] += self.dt * (-self.s_ampa[~spikes]/self.ampa_tau) 
        PSP = self.ampa_gain * (self.ampa_mask * self.weights).dot(self.s_ampa)# * (self.ampa_E - voltages) 
        if self.ampa_E is not None: PSP *= (self.ampa_E - voltages)
        return PSP

    def step_gaba(self, spikes, voltages):
        self.s_gaba[spikes] = 1
        self.s_gaba[~spikes] += self.dt * (-self.s_gaba[~spikes]/self.gaba_tau)
        PSP = self.gaba_gain*(self.gaba_mask * self.weights).dot(self.s_gaba)#* (self.gaba_E - voltages) 
        if self.gaba_E is not None: PSP *= (self.gaba_E - voltages)
        return PSP

    def step_nmda(self, spikes, voltages):
        self.x_ndma[spikes] = 1
        self.x_ndma[~spikes] += self.dt * (-self.x_ndma[~spikes] / self.ndma_tau_rise)
        self.s_ndma += self.dt * (.5*self.x_ndma*(1-self.s_ndma) - self.s_ndma/self.ndma_tau_decay) 
        PSP = self.ndma_gain * (self.ndma_mask * self.weights).dot(self.s_ndma)\
                * ( 1 / (1 + self.ndma_Mg2 * np.exp(-0.062*voltages)/3.57))
        if self.ndma_E is not None: PSP *= (self.ndma_E - voltages)
        return PSP

    def build(self, topology):
        np.random.seed(3657)
        # Instantiate builder and guide it
        builder = SynapsesBuilder(topology)
        # build connection mask and weights skeleton (not initialized)
        self.weights = builder.build_connections()
        self.mask = self.weights.copy().astype(bool)
        
        # Build neurotransmitter masks of the synapses
        neurotransmitters = builder.build_neurotransmitters(self.mask.copy())
        self.ampa_mask = neurotransmitters['AMPA']
        self.gaba_mask = neurotransmitters['GABA']
        self.ndma_mask = neurotransmitters['NDMA']

        # Build trainable mask, indicating whether the connection can be optimized or not.
        # Non trainable weights are initialized randomly.
        self.trainable_mask, self.weights = builder.build_trainable_mask(self.mask, self.weights)

        # Set up delays (only axonal or neuron level)
        self.delays, self.spike_buffer = builder.build_delays(min_delay=1, max_delay=10)
        np.random.seed()
        self.connection_dict = builder.connection_dict
        if 'synapse_params' in topology.keys():
            synapse_params = topology['synapse_params']
            self.ampa_gain = synapse_params['AMPA']['gain']
            self.ampa_tau = synapse_params['AMPA']['tau']
            self.ampa_E = synapse_params['AMPA']['E'] 
            self.gaba_gain = synapse_params['GABA']['gain']
            self.gaba_tau = synapse_params['GABA']['tau']
            self.gaba_E = synapse_params['GABA']['E'] 
            self.ndma_gain = synapse_params['NDMA']['gain']
            self.ndma_tau_rise = synapse_params['NDMA']['tau_rise']
            self.ndma_tau_decay = synapse_params['NDMA']['tau_decay']
            self.ndma_Mg2 = synapse_params['NDMA']['Mg2']
            self.ndma_E = synapse_params['NDMA']['E']
            if self.gaba_E is None:
                self.gaba_gain *= -1
        return builder.ensemble_pointers, {n : u['n'] for n, u in builder.overall_topology.items()}, builder.n_inputs
    
    def initialize(self):
        # initialize normal connections
        self.weights[self.trainable_mask]=(np.random.random(\
                size=np.sum(self.trainable_mask))) * 1

    def reset(self):
        self.s_ampa = np.zeros(self.weights.shape[1])
        self.s_gaba = np.zeros(self.weights.shape[1])
        self.s_ndma = np.zeros(self.weights.shape[1])
        self.x_ndma = np.zeros(self.weights.shape[1])
        self.spike_buffer = [deque([False for _ in range(int(d))]) for d in self.delays]


    @GET('synapses:delays')
    def get_delays(self, neuron_name):
        return self.delays.copy()

    @SET('synapses:delays')
    def set_delays(self, neuron_name, data):
        self.delays = data.copy()

    @LEN('synapses:delays')
    def len_delays(self, neuron_name):
        return self.delays.shape[0]

    @INIT('synapses:delays')
    def init_delays(self, neuron_name, min_val=0., max_val=1.):
        self.delays = np.random.randint(min_val, max_val, size=self.delays.shape[0])
    
    def trainable_weights(self, shared=False):
        if not shared:
            return self.weights[self.trainable_mask&~self.shared_weights_mask]
        else:
            return np.array([v[-1] for v in self.shared_weights_ids])

    def normalize_weights(self):
        for i, w_row in enumerate(self.weights):
            self.weights[i] /= np.linalg.norm(w_row)