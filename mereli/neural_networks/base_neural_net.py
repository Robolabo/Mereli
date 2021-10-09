import logging
import numpy as np
from itertools import product


from .decoding import DecodingWrapper
from .encoding import EncodingWrapper
from .neuron_models import NonSpikingNeuronModel, SpikingNeuronModel, Activation
from mereli.neural_networks.synapses import DynamicSynapses
from .utils.monitor import NeuralNetMonitor

class BaseNeuralNet:
    def __init__(self, neurons, synapses, encoders=None, decoders=None, monitor=None):
        self.synapses = synapses
        self.neurons  = neurons
        #* Monitor that, if in DEBUG mode, will store all the relevant neural variables.
        self.monitor = monitor

        #* Submodules of the neural network distributing its functioning
        #* and computations.
        if issubclass(type(synapses), DynamicSynapses) and issubclass(type(neurons), NonSpikingNeuronModel):
            raise Exception(logging.error('The combination of dynamic synapses and '\
                'non-spiking neuron models is not currently implemented.'))
        
        self.encoders = EncodingWrapper() if encoders is None else encoders
        self.decoders = DecodingWrapper() if decoders is None else encoders 

        #* Overall ANN directed graph description.
        self.graph = {'inputs' : {}, 'neurons' : {}, 'synapses' : {}}
        self.ensemble_names = [] # List of names of all non-input ensembles.
        self.input_ensemble_names = [] # List of names of all input ensembles.
        self.motor_ensemble_names = [] # List of names of ensembles with motor/output neurons.
        self.stimuli_names = [] # List of stimuli names (not input nodes)

        #* Flag indicating if the ANN is built and functional. 
        #* The ANN cannot be used if this flag is False.
        self.is_built = False #! USE IN CODE

        #* Variables storing the previous stim and spikes (CHECK IF NEEDED).
        self.stimuli, self.spikes, self.prev_input = None, None, None


    def step(self):
        pass

    def reset(self):
        """ Reset process of all the neural network dynamics. """
        self.t = 0 #!
        #! self.build()
        self.neurons.reset()
        self.synapses.reset()
        self.encoders.reset()
        self.decoders.reset()
        # if self.learning_rule is not None:
        #     self.learning_rule.reset()
        if self.monitor is not None:
            self.monitor.reset()
        self.spikes = np.zeros(self.weights.shape[0])
        self.stimuli = None
        self.prev_input = None

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
        #* Add encoders
        for input_name, encoder in topology['encoding'].items():
            self.add_encoder(encoder['scheme'], topology['stimuli'][input_name]['sensor'],\
                receptive_field=encoder['receptive_field']['name'], receptive_field_params=encoder['receptive_field']['params'])
        #! #* Add Learning Rule
        #! if topology.get('learning_rule', {}).get('rule') is not None:
        #!     self.learning_rule = learning_rules.get(topology.get('learning_rule', {}).get('rule'))() #TODO decouple, improve.
        #* Add Synapses
        for name, syn in topology['synapses'].items():
            syn_params = {key : val for key, val in syn.items()\
                    if key not in ['pre', 'post', 'p', 'trainable']}
            self.add_synapse(name, syn['pre'], syn['post'], conn_prob=syn['p'], use_seed=True, **syn_params)
        #* Add Decoders
        self.decoders.build(topology)
        #* Build ANN
        self.build()

    def build(self):
        #! BUILD NEURONS?
        #* Build synapses
        self.synapses.build(self.graph)
        #* Add dummy input if the ANN has no input.
        if len(self.stimuli_names) == 0: 
            self.stimuli_names.append('dummy_input')
        #* Add Identity encoders to every stimuli without 
        #* an encoder previously set.
        for stim in self.stimuli_names:
            if stim not in self.encoders.all:
                self.add_encoder('IdentityEncoding', stim)
        if self.learning_rule is not None:
            self.learning_rule.build(self.graph)
        #TODO --- Create Monitor (DEBUG MODE) ---
        # self.output_neurons = remove_duplicates([out['ensemble'] for out in topology['outputs'].values()])
        if logging.root.level == logging.DEBUG:
            self.monitor = NeuralNetMonitor({ens : self.num_ensemble_neurons(ens)\
                        for ens in self.ensemble_names},\
                        {name : self.encoders.get(name).n_stimuli for name in self.stimuli_names},\
                        {name : self.num_input_nodes(name) for name in self.input_ensemble_names},\
                        self.motor_ensemble_names)
        else:
            self.monitor = None
        # #* --- Reset dynamics ---
        # self.reset()

    def build_from_adjmat(self, w_matrix):
        #! OJO NORMALIZACION weights !!!
        assert w_matrix.shape[1] - w_matrix.shape[0] == self.num_inputs
        for i in range(w_matrix.shape[0]):
            name = f'H_{i}'
            import pdb; pdb.set_trace()
        import pdb; pdb.set_trace()



    def add_stimuli(self, name, num_nodes, sensor=None):
        """ Adds a stimuli or input node to the neural network. This stimuli can be an scalar (num_nodes=1) or 
        a vector. In most cases, it is associated to a physical sensor of the robot. The stimuli nodes are not 
        subject to encoding by their own. An encoded must be added (if required) separately.

        :param str name: name of the stimuli or input node.
        :param int num_nodes: number of nodes or dimension of the stimuli.
        :param str sensor: Reference name of the sensor associated to the stimuli. If it is None, then sensor=name.
        """
        for n in range(num_nodes):
            self.graph['inputs'][f'{name}_{n}'] = {'ensemble' : name, 'sensor' : sensor, 'idx': len(self.graph['inputs'])}
        self.input_ensemble_names.append(name)
        if sensor is None:
            sensor = name
        self.stimuli_names.append(sensor)

    def add_ensemble(self, name, num_neurons, **kwargs):
        """ Adds a neural ensemble or layer to the neural network with a specified number of neurons. 
        Provided that the ensemble is denoted as ``name``, then each neuron inside the ensemble is called ``name_i``, 
        where i is the index. 
        
        .. todo::
        
            For the moment, all the neurons are based on the same neuron model. In future version, we will 
            consider the addition of the neuron model as a neuron independent parameters.

        :param str name: name of the neural ensemble. 
        :param int num_neurons: number of neurons in the ensemble.
        :param dict kwargs: Encompasses all the neuron dependent parameters (that are different for each neuron model).
            For instance, if ``rate_model`` it can receive the parameters ``tau``, ``gain``, ``bias`` and ``activation``. 
        """
        self.ensemble_names.append(name)
        for name_kwarg, kwarg in kwargs.items():
            if not (isinstance(kwarg, np.ndarray) or isinstance(kwarg, list)):
               kwargs[name_kwarg] = [kwarg] * num_neurons
        for n in range(num_neurons):
            self.add_neuron(f'{name}_{n}', ensemble=name,
                **{k : val[n] for k, val in kwargs.items()})

    def add_neuron(self, name, ensemble=None, **kwargs):
        """ Adds a single neuron to the neural network.
        
        .. todo::
        
            For the moment, all the neurons are based on the same neuron model. In future version, we will 
            consider the addition of the neuron model as a neuron independent parameters.

        :param str name: name of the neuron. 
        :param str ensemble: name of the neural ensemble to which this neuron belongs. If None, then ensemble = name.
        :param dict kwargs: Encompasses all the neuron dependent parameters (that are different for each neuron model).
            For instance, if ``rate_model`` it can receive the parameters ``tau``, ``gain``, ``bias`` and ``activation``. 
        """
        if ensemble not in self.ensemble_names:
            self.ensemble_names.append(ensemble)
        if 'activation' in kwargs:
            kwargs['activation'] = Activation.from_name(kwargs['activation'])
        self.neurons.add(**kwargs)#!
        ensemble = ensemble if ensemble is not None else name
        if ensemble not in self.ensemble_names:
            self.ensemble_names.append(ensemble)
        
        extra_params = {param : getattr(self.neurons, param)[-1] 
                        for param in self.neurons.__dict__.keys()\
                        if not param.startswith('_') and param != 'dt'}
        self.graph['neurons'].update({name : 
            {**{'ensemble' : ensemble, 'idx' : len(self.neurons)-1, 'is_motor' : False}, 
            **extra_params} 
        })

    def delete_neuron(self, name):
        """ Deletes a single neuron from the neural network. It removes any possible trace of the 
        neuron within the ANN. Specially, it destroys the node and associated edges within the ANN 
        graph, removes the neuron model component and its dynamics and destroys any ensemble with 
        the neuron as only member.

        .. warning::

            After calling this method, the ``build`` method must be called before running new 
            simulations. 

        :param str name: name of the neuron to be removed.
        """
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
            if ensemble in self.motor_ensemble_names:
                self.motor_ensemble_names.remove(ensemble)
        #* Remove any synapse with the neuron as pre or post
        for syn_name, syn in [*self.graph['synapses'].items()]:
            if syn['pre'] == name or syn['post'] == name:
                self.delete_synapse(syn_name)

    def add_synapse(self, name, pre, post, weight=1., conn_prob=1., 
            trainable=True, use_seed=False, **kwargs):
        """ Adds a new group of synapses between neurons in ensembles ``pre`` and ``post``. 

        :param str name: name of the synapse group. Individual synapses in group are called name_i, where 
            i is in [0, N_c-1], being N_c the number of connections in the group.
        :param str pre: name of the existing ensemble to which pre-synaptic neurons belong.
        :param str post: name of the existing ensemble to which post-synaptic neurons belong.
        :param weight: TODO
        :param float conn_prob: synapse creation probability between ensembles pre and post. ``conn_prob=1`` means 
            fully connected and ``conn_prob=0`` means utter disconnection. In-between these two extremes, the 
            sparsity or connectivity level of the synapse group can be varied. Clearly, if there is only one 
            neuron in both ``pre`` and ``post`` ensembles, then the conn_prob makes no sense.
        :param bool trainable: flag indicating whether the created synapses can be optimized or not.
        :param int use_seed: integer value (or None) of the seed to be used when creating the synapse group, in the
            steps involving randomness. If ``seed=None`` then no seed is used at all.
        """
        if post in self.graph['inputs'] or post in self.input_ensemble_names:
            raise Exception(logging.error('An input node or ensemble cannot be '\
                'a postsynaptic neuron or ensemble.'))
        #* Check if pre is neuron or ensemble.
        if pre not in {**self.graph['inputs'], **self.graph['neurons']}:
            if pre not in self.input_ensemble_names + self.ensemble_names:
                raise Exception(logging.error('Connection presynaptic neuron or ensemble '\
                    '"{}" does not exist').format(pre))
            pre = [name for name, node in {**self.graph['inputs'], **self.graph['neurons']}.items() if node['ensemble'] == pre]
        else:
            pre = [pre]
        #* Check if post is neuron or ensemble.
        if post not in {**self.graph['inputs'], **self.graph['neurons']}:
            if post not in self.input_ensemble_names + self.ensemble_names:
                raise Exception(logging.error('Connection postsynaptic neuron or ensemble '\
                    '"{}" does not exist').format(post))
            post = [name for name, node in {**self.graph['inputs'], **self.graph['neurons']}.items() if node['ensemble'] == post]
        else:
            post = [post]

        #* Add individual Synapses
        #! REVISAR SEED
        if use_seed:
            np.random.seed(44 + len(self.graph['synapses']))
        for i, (pre_node, post_node) in enumerate(product(pre, post)):
            if np.random.random() < conn_prob:
                synapse_config = {
                    'pre' : pre_node, 
                    'post' : post_node,
                    'weight': weight if weight is not 'random' else  0.1 * np.random.randn(), 
                    'trainable' : trainable,
                    'group' : name, 'idx' : len(self.graph['synapses']), 'enabled' : True}
                if self.learning_rule is not None:
                    synapse_config.update({'learning_rule' : {p : 0. for p in ['A', 'B', 'C', 'D']}})
                if self.synapse_model == 'dynamic_synapse':
                    #! Add min and max possible delays?
                    synapse_config.update({'delay' : np.random.randint(1, 10)})
                syn_name = name+'_'+str(i) if len(pre + post) > 2 else name #  "{}_{}".format(name, i) 
                self.graph['synapses'][syn_name] = {**synapse_config, **kwargs}
        if use_seed:
            np.random.seed(None)


    def delete_synapse(self, name):
        """ Removes an existing synapse from the ANN. 
        Before calling this method, the build method must be called to 
        actually materialize the structural changes.

        :param str name: name of the synapse to be removed.
        """
        self.graph['synapses'].pop(name, None)


    def set_motor(self, ensemble_name):
        """ Sets the neurons in the given ensemble as motor or output of the network.
        
        :param str ensemble_name: name of the ensemble or layer to be set as motor. 
            The given name must already exist in the neural network.
        """
        if ensemble_name not in self.ensemble_names:
            raise Exception(logging.error('Ensemble "{}" does not exist').format(ensemble_name))
        if ensemble_name in self.motor_ensemble_names:
            return
        self.motor_ensemble_names.append(ensemble_name)
        for neuron in self.graph['neurons'].values():
            if neuron['ensemble'] == ensemble_name:
                neuron['is_motor'] = True

    def add_encoder(self, scheme, sensor, receptive_field=None, receptive_field_params={}):
        """ Adds an encoder to a sensor stimuli node. Encoders are pre-processing stages of the input 
        stimuli or state. Originally, they were conceived as processes to encode real-valued 
        signals into spike trains that can be suitably processed by Spiking Neural Networks. 
        Nonetheless, the current purpose of encoders aims to be more general, ranging from simple 
        normalization or standarization to basis expansions. 
        
        :param str scheme:
        :param str sensor: reference name of the sensor to which the encoder is attached.
        :param str receptive_field: reference name of the receptive field to be used (generally in 
            spiking neural networks) or None if no receptive field is required.
        :param dict receptive_field_params: parameters of the receptive field to be used.

        ..todo::

            Receptive field dependency injection.

        """
        raw_inputs = [inp for inp in self.graph['inputs'].values() if inp['sensor'] == sensor]
        self.encoders.add(scheme, sensor, len(raw_inputs), receptive_field=receptive_field,\
                receptive_field_params=receptive_field_params)
        if receptive_field is not None:
            if 'n_neurons' in receptive_field_params and receptive_field_params['n_neurons'] > 1:
                #* Correct the input nodes if the encoding augments their dimension.
                ensemble_name = tuple(raw_inputs)[0]['ensemble']
                for n in range(len(raw_inputs), receptive_field_params['n_neurons'] * len(raw_inputs)):
                    prev_idx = self.graph['inputs'][ensemble_name+'_'+str(n-1)]['idx']
                    for inp_node in filter(lambda x: x['idx'] >= prev_idx + 1, self.graph['inputs'].values()):
                        inp_node['idx'] += 1
                    self.graph['inputs'].update({'{}_{}'.format(ensemble_name, n) :\
                        {'ensemble' : ensemble_name, 'sensor' : sensor, 'idx': prev_idx + 1}})

    def add_decoder(self, scheme, ensemble_name, action_name, decoder_params={}):
        output_dim = self.num_ensemble_neurons(ensemble_name) # Num. Output Neurons in ensemble
        self.decoders.add(scheme, ensemble_name, action_name, output_dim, decoder_params=decoder_params)

    def reset_graph(self):
        """ Resets the neural network graph by removing all the nodes that are neither outputs nor inputs.
        It deletes all the synapses.
        """
        neuron_names = tuple(self.graph['neurons'].keys())
        synapse_names = tuple(self.graph['synapses'].keys())
        for neuron in neuron_names:
            if not self.is_motor(neuron):
                self.delete_neuron(neuron)
        for syn in synapse_names:
            self.delete_synapse(syn)
        self.build()


    def is_motor(self, neuron_name):
        """ Determines whether a neuron is a motor/output neuron or not.
        
        :param str neuron_name: name of the neuron
        
        :returns: boolean value indicating if the neuron is motor.
        """
        return self.graph['neurons'][neuron_name]['is_motor']

    def num_ensemble_neurons(self, ensemble):
        """ Gets the number of neurons composing a given ensemble/layer.

        :param str ensemble: name of the neural (non-input) ensemble.

        :returns: integer value representing the number of neurons in ensemble. 
        """
        return len(self.ensemble_indices(ensemble))

    def num_input_nodes(self, ensemble):
        """ Gets the number of nodes composing a given input layer or ensemble.

        :param str ensemble: name of the input ensemble.

        :returns: integer value representing the number of nodes in input ensemble. 
        """
        return len(self.input_ensemble_indices(ensemble))

    def ensemble_indices(self, ens_name, consider_inputs=False):
        """ Indices of the neurons of the requested ensemble. 
        
        :param str ens_name: name of the ensemble.
        :param bool consider_inputs: whether to take input nodes into account for the response or not.

        :returns: numpy array with the neuron indices. The size of the array equals the number of
            neurons in the ensemble.
        """
        if ens_name not in self.ensemble_names:
            raise Exception(logging.error('Requested ensemble "{}" does not exist.'.format(ens_name)))
        
        indices = np.array([neuron['idx'] for neuron in self.graph['neurons'].values() if neuron['ensemble'] == ens_name])
        if consider_inputs:
            indices += self.num_inputs #!
        return indices

    def input_ensemble_indices(self, input_name):
        """ Indices of the input nodes of the requested input ensemble. 
        
        :param str input_name: name of the input ensemble.

        :returns: numpy array with the input node indices.
        """
        if input_name not in self.input_ensemble_names:
            raise Exception(logging.error('Requested input ensemble "{}" does not exist.'.format(input_name)))
        indices = np.array([node['idx'] for node in self.graph['inputs'].values() if node['ensemble'] == input_name])
        return indices

    def voltage_of(self, ensemble, neuron):
        """ Returns the current voltage of the requested neuron. By voltage, we generally refer to the state of the 
        continuous time neural model.  
        
        :param str ensemble: name of the ensemble to which the neuron belongs.
        :param int neuron: index of the neuron (in [0,n-1], where n is the num. 
            of neurons in the ensemble) in the ensemble.
        """
        index = self.graph['neurons'][ensemble + '_' + str(neuron)]['idx']
        return self.neurons.voltages[index]

    @property
    def num_neurons(self):
        """ Number of neurons in the ANN (non-input). """
        return len(self.neurons)

    @property
    def num_inputs(self):
        """ Number of input nodes in the neural network. """
        return len(self.graph['inputs'])
    
    @property
    def num_motor(self):
        """ Number of motor or output neurons in the neural network. """
        return len(self.motor_neurons)
    
    @property
    def num_hidden(self):
        """ Number of hidden neurons in the neural network. """
        return self.num_neurons - self.num_motor

    @property
    def motor_neurons(self):
        """ Indices of motor neurons without considering input nodes. When addressing the 
        weight matrix or any kind of ANN adj. mat., the number of inputs MUST be added.
        """
        return np.hstack([self.ensemble_indices(motor) for motor in self.motor_ensemble_names])

    @property
    def is_spiking(self):
        """ Whether the neural network is a spiking neural network or not. """
        return issubclass(type(self.neurons), SpikingNeuronModel)

    @property
    def voltages(self):
        """ Instantaneous voltage vector (membrane voltage of each neuron membrane)
        at current simulation timestep."""
        return self.neurons.voltages
    
    @property
    def weights(self):
        """ Getter of the numpy weight matrix W. This matrix is essentially the weighted adjacency matrix 
        of the overall neural network. 
        
        :returns: a numpy array representing the (N x N+I) weighted adjacency matrix.
        """
        return self.synapses.weights

    @property
    def in_degrees(self):
        """ In-Degree vector of the neural network DiGraph."""
        return np.r_[np.zeros(self.num_inputs),  self.synapses.mask.sum(1)]

    @property
    def laplacian(self):
        """ Laplacian Matrix of the neural network DiGraph."""
        return np.diag(self.in_degrees)\
            - np.r_[np.zeros([self.num_inputs,self.num_inputs+self.num_neurons]), self.weights]

    
    def laplacian_eig(self):
        return np.linalg.eig(self.laplacian)

    