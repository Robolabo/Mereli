.. _configuration_files:

*****************************************
Configuration Files
*****************************************



Overview
=============



World Configuration
=====================

* *engine* (str)
    Physics and render engine to be used in the simulation. Currently, it can be either 2D and 3D.

* *height* (float)
    Height in metres of the environment arena.

* *width* (float)
    Width in metres of the environment arena.

* *objects* (dict)
    Python ``dict``gathering all the entites/objects instantiated in the arena. The entities are gathered in groups that 
    share the same characteristics. Therefore, each key-value pair maps the name of the group of entities to the configuration 
    of the group members. The most important configuration fields of each entity group are the type of the entities of the 
    group, the number of instances and the initializers. Moreover, in the case of robot entities, the controller, the enabled sensors and 
    the employed actuators are also stated here. 

    Example::
    
    >>> "robotA" : {
    >>>     "type" : "robot",
    >>>     "num_instances" : 2,
    >>>     "controller" : "neural_controller",
    >>>     "sensors" : {
    >>>         "yellow_light_sensor" : {"n_sectors" : 4, "range" : 5},
    >>>         "IR_receiver" : {"n_sectors" : 4, "range" : 2, "msg_length" : 1, "selection_scheme" : "cyclic"},
    >>>         "distance_sensor3D" : {"n_sectors" : 4, "range" : 3}
    >>>     },
    >>>     "actuators" : {
    >>>         "joint_velocity_actuator" : {"joint_ids" : [0, 1], "max_velocity" : 13},
    >>>         "IR_transmitter" : {"quantize": false, "range" : 15, "msg_length":1, "K" : 2}
    >>>     },
    >>>     "initializers" : {
    >>>         "positions" : {"name" : "random_uniform",  "params" : {"low" : [-1, -1], "high" : [1, 1], "size" : 2}},
    >>>         "orientations" : {"name" : "random_uniform",  "params" : {"low":0, "high" : 6.28, "size" : 1}}
    >>>     },
    >>>     "perturbations" : {
    >>>         "stimuli_inhibition" : {"affected_robots": [1], "stimuli" : "IR_receiver", "replace_value" : 0}
    >>>     },
    >>>     "params" : {"trainable" : true}
    >>>   },
    >>>  "yellow_light" : {
    >>>      "type" : "light_source",
    >>>      "num_instances": 1,
    >>>      "controller" : null,
    >>>      "initializers" : {
    >>>          "positions" : {"name" : "fixed", "params" : {"fixed_values": [[0, 3]]}}
    >>>      },
    >>>      "params" : {"range" : 20, "color" : "yellow"}
    >>>  }

    +----------------------+-----------------------------+---------------------+-----------+----------------------------+
    | Sensor Reference     |  Sensor Class               | Parameter           |  Default  |   Description              |          
    +----------------------+-----------------------------+---------------------+-----------+----------------------------+
    |                      |                             |  n_sectors          |    4      |                            |
    | distance_sensor      |   DistanceSensor            +---------------------+-----------+----------------------------+
    |                      |                             |   range             |    2      |                            |
    +----------------------+-----------------------------+---------------------+-----------+----------------------------+
    |                      |                             |  n_sectors          |    4      |                            |
    | | IR_receiver        | | IRCommunicationReceiver   +---------------------+-----------+----------------------------+
    |                      | | BufferedIRCommRX          |   range             |    2      |                            |
    |                      |                             +---------------------+-----------+----------------------------+
    |                      |                             |   msg_length        |    1      |                            |
    |                      |                             +---------------------+-----------+----------------------------+
    |                      |                             |   max_hops          |    10     |                            |
    |                      |                             +---------------------+-----------+----------------------------+
    |                      |                             |   selection_scheme  |  "cyclic" |                            |
    +----------------------+-----------------------------+---------------------+-----------+----------------------------+
    |                      |                             |  n_sectors          |    4      |                            |
    |   light_sensor       |   LightSensor               +---------------------+-----------+----------------------------+
    |                      |                             |   range             |    2      |                            |
    +----------------------+-----------------------------+---------------------+-----------+----------------------------+
    | ground_sensor        |    GroundSensor             |                     |           |                            |
    +----------------------+-----------------------------+---------------------+-----------+----------------------------+
    | color_sensor         |     ColorSensor             |                     |           |                            |
    +----------------------+-----------------------------+---------------------+-----------+----------------------------+

    
Topology Configuration
=========================

* **dt** (float)
    Euler time step used to iteratively solve the differential equations of the neurons.

* *time_scale* (float)

* *neuron_model* (str)
    ==============  ================   ======================================
    Reference Name  Python class        Description
    ==============  ================   ======================================
    rate_model      RateModel           aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    adex            AdExModel           aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    izhikevich      IzhikevichModel     aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    lif             LIFModel            aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    exp_lif         ExpLIFModel         aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    morris_lecar    MorrisLecarModel    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    ==============  ================   ======================================

    .. note::
        ``MorrisLecarModel`` and ``LIFModel`` classes do exist in the simulator but cannot be used for the 
        moment.

* *synapse_model*  (str)
    ==============  ================   ======================================
    Reference Name  Python class        Description
    ==============  ================   ======================================
    static          RateModel           aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    adex            AdExModel           aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    ==============  ================   ======================================

* *stimuli* (dict)

    Example::
    
    >>> "stimuli": {
    >>>     "I1" : {"n" : 4, "sensor" : "yellow_light_sensor"},
    >>>     "I2" : {"n" : 4, "sensor" : "red_light_sensor"},
    >>>     "I3" : {"n" : 4, "sensor" : "distance_sensor3D"},
    >>>     "I4" : {"n" : 1, "sensor" : "task_sensor"},
    >>>     "I5" : {"n" : 4, "sensor" : "IR_receiver:msg"}
    >>> }

* *ensembles* (dict)
    Python ``dict`` defining the neuron ensembles or layers of the architecture. For the moment, 
    all the ensembles' neurons are based on the same neuron model established in the ``neuron_model`` 
    field. Nonetheless, the neurons of each ensemble can have different neuron parameters (e.g. one
    can define an ensemble of fast rate model neurons with a small :math:`\tau` and another ensemble composed 
    by slow neurons with a large :math:`\tau`). The ``dict`` maps ensemble names to ensemble parameters. There 
    is only one compulsory parameter named as ``n`` that represents the number of neurons in the ensemble. 
    The other parameters are gathered within the values of a subdict with key ``"params"``. 
    The following example shows the ensemble configuration of an architecture with rate models and four  ensembles 
    (``H1``, ``H2``, ``OUT_COMM`` and ``OUT_MOT``):

    Example::

    >>> "neuron_model" : "rate_model",
    >>> "ensembles": {
    >>>     "H1" : {"n" : 10, "params" : {"tau" : 2, "bias" : -1, "gain" : 0.75}},
    >>>     "H2" : {"n" : 5, "params" : {"tau" : 1, "bias" : -0.5}},
    >>>     "OUT_COMM" : {"n" : 1, "params" : {"tau": 0.5}},
    >>>     "OUT_MOT" : {"n" : 2, "params" : {"tau": 0.5, "activation" : "tanh"}}
    >>> }

    It can be observed in the example, the parameters of the ``"params"`` field  are optional and 
    it they are not defined, the default value is used. Moreover, the parameters shown in the example 
    are specifically used only in the case of the ``rate_model``. For other neuron models, the parameters 
    are different according to the neuron requirements. The following table gathers the possible parameters 
    of each of the available neuron models: 

    +--------------+---------------+----------------+-----------+----------------------------+
    | Neuron Model | Parameter     |  Math Notation | Default   |   Description              |          
    +--------------+---------------+----------------+-----------+----------------------------+
    |              |   tau         |  :math:`\tau`  |   1.0     | Neuron's time constant     |
    |  rate_model  +---------------+----------------+-----------+----------------------------+
    |              |   bias        |  :math:`\beta` |   0.0     |  Neuron bias or offset     |
    |              +---------------+----------------+-----------+----------------------------+
    |              |   gain        |  :math:`g`     |   1.0     | Neuron gain                |
    |              +---------------+----------------+-----------+----------------------------+
    |              | activation    |  :math:`f`     | "sigmoid" | Neuron activation function |
    +--------------+---------------+----------------+-----------+----------------------------+
    |              |   tau_w       |  :math:`\tau_w`|   1.0     | Neuron's time constant     |
    |  adex        +---------------+----------------+-----------+----------------------------+
    |              |   tau_m       |  :math:`\tau_m`|   0.0     |  Neuron bias or offset     |
    |              +---------------+----------------+-----------+----------------------------+
    |              |   V_rest      |  :math:`g`     |   1.0     | Neuron gain                |
    |              +---------------+----------------+-----------+----------------------------+
    |              |   V_reset     |  :math:`f`     | "sigmoid" | Neuron activation function |
    |              +---------------+----------------+-----------+----------------------------+
    |              |   A           |  :math:`A`     | "sigmoid" | Neuron activation function |
    |              +---------------+----------------+-----------+----------------------------+
    |              |   B           |  :math:`B`     | "sigmoid" | Neuron activation function |
    |              +---------------+----------------+-----------+----------------------------+
    |              |   theta_rest  |  :math:`A`     | "sigmoid" | Neuron activation function |
    |              +---------------+----------------+-----------+----------------------------+
    |              |   R           |  :math:`R`     | "sigmoid" | Neuron activation function |
    |              +---------------+----------------+-----------+----------------------------+
    |              |refractoriness |  :math:`A`     | "sigmoid" | Neuron activation function |
    +--------------+---------------+----------------+-----------+----------------------------+
    |              |   A           |  :math:`a`     |   0.02    | Neuron's time constant     |
    |  izhikevich  +---------------+----------------+-----------+----------------------------+
    |              |   B           |  :math:`b`     |   0.2     |  Neuron bias or offset     |
    |              +---------------+----------------+-----------+----------------------------+
    |              |   C           |  :math:`c`     |   -65     | Neuron gain                |
    |              +---------------+----------------+-----------+----------------------------+
    |              |   D           |  :math:`d`     |    8.0    | Neuron activation function |
    +--------------+---------------+----------------+-----------+----------------------------+
    |              |   tau         |  :math:`\tau`  |   20      | Neuron's time constant     |
    |  exp_lif     +---------------+----------------+-----------+----------------------------+
    |              |   R           |  :math:`R`     |   1       |  Neuron bias or offset     |
    |              +---------------+----------------+-----------+----------------------------+
    |              |   v_rest      |  :math:`c`     |  -65.0    | Neuron gain                |
    |              +---------------+----------------+-----------+----------------------------+
    |              |   time_refrac |  :math:`d`     |   10.     | Neuron activation function |
    |              +---------------+----------------+-----------+----------------------------+
    |              |   thresh      |  :math:`d`     |  -40.0    | Neuron activation function |
    +--------------+---------------+----------------+-----------+----------------------------+


* **synapses** (dict)

    Python ``dict`` defining the groups of synapses connecting the neurons and input nodes in defined in 
    ``stimuli`` and ``ensembles`` fields. Each key-value of the synapses dictionary defines the connections 
    between two ensembles (either of neurons or input nodes) and maps the synapse name to the synapse configuration. 
    The synapse configuration encompasses several parameters that structurally define the connection. Among them, 
    the ``"pre"`` and ``"post"`` parameters are compulsory and define the pre-synaptic and post-synaptic ensembles. 
    These ensemble names have to be defined either in the ``stimuli`` or ``ensembles`` fields. Moreover, notice that 
    each synapse entry can define a set of connections instead of a single one in the cases in which the number of 
    neurons or nodes in the ensembles is greater than 1. The following extract of code extends the example exposed 
    in the ``ensembles`` and ``stimuli`` explanation, showing how to connect the ensembles:

    Example::

    >>> "synapses" :  {
    >>>    "I1-H1" : {"pre":"I1", "post":"H1", "trainable":true, "p":1.0},
    >>>    "I2-H1" : {"pre":"I2", "post":"H1", "trainable":true, "p":1.0},
    >>>    "I3-H1" : {"pre":"I3", "post":"H1", "trainable":true, "p":1.0},
    >>>    "I4-H1" : {"pre":"I4", "post":"H1", "trainable":true, "p":1.0},
    >>>    "I5-H1" : {"pre":"I5", "post":"H1", "trainable":true, "p":1.0},
    >>>    "H1-H2" : {"pre":"H1","post":"H2", "trainable":true, "p":1.0},
    >>>    "H2-OUT_COMM" : {"pre":"H2","post":"OUT_COMM", "trainable":true, "p":1.0},
    >>>    "H2-OUT_MOT" : {"pre":"H2","post":"OUT_MOT", "trainable":true, "p":1.0},
    >>> }

    Additionally, the following table gathers the possible parameters to be defined in a synapse entry. 
    Notice that the ``neuroTX`` parameter is only valid when using spiking neural networks 
    (spiking neurons + dynamic synapses). It is worth mentioning the function of ``p``, which represents 
    the connection probability. Its meaning is that each synapse joining two ensembles will be created 
    with a probability ``p``. This parameter is specially important when building large unstructured 
    neural sparse meshes in which the connectivity is, say, 25%. In regular ANNs it is normally fixed 
    to 1.0 (fully connected).

    +---------------+----------------+------------------------------------------------------------+
    | **Parameter** | **Default**    |   **Description**                                          |          
    +---------------+----------------+------------------------------------------------------------+
    |   pre         |   Compulsory   | Name of the pre-synaptic neuron ensemble.                  |
    +---------------+----------------+------------------------------------------------------------+
    |   post        |   Compulsory   | Name of the post-synaptic neuron ensemble.                 |
    +---------------+----------------+------------------------------------------------------------+
    |   p           |     1.0        | Connection probability among ``pre`` and ``post`` neurons. |
    +---------------+----------------+------------------------------------------------------------+
    |   neuroTX     |  "AMPA+NDMA"   | Type of synapse neurotransmitter. Only valid if            |
    |               |                | dynamic_synapse is used. Possible values: "AMPA+NDMA",     |
    |               |                | "AMPA", "GABA" or "NDMA".                                  |
    +---------------+----------------+------------------------------------------------------------+
    | trainable     |    True        | Flag indicating if the weights of the synapses             |
    |               |                | can be optimized. Currently not used.                      |
    +---------------+----------------+------------------------------------------------------------+


* *outputs* (dict)
    The outputs field states which of the previously defined ensembles are output layers of the architecture.
    It also maps these output layers, responsible of generating actions, to the corresponding actuators.  
    The ``dict`` maps output names (do not confuse with ensemble names) with the output configuration. 
    The output configuration states the ensemble name (has to be defined) and the actuator name (has to be an 
    implemented actuator). The following example defines ``OUT_COMM`` and ``OUT_MOT`` neuron ensembles as outputs:

    Example::

    >>> "outputs" : {
    >>>    "outA" : {"ensemble" : "OUT_MOT", "actuator" : "joint_velocity_actuator", "enc": "real"},
    >>>    "outB" : {"ensemble" : "OUT_COMM", "actuator" : "IR_transmitter", "enc": "real"}
    >>> }

    +----------------------------+-----------------------------+----------------------------------------------------------------+
    | **Actuator Name**          | **Actuator Class**          |   **Description**                                              |          
    +----------------------------+-----------------------------+----------------------------------------------------------------+
    |   joint_velocity_actuator  |   JointVelocityActuator     | Actuator for the velocity control of joints.                   |
    +----------------------------+-----------------------------+----------------------------------------------------------------+
    |   joint_position_actuator  |   JointPositionActuator     | Actuator for the position control of joints.                   |
    +----------------------------+-----------------------------+----------------------------------------------------------------+
    |   IR_transmitter           |   CommunicationTransmitter  | IR-based Communication transmitter.                            |
    +----------------------------+-----------------------------+----------------------------------------------------------------+
    |   led_actuator             |   LedActuator3D/LedActuator | Actuator for controlling the LEDs of a robot.                  |
    +----------------------------+-----------------------------+----------------------------------------------------------------+
    |   grasp_actuator           |   GraspActuator             | | High level simplified actuator for grasping and dropping     |
    |                            |                             | | small lightweight objects (TODO: for the moment only cubes). |
    +----------------------------+-----------------------------+----------------------------------------------------------------+
 
    .. note::
        Just like the sensor classes, some actuator classes have a 2D and 3D implementation. This feature is temporal 
        until 2D and 3D classes are standardized.

    .. note::
        The ``IR_transmitter`` actuator name is provisional and it will eventually changed to a more descriptive name.

* *encoding* (dict)

* *decoding* (dict)
    
    Example::

    >>> "decoding" : {
    >>>     "outA" : {"scheme" : "IdentityDecoding", "params" : {"is_cat" : false}},
    >>>     "outB" : {"scheme" : "IdentityDecoding", "params" : {"is_cat" : false}}
    >>> }

* *learning_rule* (dict)


Algorithm Configuration
===========================


Examples
=============