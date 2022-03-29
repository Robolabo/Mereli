import numpy as np
from mereli.controllers import RobotController
from mereli.neural_networks import NeuralNetwork
from mereli.register import controller_registry
from mereli.utils import flatten_dict, key_of, increase_time, RegexpDict

@controller_registry(name='neural_controller')
class NeuralController(RobotController):
    """ Neural controller class for robots.
    ==================================================================================
    - Params:
        topology [dict]: configuration dict of the ANN topology.
    - Attributes:
        neural_network [NeuralNetwork] : neural network instance to 
                process stimuli and generate actions.
        out_act_mapping [dict] : map between ouput names and actuator names.
        comm_state [int] : State or mode of the communication (if any).
                The currently implemented states are 0 (RELAY) and 1 (SEND/BROADCAST).
    ===================================================================================
    """
    def __init__(self, *args, **kwargs):
        super(NeuralController, self).__init__(*args, **kwargs)
        #self.preprocessing = Preprocessing([sens['sensor'] for sens in topology['stimuli'].values()])
        #for val in topology['stimuli'].values():
        #    val['sensor'] = val['sensor'].split('@')[0]
        self.neural_network = None
        self.out_act_mapping = {}
        self.comm_state = 1 # Communication state (0 : RELAY, 1 : SEND)
        self.t = 0

    def add_ann_from_dict(self, topology):
        #! CHECK BUGS
        self.neural_network = NeuralNetwork(topology['dt'], time_scale=topology['time_scale'],\
                neuron_model=topology['neuron_model'], synapse_model=topology['synapse_model'])
        self.neural_network.build_from_dict(topology)
        self.out_act_mapping = {out_name : snn_output['actuator'] \
                    for out_name, snn_output in topology['outputs'].items()}

    def add_neural_network(self, neural_network, actuator_mapping):
        self.neural_network = neural_network
        self.out_act_mapping = actuator_mapping

    @increase_time
    def step(self, state, reward=0.0):
        if len(state):
            state = flatten_dict(state)
        raw_actions = self.neural_network.step(state, reward)

        # actions = {self.out_act_mapping[name] : ac for name, ac in raw_actions.items() \
        #            if 'IR_transmitter' not in self.out_act_mapping[name]}

        #* Map neuron output names to the corresponding actuator name
        actions = {self.out_act_mapping[name] : ac for name, ac in raw_actions.items()}
        # for action in actions:
        #     if ':' in action:
        #         action_split = action.split(':')
        #         if action_split[0] in actions:
        #             actions[action_split[0]] = {'value' : actions[action_split[0]], action_split[1] : actions[action]}
        
        #* Convert all actions to numpy arrays
        for key, action in filter(lambda item: not isinstance(item[1], np.ndarray), actions.items()):
            actions[key] = np.array(action) if isinstance(action, list) else np.array([action])
        if 'wheel_actuator' in actions.keys():
            if type(actions['wheel_actuator']) in [int, bool]:
                actions['wheel_actuator'] = np.array(([0., 0.], [.5, -.5], [-.5, .5])[actions['wheel_actuator']])
            elif type(actions['wheel_actuator']) in [list, np.ndarray] and len(actions['wheel_actuator']) > 1:
               actions['wheel_actuator'] = np.array(actions['wheel_actuator'])
            else:
                actions['wheel_actuator'] = np.array((actions['wheel_actuator'][0], -actions['wheel_actuator'][0])).flatten()
        return actions
    
    def reset(self):
        self.t = 0
        self.comm_state = 1 #* role of agent in communication, 0 is relay mode and 1 is send mode.
        if self.neural_network is not None:
            self.neural_network.reset()





# class Postprocessing:
#     pass

# import copy

# #! PROV: MOVER A OTRO FICHERO
# class Preprocessing:
#     def __init__(self, sensors):
#         self.operations = RegexpDict({
#             'max' : lambda x, key=None: np.array(max(x)),
#             'mean' : lambda x, key=None: np.mean(x),
#             'min' : lambda x, key=None: min(x),
#             'index=[0-9]{1,2}$' : lambda x, key: x[int(key.split('=')[1])],
#             'index=([0-9]{1,2}:[0-9]{1,2})' : lambda x, key: np.array([x[i] for i in range(*map(int, key.split('=')[1].split(':')))]),
#             #'index=(([0-9]{1,2}),){1,20}[0-9]{1,2}$' : lambda x, key: np.array([x[i] for i in key.split(',')])
#         })
#         self.sensors = {sens.split('@')[0] : sens for sens in sensors}
#         # self.sensor_preproc = {sens : self.operations.get(op, lambda x: x)\
#         #         for sens, op in map(lambda z: z.split('@'), filter(lambda x: '@' in x, copy(sensors)))}
#         self.sensor_preproc = copy.deepcopy({sens.split('@')[0] : self.operations.get(sens.split('@')[1], None)
#         if '@' in sens else None for sens in sensors})
#     def __call__(self, stimuli):
#         for key, stim in stimuli.items():
#             # if key == 'yellow_light_sensor' : import pdb; pdb.set_trace()
#             if self.sensors.get(key) and '@' in self.sensors.get(key):
#                 ope = self.operations[self.sensors[key].split('@')[1]]
#                 stimuli[key] = ope(stim)
#         return stimuli        

