import numpy as np
from mereli.controllers import RobotController
from mereli.neural_networks import NeuralNetwork
from mereli.register import controller_registry, controllers
from mereli.utils import flatten_dict, key_of, increase_time, RegexpDict

@controller_registry(name='neural_controller')
class NeuralController(RobotController):
    """ Neural controller class for robots.
   
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

@controller_registry(name='neural_orchestrator')
class NeuralOrchestrator(NeuralController):
    def __init__(self, *args, behaviors=[], **kwargs):
        super(NeuralOrchestrator, self).__init__(*args, **kwargs)
        self.behaviors = [controllers[beh](*args,  **kwargs) for beh in behaviors]
        self.init_state = self.behaviors[0]
        self.state = self.init_state

    def step(self, state, reward=0.0):
        self.state.step(state)
        actions = super().step(state, reward=0.0)
        if "BEHAVIOR" in actions:
            self.state = self.behaviors[actions["BEHAVIOR"].item()]
            actions.pop("BEHAVIOR")
        # print(f'Executing {type(self.state).__name__}')
        return {**self.state.step(state), **actions}

    def reset(self):
        super().reset()
        self.state = self.init_state

