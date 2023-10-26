import numpy as np
from mereli.controllers import RobotController
from mereli.neural_networks import NeuralNetwork
from mereli.register import controller_registry, controllers
from mereli.utils import flatten_dict, key_of, increase_time, RegexpDict

@controller_registry(name='neural_controller')
class NeuralController(RobotController):
    """ Neural controller class for robots.
   
    """
    def __init__(self, *args, checkpoint=None, **kwargs):
        super(NeuralController, self).__init__(*args, **kwargs)
        #self.preprocessing = Preprocessing([sens['sensor'] for sens in topology['stimuli'].values()])
        #for val in topology['stimuli'].values():
        #    val['sensor'] = val['sensor'].split('@')[0]
        self.checkpoint = checkpoint
        self.neural_network = None
        self.out_act_mapping = {}
        self.comm_state = 1 # Communication state (0 : RELAY, 1 : SEND)
        if self.checkpoint is not None:
            self.neural_network = NeuralNetwork(0.1) 
            self.neural_network.load(self.checkpoint)
            __import__('pdb').set_trace()
            

    def add_ann_from_dict(self, topology):
        return
        #! CHECK BUGS
        self.neural_network = NeuralNetwork(topology['dt'], time_scale=topology['time_scale'],\
                neuron_model=topology['neuron_model'], synapse_model=topology['synapse_model'])
        self.neural_network.build_from_dict(topology)
        self.out_act_mapping = {out_name : snn_output['actuator'] \
                    for out_name, snn_output in topology['outputs'].items()}

    def add_neural_network(self, neural_network, actuator_mapping):
        self.neural_network = neural_network
        self.out_act_mapping = actuator_mapping

    def step(self, state, reward=0.0):
        stimuli = {k : self.get_sensor_reading(k) for k in self.robot.sensors}
        if len(stimuli):
            stimuli = flatten_dict(stimuli)
        raw_actions = self.neural_network.step(stimuli, reward)

        # actions = {self.out_act_mapping[name] : ac for name, ac in raw_actions.items() \
        #            if 'IR_transmitter' not in self.out_act_mapping[name]}

        #* Map neuron output names to the corresponding actuator name
        if len(self.out_act_mapping) > 0:
            actions = {self.out_act_mapping[name] : ac for name, ac in raw_actions.items()}
        else:
            actions = raw_actions.copy()
        #* Convert all actions to numpy arrays
        for key, action in filter(lambda item: not isinstance(item[1], np.ndarray), actions.items()):
            actions[key] = np.array(action) if isinstance(action, list) else np.array([action])
        # Update actions to actuators
        for name, actuator in self.robot.actuators.items():
            if name in actions:
                actuator.action = actions[name]
        return actions
        
    def reset(self):
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

