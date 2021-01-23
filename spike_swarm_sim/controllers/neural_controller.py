import numpy as np
from spike_swarm_sim.controllers import RobotController
from spike_swarm_sim.neural_networks import NeuralNetwork
from spike_swarm_sim.register import controller_registry
from spike_swarm_sim.utils import flatten_dict, key_of, increase_time

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
    def __init__(self, topology, *args, **kwargs):
        super(NeuralController, self).__init__(*args, **kwargs)
        self.neural_network = NeuralNetwork(topology)
        self.out_act_mapping = {out_name : snn_output['actuator'] \
                    for out_name, snn_output in topology['outputs'].items()}
        self.comm_state = 1 # Communication state (0 : RELAY, 1 : SEND)
        self.t = 0

    @increase_time
    def step(self, state, reward=0.0):
        if len(state):
            state = flatten_dict(state)
        state['wireless_receiver:state'] = np.array([self.comm_state])
        raw_actions = self.neural_network.step(state, reward)
        actions = {self.out_act_mapping[name] : ac for name, ac in raw_actions.items() \
                   if 'wireless_transmitter' not in self.out_act_mapping[name]}
        if 'wireless_transmitter' in self.out_act_mapping.values():
            msg = raw_actions[key_of(self.out_act_mapping, 'wireless_transmitter')]
            is_response = 1
            if 'wireless_transmitter:priority' in self.out_act_mapping.values():
                is_response = raw_actions[key_of(self.out_act_mapping, 'wireless_transmitter:priority')]
            if 'wireless_transmitter:state' in self.out_act_mapping.values() and self.t > 10:
                self.comm_state = raw_actions[key_of(self.out_act_mapping, 'wireless_transmitter:state')]

            #* relay or bcast
            msg = msg if self.comm_state else state['wireless_receiver:msg'].copy()
            n_hops = state['wireless_receiver:n_hops'] + 1 if not self.comm_state else 1
            destination = state['wireless_receiver:sender'].item() if is_response and state['wireless_receiver:sender'] > 0 else 0
            actions['wireless_transmitter'] = {'destination': destination, 'sender' : state['wireless_receiver:sender'], 'priority':is_response, 'en' : 1, \
                    'n_hops': n_hops, 'state' : self.comm_state, 'msg' : msg, 'sending_direction' : state['wireless_receiver:sending_direction']}
            
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