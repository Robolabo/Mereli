import numpy as np
import pybullet as p

from .base_actuator import Actuator
from mereli.register import actuator_registry
from mereli.utils import softmax
from mereli.globals import global_states
from mereli.communication import IRFrame


@actuator_registry(name='stateful_tx')
class StatefulCommTX(Actuator):
    """
    """
    def __init__(self, *args, dt=0.05, tau_m=1, range=4, state_dim=5, **kwargs):
        super(StatefulCommTX, self).__init__(*args, **kwargs)
        self.state_dim = state_dim
        self.range = range
        self.dt = dt
        self.tau_m = tau_m
        self.reset()
        
    def step(self, delta_state):
        self.state += (self.dt / self.tau_m) * (delta_state - self.state)
        self.state = np.clip(self.state, a_min=-1, a_max=1)
        # self.state = delta_state #np.clip(self.state, a_min=0, a_max=1)

    def reset(self):
        self.state = np.random.random(self.state_dim)#np.zeros(self.state_dim)#


@actuator_registry(name='comm_tx_a')
class CommTXTypeA(Actuator):
    """
    """
    def __init__(self, *args, range=4, n=5, **kwargs):
        super(CommTXTypeA, self).__init__(*args, **kwargs)
        self.n = n
        self.range = range
        self.msg = 0
        
    def step(self, msg):
        self.msg = msg.item()
        if global_states.RENDER:
            if 'led_actuator' in self.actuator_owner.actuators:
                self.actuator_owner.actuators['led_actuator'].step(np.ones(8) * self.msg > 0.5)

    
    def reset(self):
        self.msg = 0