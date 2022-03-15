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
    def __init__(self, *args, dt=0.1, tau_m=2, range=4, state_dim=5, **kwargs):
        super(StatefulCommTX, self).__init__(*args, **kwargs)
        self.state_dim = state_dim
        self.range = range
        self.dt = dt
        self.tau_m = tau_m
        self.reset()
        
    def step(self, delta_state):
        # self.state += (self.dt / self.tau_m) * (delta_state - self.state)
        self.state = delta_state #np.clip(self.state, a_min=0, a_max=1)

    def reset(self):
        self.state = np.zeros(self.state_dim)# np.random.random(self.state_dim)
