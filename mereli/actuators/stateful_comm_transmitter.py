import numpy as np
import pybullet as p

from .base_actuator import Actuator
from mereli.register import actuator_registry
from mereli.utils import softmax, tanh 
from mereli.globals import global_states
from mereli.communication import IRFrame


@actuator_registry(name='stateful_tx')
class StatefulCommTX(Actuator):
    """
    """
    def __init__(self, *args, dt=0.1, tau_m=10, range=4, state_dim=2, **kwargs):
        super(StatefulCommTX, self).__init__(*args, **kwargs)
        self.state_dim = state_dim
        self.range = range
        self.dt = dt
        self.tau_m = tau_m
        self.reset()
        
    def step(self, control):
        # self.state += (self.dt / self.tau_m) * (control- self.state)
        self.state += (self.dt / self.tau_m) * (control)
        self.state = np.clip(self.state, a_min=-1, a_max=1)
        # self.state = delta_state #np.clip(self.state, a_min=0, a_max=1)

    def reset(self):
        # self.state = np.zeros(self.state_dim)  
        self.state = np.random.uniform(-0.05, 0.05, self.state_dim)# np.zeros(self.state_dim)  

@actuator_registry(name='ori_stateful_tx')
class OrientStatefulCommTX(Actuator):
    """
    """
    def __init__(self, *args, dt=0.1, init_state='random', init_ori='random', tau_ori=5, tau_st=5, range=4, state_dim=5, **kwargs):
        super(OrientStatefulCommTX, self).__init__(*args, **kwargs)
        self.state_dim = state_dim
        self.range = range
        self.init_state = init_state
        self.init_ori = init_ori
        self.dt = dt
        self.tau_ori = tau_ori
        self.tau_st = tau_st
        self.orientation = 0
        self.reset()
        
    def step(self, control):
        ########
        # if self.actuator_owner.id > 1:
        #     control = [0,-1]
        ##########
        delta_ori = control[0]
        speed = (control[1] + 1) / 2
        self.orientation += (self.dt / self.tau_ori) * (2*np.pi*delta_ori - self.orientation) 
        self.orientation = np.clip(self.orientation, a_min=0, a_max=2*np.pi)
        heading_ori = np.r_[np.cos(self.orientation), np.sin(self.orientation)]
        if speed > 0.5:
            self.state += (self.dt / self.tau_st) * heading_ori
        self.state = np.clip(self.state, a_min=-1, a_max=1)

    def reset(self):
        # position = {2 : [0.2, 0.2], 3 : [-0.2, -0.2], 4 : [-0.2, 0.2]}.get(self.actuator_owner.id)
        # if position is not None:
        #     position = np.array(position)
        #     self.state = position
        # else:
        if self.init_state == 'random':
            self.state = np.random.uniform(-0.05, 0.05, self.state_dim)
        elif self.init_state == 'zero':
            self.state = np.array([0.0, 0.0])
        else:
            self.state = np.array(self.init_state).astype(float)
        if self.init_ori == 'random':
            self.orientation = np.random.uniform(0, 2*np.pi)
        elif self.init_ori == 'zero':
            self.orientation = 0.0
        else:
            self.orientation = self.init_ori

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
