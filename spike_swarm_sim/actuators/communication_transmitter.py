import numpy as np
import pybullet as p

from .base_actuator import Actuator
from spike_swarm_sim.register import actuator_registry
from spike_swarm_sim.utils import softmax
from spike_swarm_sim.globals import global_states
from spike_swarm_sim.communication import IRFrame


@actuator_registry(name='IR_transmitter')
class CommunicationTransmitter(Actuator):
    """ Communication transmitter actuator. It isotropically transmits a 
    frame with a message and its context. The propagation simulation is 
    implemented at the receiver side, this class only updates the transmitted 
    frame of each robot.
    =========================================================================
    - Params:
        range [float] : maximum distance of message reception, in centimeters.
        msg_length [int] : number of components of the message.
        quantize [bool] : whether to quantize the message to a set of possible 
                symbols or not.
    """
    def __init__(self, *args, range=2, msg_length=1, **kwargs):
        super(CommunicationTransmitter, self).__init__(*args, **kwargs)
        self.channel = 0
        self.msg_length = msg_length
        self.range = range
        self.frame = None
        self.reset()
        
    def step(self, tx_frame):
        #* Select cluster using softmax on distances to clusters
        self.frame = tx_frame
        # if global_states.RENDER:
        #     color = [self.frame.msg[0], 0, 0]
        #     self.actuator_owner.physics_client.set_color(self.actuator_owner.id, 3, color, opacity=0.7)

    def reset(self):
        self.frame = IRFrame(msg_len=self.msg_length)
        self.frame.sender = self.actuator_owner.id
        self.frame.original_sender = self.actuator_owner.id