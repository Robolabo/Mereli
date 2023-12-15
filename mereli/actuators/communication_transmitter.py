import numpy as np
import pybullet as p

from .base_actuator import Actuator
from mereli.register import actuator_registry
from mereli.utils import softmax
from mereli.globals import global_states
from mereli.communication import IRFrame


@actuator_registry(name='IRCommTX')
class CommunicationTransmitter(Actuator):
    """ Communication transmitter actuator. It isotropically transmits a 
    frame with a message and its context. The propagation simulation is 
    implemented at the receiver side, this class only updates the transmitted 
    frame of each robot.

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
        self.link_ids = []
        self.range = range
        self.frame = None
        
    def step(self):
        #* Select cluster using softmax on distances to clusters
        # self.frame = tx_frame
        params = {'frame' : self.action}
        for link_id in self.link_ids:
            self.physics_client.set_link_params(self.robot.id, link_id, **params)
        # if global_states.RENDER and 'led_actuator' in self.actuator_owner.actuators:
        #     led = np.round(4 * tx_frame.msg) / 4
        #     self.actuator_owner.actuators['led_actuator'].step(led * np.ones(8))
       

    def reset(self):
        self.frame = IRFrame(msg_len=self.msg_length)
        self.frame.sender = self.actuator_owner.id
        self.frame.original_sender = self.actuator_owner.id
        self.link_ids = [self.physics_client.physical_sensors['distance_sensor'][i]['idx'] for i in range(8)]
