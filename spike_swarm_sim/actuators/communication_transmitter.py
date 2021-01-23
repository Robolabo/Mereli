import numpy as np
from spike_swarm_sim.register import actuator_registry
from spike_swarm_sim.utils import softmax

@actuator_registry(name='wireless_transmitter')
class CommunicationTransmitter:
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
    def __init__(self, range=120, msg_length=1, quantize=True):
        self.channel = 0
        self.msg_length = msg_length
        self.range = range
        self.quantize = quantize
        if self.quantize:
            self.clusters = [centroid for centroid in \
                    zip(*map(lambda v: v.flatten(),\
                    np.meshgrid(*[np.linspace(0, 1, 4) for _ in np.arange(self.msg_length)])))]
            self.clusters = np.array(self.clusters)
        self.frame = None
        self.reset()
        
    def step(self, action):
        #* Select cluster using softmax on distances to clusters
        if self.quantize:
            dists = np.linalg.norm(action['msg'] - self.clusters, axis=1)
            # Max. dist in hypercube is sqrt(dim(x))
            max_distance = np.sqrt(len(action['msg']))
            probs = softmax(1 - dists / max_distance, tau=1e-2)
            cluster = np.random.choice(range(len(self.clusters)), p=probs)
            action['msg'] = self.clusters[cluster]
        self.frame['msg'] = action['msg']
        self.frame['priority'] = action['priority']
        self.frame['sender'] = action['sender']
        self.frame['destination'] = action['destination'] \
                if 'destination' in action.keys() else 0
        self.frame['enabled'] = action['enabled']\
                if 'enabled' in action.keys() else True
        self.frame['n_hops'] = action['n_hops']
        self.frame['state'] = action['state']
        self.frame['sending_direction'] = action['sending_direction']

    def reset(self):
        self.frame = {
            'msg' : [0 for _ in range(self.msg_length)],
            'enabled' : True,
            'state' : 1,
            'n_hops' : 1,
            'sender' : -1,
            'destination' : 0,
            'source' : 0,
            'priority' : 0,
            'sending_direction' : 0,
        }