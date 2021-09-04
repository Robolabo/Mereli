
import copy
import numpy as np
from spike_swarm_sim.register import communication_registry
from spike_swarm_sim.utils.activations import softmax

COMM_STATES = {
    'RELAY' : 0,
    'BCAST' : 1,
}


class IRFrame:
    """ Frame class used by IR_receiver and IR_transmitter of the robots to 
    communicate. 

    :param int msg_len: dimension of the message content vector.

    :var np.ndarray msg: message vector of dim ``msg_len``.
    :var float signal_strength: signal strength to be filled when the message is 
        received. If the message has not been received by any robot yet, then its
        value equals to 0.0.
    :var bool enabled: flag indicating if the message is enabled and can be processed 
        by other robots.
    :var float rx_ori: relative orientation (in radians) of the sector from where the 
        sender robot emitted the frame. The angle is relative to the heading orientation 
        of the sender robot.
    :var float tx_ori: relative orientation (in radians) of the sector from where the 
        frame was received. The angle is relative to the heading orientation of the robot.      
    :var int sender: id of the last robot who sent the frame. In a multi-hop communication of 
        relayed messages it still equals to the last robot who last relayed the frame (see
        ``original_sender`` for the attribute corresponding to original robot generating the frame).
    :var int receiver: id of the robot who last received the frame. Every time that an IR_receiver 
        captures a frame, it fills its id within this field. If no robot has received the frame yet 
        or a robot in process of transmitting it, then its value equals to ``None``.
    :var int original_sender: id of original sender of the frame. It is useful to keep track 
        of the original robot who elaborated the message when the frame has been relayed by 
        multiple robots (``RELAY`` mode).    
    :var int destination: id of the final destination (robot) of the frame. The destination is 
        an intentional recipient of the message that can be multiple nodes away in the swarm. 
        Do not confuse with ``receiver``, which corresponds to the robot receiving the frame in 
        a single hop communication (unintentional). This attribute is not used yet in any comm. 
        system
    :var int priority: priority of the frame (not used yet).
    :var int n_hops: number of hops since the frame was originally sent. 
        Only meaningful if robots can relay message (``RELAY`` mode of comm. systems.).
    """
    def __init__(self, msg_len=1):
        self.msg_len = msg_len
        self.msg = np.zeros(msg_len).astype(float)
        self.signal_strength = 0.0
        self.enabled = True
        self.tx_ori = None
        self.rx_ori = None
        self.sender = None
        self.receiver = None
        self.destination = None
        self.original_sender = None
        self.priority = None
        self.n_hops = 0

    @property
    def encoded_tx_ori(self):
        """ Transform the attribute ``tx_ori`` in radians in the unit circle encoded vector
        
        .. math::

            `(\cos(\theta_{tx}),\, \sin(\theta_{tx}) )^T`

        If the angle ``tx_ori`` was ``None``, then it returns the zero vector.

        :returns: numpy array with the encoded angle.
        """
        if self.tx_ori is None:
            return np.array([0., 0.])
        enc_ori = np.r_[np.cos(self.tx_ori), np.sin(self.tx_ori)].round(2)
        enc_ori[np.abs(enc_ori) < 1e-5] = 0.0
        return enc_ori

    @property
    def encoded_rx_ori(self):
        """ Transform the attribute ``rx_ori`` in radians in the unit circle encoded vector
        
        .. math::

            `(\cos(\theta_{rx}),\, \sin(\theta_{rx}) )^T`

        If the angle ``rx_ori`` was ``None``, then it returns the zero vector.

        :returns
        """
        if self.rx_ori is None:
            return np.array([0., 0.])
        enc_ori = np.r_[np.cos(self.rx_ori), np.sin(self.rx_ori)].round(2)
        enc_ori[np.abs(enc_ori) < 1e-5] = 0.0
        return enc_ori

    @property
    def as_dict(self):
        """ Return the frame as a python ``dict``."""
        return {'msg' : self.msg, 'signal' : self.signal_strength, 
            'receiving_direction' : self.encoded_rx_ori, 'sending_direction' : self.encoded_tx_ori, 
            'raw_tx_angle' : self.tx_ori, 'raw_rx_angle' : self.rx_ori}

    def increase_hops(self):
        """ Increase the number of hops of the frame."""
        self.n_hops += 1

    def get_copy(self):
        """ Get a copy of the frame object.
        
        :returns: copied ``IRFrame`` object.
        """
        return copy.deepcopy(self)



@communication_registry(name='IR_comm')
class IRCommunication:
    """ IR communication system class. It provides the communication logic decoupled from the 
    transmitter and the receiver. It has two main methods ``step_pre`` and ``step_post`` that 
    are respectively applied to the states and actions before and after the controller execution. 
    Its main roles are to convert the ``IRFrame`` object to a ``dict`` that can be processed by the
    controllers and to elaborate the new frame to be transmitted using the controller actions (message, 
    new communication state, ...). It also can quantize the message to be transmitted to a set of 
    fixed symbols or centroids.

    .. note::
        This class is not responsible for elaborating the message or changing the communication state. 
        These operations must be implemented in the controller to be used.

    :param bool quantize: whether the transmitted message is quantized or not.
    :param int K: if ``quantize=True``, it indicates the number of equispaced symbols of the quantization 
        in each message dimension. Provided that M is the message length, the total number of symbols 
        in the quantization dictionary is K^M.  
    
    :var str comm_state: state of the communication. Currently, it can be ``BCAST`` or ``RELAY``.
    :var str tx_name: reference name of the communication transmitter linked to the communication system.
    :var str rx_name: reference name of the communication receiver linked to the communication system.
    :var int owner_id: identifier of the robot owning and executing the comm. sys.
    :var IRFrame rx_frame: last frame received during the simulation. If no frame was received yet then its
        value is ``None``.
    :var IRFrame tx_frame: last frame transmitted during the simulation. If no frame was received yet then its
        value is ``None``.
    :var dict registry: registry of the sequence of communication states ``registry['comm_states']`` and 
        of frames ``registry['frames']``. Registry not implemented yet.
    :var np.ndarray centroids: if ``quantize=True`` it stores the centroid or symbol dictionary of the msg quantization.
    """
    def __init__(self, quantize=False, K=3):
        self.quantize = quantize
        self.comm_state = 'BCAST'
        self.tx_name = 'IR_transmitter'
        self.rx_name = 'IR_receiver'
        self.owner_id = None
        self.rx_frame = None
        self.tx_frame = None
        self.registry = {'comm_states' : [], 'frames' : []}
        #* If message quantization is enabled, create centroids
        self.centroids = None
        if self.quantize:
            self.K = K
            self.centroids = [centroid for centroid in zip(*map(lambda v: v.flatten(),\
                    np.meshgrid(*[np.linspace(0, 1, self.K) for _ in np.arange(self.msg_length)])))]
            self.centroids = np.array(self.centroids)

    def step_pre(self, frame):
        """ Communication system logic to be applied before the controller execution. 
        
        :param IRFrame frame: IR frame received by the IR_receiver.

        :returns: frame ``dict`` to be fed to the controller. 
        """
        frame.increase_hops()
        self.rx_frame = frame #! copy or direct?
        return {**frame.as_dict, **{'state' : self.comm_state_code}}

    def step_post(self, actions):
        """ Communication system logic to be applied after the controller execution. 

        :param dict actions: ``dict`` with all the actions (keys are actuator names and values are the 
            actions). The action of the ``IR_transmitter`` must be included.

        :returns: frame ``dict`` with the modified actions. The action corresponding to the ``IR_transmitter`` 
            is now an IRFrame object to be transmitted.
        """
        if self.tx_name in actions:
            new_msg = np.array(actions[self.tx_name])
            if self.quantize:
                new_msg = self.quantize_func(new_msg)
            #* Update communication state according to controller
            self.comm_state = actions.get(self.tx_name + ':state', self.comm_state)
            #* Build the frame to be transmitted
            tx_frame = IRFrame(msg_len=self.rx_frame.msg_len)
            tx_frame.priority = actions.get(self.tx_name + ':priority', self.rx_frame.priority)
            tx_frame.enabled = actions.get(self.tx_name + ':enabled', True)
            tx_frame.msg = {
                'BCAST' : new_msg,
                'RELAY' : self.rx_frame.msg.copy()
            }.get(self.comm_state, new_msg)
            tx_frame.sender = self.owner_id
            tx_frame.original_sender = {
                'BCAST' : self.owner_id,
                'RELAY' : self.rx_frame.sender
            }.get(self.comm_state, new_msg) 
            if self.comm_state == 'BCAST':
                tx_frame.n_hops = self.rx_frame.n_hops
            self.tx_frame = tx_frame.get_copy()
            actions[self.tx_name] = tx_frame
        return actions

    def quantize_func(self, msg, tau=0.1):
        distances = np.linalg.norm(msg - self.centroids, axis=1)
        # Max. dist in hypercube is sqrt(dim(x))
        max_distance = np.sqrt(len(msg))
        probs = softmax(1 - distances / max_distance, tau=tau)
        symbol = np.random.choice(range(len(self.centroids)), p=probs)
        return self.centroids[symbol]

    @property
    def comm_state_code(self):
        """ Get the code of the current communication state. 
        The codes are:

        .. code-block:: python
        
            COMM_STATES = {
                'RELAY' : 0,
                'BCAST' : 1,
            }
        
        :returns: ``int`` with the code of the communication state. 
        """
        return COMM_STATES[self.comm_state]

    def set_owner(self, idx):
        """ Set the identifier of the robot owning and executing the comm. sys.
        
        :param int idx: identifier of the robot.
        """
        self.owner_id = idx

    def reset(self):
        """ Reset method of the communication state. """
        self.comm_state = 'BCAST'
        self.rx_frame = None
        self.tx_frame = None
        self.registry = {'comm_states' : [], 'frames' : []}