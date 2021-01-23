import numpy as np
from .base_sensor import DirectionalSensor
from spike_swarm_sim.register import sensor_registry
from spike_swarm_sim.utils import compute_angle, angle_diff
from .utils.propagation import ExpDecayPropagation


@sensor_registry(name='wireless_receiver')
class CommunicationReceiver(DirectionalSensor):
    """ Communication Receiver mimicking IR technology.
    ========================================================================
    - Args:
        msg_length [int] : number of components of the message. Must be the 
                    same as in the Transmitter definition.
        max_hops [int] : maximum number of hops before frame discard. 
    ========================================================================
    """
    def __init__(self, *args, msg_length=1, max_hops=10, **kwargs):
        super(CommunicationReceiver, self).__init__(*args, **kwargs)
        self.msg_length = msg_length
        self.max_hops = max_hops
        self.propagation = ExpDecayPropagation(rho_att=1/200, phi_att=1)

    def _target_filter(self, obj):
        """ Filtering of potential sender robots. """
        return type(obj).__name__ == 'Robot' and 'wireless_transmitter' in obj.actuators

    def _step_direction(self, rho, phi, direction_reading, direction, obj=None, diff_vector=None):
        """ Step the sensor of a sector, receiving the frame messages and the underlying
        context. For a detailed explanation of this method see DirectionalSensor._step_direction.
        """
        condition = obj is not None\
                    and rho <= self.range\
                    and rho <= obj.actuators['wireless_transmitter'].range\
                    and obj.actuators['wireless_transmitter'].frame['enabled']
        #* Fill initial reading with empty frame
        if direction_reading is None:
            direction_reading = self.empty_msg
        if condition:
            signal_strength = self.propagation(rho, phi)#np.exp(-((rho/100)**2) / 0.4)
            if signal_strength > direction_reading['signal']:
                sending_direction = np.argmin([angle_diff(sdir, compute_angle(diff_vector) + np.pi)\
                                for sdir in self.directions(obj.theta)])
                sending_angle = self.directions(0.)[sending_direction]
                receiving_angle = self.directions(0.)[direction]
                direction_reading['sending_direction'] = np.r_[np.cos(sending_angle), np.sin(sending_angle)]
                direction_reading['receiving_direction'] = np.r_[np.cos(receiving_angle), np.sin(receiving_angle)]
                direction_reading['receiving_direction'][np.abs(direction_reading['receiving_direction']) < 1e-5] = 0.0
                # msg = .actuators['wireless_transmitter'].msg[send_dir] #! if directional transmission
                direction_reading['msg'] = np.array(obj.actuators['wireless_transmitter'].frame['msg'])  #! if isotropic
                direction_reading['signal'] = np.array([signal_strength])
                direction_reading['priority'] = np.array([obj.actuators['wireless_transmitter'].frame['priority']])
                direction_reading['destination'] = np.array([obj.actuators['wireless_transmitter'].frame['destination']])
                direction_reading['sender'] = np.array([obj.id]) if obj.actuators['wireless_transmitter'].frame['state'] \
                                            else obj.actuators['wireless_transmitter'].frame['sender']
                direction_reading['n_hops'] = obj.actuators['wireless_transmitter'].frame['n_hops']
                if direction_reading['n_hops'] > 1:
                    direction_reading['sending_direction'] = obj.actuators['wireless_transmitter'].frame['sending_direction']
        return direction_reading

    def step(self, *args, **kwargs):
        """ Steps the communication receiver. With the sensed frames from all directions it 
        firstly discards those with more hops than a thresh. Then, the selection of a 
        unique frame is carried out stochastically among those frames whose sender is not the receiver. 
        If no message is sensed, the measurement is an empty frame.
        """
        frames = super().step(*args, **kwargs)
        if len(np.where(np.array(frames) == 0.0)[0]):
            frames = [self.empty_msg for _ in range(len(frames))]
        # Discard very old frames (max 10 hops)
        frames = [frame for frame in frames if frame['n_hops'] < self.max_hops and frame['sender'] != -1]
        if len(frames) == 0:
            frames = [self.empty_msg]

        #* Select only a direction
        signal_strengths = np.hstack([frame['signal'] for frame in frames])
        senders = np.hstack([frame['sender'].item() for frame in frames])

        #* --- MESSAGE SELECTION --- *#
        selected_direction = np.argmax(signal_strengths)
        if any(np.logical_and(senders != self.sensor_owner.id, senders != -1)):
            # candidate_frames = np.array(frames)[np.logical_and(senders != self.sensor_owner.id, senders != -1)]
            # probs = softmax(np.array([frame['n_hops'] for frame in candidate_frames]).flatten()/2)
            elements = np.where(np.logical_and(senders != self.sensor_owner.id, senders != -1))[0]
            selected_direction = np.random.choice(elements,)
            # selected_direction = elements[np.argmax(signal_strengths[elements])]
            frames[selected_direction]['am_i_sender'] = np.array([0])
        else:
            selected_direction = 0
            frames = [self.empty_msg]
            frames[selected_direction]['am_i_sender'] = np.array([0])
            frames[selected_direction]['am_i_targeted'] = np.array([0])
        return frames[selected_direction]

    def remove_duplicates(self, frames):
        """#TODO: Remove received duplicate frames."""
        raise NotImplementedError
            
    @property
    def empty_msg(self):
        """ Format of an empty message. """
        return {'signal' : np.array([0.0]), 'msg' : np.zeros(self.msg_length), \
                'sending_direction' : np.zeros(2), 'receiving_direction' : np.zeros(2),\
                'priority' : np.zeros(1), 'destination' : np.array([-1]), \
                'sender' : -1*np.ones(1), 'n_hops' : 1}
