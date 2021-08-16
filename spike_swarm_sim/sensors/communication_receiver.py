import numpy as np
from .base_sensor import DirectionalSensor
from spike_swarm_sim.register import sensor_registry
from spike_swarm_sim.objects import Robot
from spike_swarm_sim.utils import compute_angle, angle_diff, issubclass_of_any, circle_distance
from .utils.propagation import ExpDecayPropagation
from spike_swarm_sim.communication import IRFrame

@sensor_registry(name='IR_receiver')
class IRCommunicationReceiver(DirectionalSensor):
    """ Communication Receiver based on IR technology.
    The sensor is partitioned into multiple sectors that provide measurements 
    solely of their sector coverage and about the corresponding sensing orientation.
    The reading of each sector is an message frame (python ``dict``) with the message content 
    and different context information (signal strength, RX orientation, etc.). 
    This sensor is paired with the ``IR_transmitter`` actuator, meaning that robots must 
    be equipped with both RX and TX in order to work. Additionally, the IR receiver can only 
    sense a single frame per simulation cycle (even though there could be multiple senders 
    in the coverage). Within each sector area, the selected frame is the one sent by the closest 
    robot. Besides, there is a frame selection function to decide the sector whose perceived frame 
    is currently received. The implemented selection scheme are a random selection, that picks sectors 
    randomly, and the cyclic selection, that deterministically iterates the cyclic sequence of sectors 
    to be selected. 

    **Reference Name**: ``IR_receiver``.

    :param Robot sensor_owner: robot object owning and reading from the sensor.
    :param float range: range of coverage of the sensor.
    :param float noise_sigma: std. dev. of the white noise attached to the measurement.
    :param int n_sectors: number of sectors of the sensor.
    :param int msg_length: number of components of the message. Must be the 
        same as in the Transmitter definition.
    :param int max_hops: maximum number of hops before frame discard. 
    :param str selection_scheme: As only one frame can be perceived by the sensor at each time step 
        from all directions, a frame selection scheme is required. This parameter indicates the 
        selection scheme to be employed. Possible values: ``"random"`` or ``"cyclic"``.

    :var float aperture: aperture in radians of each sector of the sensor.
    :var int current_direction: indicates the last direction/sector from where a frame was received.
        It is only useful when using cyclic selection. 
    :var ExpDecayPropagation propagation: propagation model to map ``rho`` and ``phi`` into the 
        distance estimation bounded in [0, 1]. 
    """
    def __init__(self, *args, msg_length=1, max_hops=10, selection_scheme='random', **kwargs):
        super(IRCommunicationReceiver, self).__init__(*args, **kwargs)
        self.msg_length = msg_length
        self.max_hops = max_hops
        self.selection_scheme = selection_scheme
        self.current_direction = 0 # Used by the cyclic selection
        self.propagation = ExpDecayPropagation(rho_att=0.7, phi_att=1.)

    def target_filter(self, obj):
        """ Method devoted to filtering the world objects that should be targeted for a particular sensor.
        In this case it filters out, among all neighboring objects, only the robots with the IR transmitter 
        enabled.

        :param WorldObject obj: Potential world object to be sensed.
        
        :returns: Boolean response revealing whether the obj should be explored by the sensor or not.
        """
        return issubclass(type(obj), Robot) and 'IR_transmitter' in obj.actuators

    def step_direction(self, rho, phi, direction_reading, direction, obj=None, diff_vector=None):
        """ Method that specifies the particular behavior of a directional sensor in each sensing direction.
        It must return the sensed frame of the current direction. Only objects that are within the range and 
        aperture are considered.

        :param float rho: Euclidean distance between the object sensing and the object (obj) sensed.
        :param float phi: angle between the direction of the sensor and the line passing through 
            both sensing and sensed object positions.
        :param float direction_reading: Current reading in the featured direction 
            to be potentially overwritten. In some cases such as the communication receiver it can be a ``dict``.
        :param int direction: integer refering to the current sensing direction between 0 and n_sectors - 1.
        :param WorldObject obj: Optionally, the object that is being sensed can be used.
        :param np.ndarray diff_vector: Optionally, the vector resulting from the difference 
            of the between object positions can be used. However, most of the times, 
            rho and phi are sufficient. Notice that rho=|diff_vector|.
        
        :returns: Reading of the sensor in the current direction.
        """
        condition = obj is not None\
                    and rho <= self.range\
                    and phi <= self.aperture\
                    and rho <= obj.actuators['IR_transmitter'].range\
                    and obj.actuators['IR_transmitter'].frame.enabled
        
        #* Fill initial reading with empty frame
        if direction_reading is None:
            direction_reading = self.empty_frame
        if condition:
            signal_strength = self.propagation(rho, phi)
            if signal_strength > direction_reading.signal_strength:
                # Cast a ray between my_pos and tar_pos to detect potential obstacles.
                my_pos = self.get_sensor_position(direction) + np.r_[0, 0, 0.1] #+ np.r_[0, 0, 0.017]
                tar_pos = obj.position + np.r_[0, 0, 0.07] # my_pos[2]]
                ray_res = self.sensor_owner.physics_client.ray_cast(my_pos, tar_pos)
                if ray_res == obj.id:
                    received_frame = obj.actuators['IR_transmitter'].frame
                    sending_direction = 0 #!np.argmin([angle_diff(sdir, compute_angle(diff_vector) + np.pi) for sdir in self.directions(obj.orientation)])
                    received_frame.tx_ori = self.directions(0.)[sending_direction] #!
                    received_frame.rx_ori = self.directions(0.)[direction]
                    received_frame.receiver = self.sensor_owner.id
                    received_frame.signal_strength = signal_strength
                    # direction_reading['msg'] = np.array(obj.actuators['IR_transmitter'].frame['msg'])
                    # direction_reading['priority'] = np.array([obj.actuators['IR_transmitter'].frame['priority']])
                    # direction_reading['destination'] = np.array([obj.actuators['IR_transmitter'].frame['destination']])
                    # direction_reading['sender'] = np.array([obj.id]) if obj.actuators['IR_transmitter'].frame['state'] \
                    #                             else obj.actuators['IR_transmitter'].frame['sender']
                    # direction_reading['n_hops'] = obj.actuators['IR_transmitter'].frame['n_hops']
                    # if direction_reading['n_hops'] > 1:
                    #     direction_reading['sending_direction'] = obj.actuators['IR_transmitter'].frame['sending_direction']
                    direction_reading = received_frame
        return direction_reading

    def step(self, *args, **kwargs):
        """ Steps the communication receiver. With the sensed frames from all directions it 
        applies the selection of a 
        unique frame is carried out among those frames whose sender is not the receiver.
        If no message is sensed, the measurement is an empty frame. Notice that this method extends the 
        functionality of the method ``DirectionalSensor.step`` (it does not overwrite it).
        
        :param list neighborhood: List of target neighboring entities (``WorldObject`` types) (excluding the robot 
            reading the sensor).
        
        :returns: numpy array with the measurement in each direction. In exceptional cases 
            it may return a list of python dictionaries (see ``CommunicationReceiver``).
        """
        frames = super().step(*args, **kwargs)
        # if len(np.where(np.array(frames) == 0.0)[0]):
        #     frames = [self.empty_frame for _ in range(len(frames))]
        selected_frame = {
            'cyclic' : self.cyclic_selection(frames),
            'random' : self.random_selection(frames),
        }[self.selection_scheme]    
        return selected_frame

    def random_selection(self, frames):
        """ Random selection scheme that selects a single frame from all possible frames.  
        The selection is purely random among those frames whose sender is not the robot owning the receiver. 
        Additionally, those frames with a large number of hops are discarded. The number of hops is only 
        considered if the communication state is used (see ``IR_transmitter``), so that robots are able to 
        relay messages.

        :param list frames: list of frame dicts (to be changed to frame objects) that are candidate to be selected.

        :returns: a single selected frame.  
        """
        # Discard very old frames (max 10 hops) or empty frames
        frames = [frame for frame in frames if frame.n_hops < self.max_hops and frame.sender is not None]
        if len(frames) == 0:
            frames = [self.empty_frame]
        #* Select only a direction
        signal_strengths = np.hstack([frame.signal_strength for frame in frames])
        senders = np.hstack([frame.sender for frame in frames])
        selected_direction = np.argmax(signal_strengths)
        if any(np.logical_and(senders != self.sensor_owner.id, senders != None)):
            elements = np.where(np.logical_and(senders != self.sensor_owner.id, senders != None))[0]
            selected_direction = np.random.choice(elements,)
            # frames[selected_direction]['am_i_sender'] = np.array([0])
        else:
            selected_direction = 0
            frames = [self.empty_frame]
            # frames[selected_direction]['am_i_sender'] = np.array([0])
            # frames[selected_direction]['am_i_targeted'] = np.array([0])
        return frames[selected_direction]

    def cyclic_selection(self, frames):
        """ Cyclic selection scheme that selects a single frame from all possible frames.  
        The selection is deterministic and selects frames following the cyclic sequence of sectors. 
        This means that, in an example with 4 sectors, the sequence of sectors whose frame is selected would be:

        Selection Sequence: S1 - S2 - S3 - S4 - S1 - S2 - S3 - S4- .....

        The drawback is that the sectors are selected even if there is no sender within their coverage area. In 
        these cases, the selected frame is an empty frame. 

        :param list frames: list of frame dicts (to be changed to frame objects) that are candidate to be selected.

        :returns: a single selected frame.  
        """
        selected_frame = frames[self.current_direction].get_copy()
        self.current_direction = (self.current_direction + 1) % self.n_sectors
        return selected_frame

    def reset(self):
        """ Reset method of the sensor. """
        # super().reset()
        self.current_direction = 0
    
     
    @property
    def empty_frame(self):
        """ Returns an empty frame ``dict``. """
        return IRFrame(msg_len=self.msg_length)
        # return {'signal' : np.array([0.0]), 'msg' : np.zeros(self.msg_length), \
        #         'sending_direction' : np.zeros(2), 'receiving_direction' : np.zeros(2),\
        #         'priority' : np.zeros(1), 'destination' : np.array([-1]), \
        #         'sender' : -1 * np.ones(1), 'n_hops' : 1}

@sensor_registry(name='buffered_IR_receiver')
class BufferedIRCommRX(IRCommunicationReceiver):
    def __init__(self,  *args, **kwargs):
        super(BufferedIRCommRX, self).__init__(*args, **kwargs)
        self.prev_msg = np.zeros(self.n_sectors)

    def step(self, *args, **kwargs):
        curr_dir = self.current_direction
        frame = super().step(*args, **kwargs)
        curr_msg = self.prev_msg.copy()
        curr_msg[curr_dir] = frame['msg'][0]
        frame['msg'] = curr_msg.copy()
        self.prev_msg = curr_msg
        if self.noise_sigma > 0:
            frame['msg'] += np.random.randn(len(frame['msg'])) * self.noise_sigma
            frame['signal'] += np.random.randn() * self.noise_sigma
        return frame

    def reset(self):
        super().reset()
        self.prev_msg = np.zeros(self.n_sectors)

# @sensor_registry(name='IR_receiver2')
# class IRCommunicationReceiver3D(IRCommunicationReceiver):
#     def _step_direction(self, rho, phi, direction_reading, direction, obj=None, diff_vector=None):
#         """ Step the sensor of a sector, receiving the frame messages and the underlying
#         context. For a detailed explanation of this method see DirectionalSensor._step_direction.
#         """
#         condition = obj is not None\
#                     and rho <= self.range\
#                     and rho <= obj.actuators['IR_transmitter'].range\
#                     and obj.actuators['IR_transmitter'].frame['enabled']
#         #* Fill initial reading with empty frame
#         if direction_reading is None:
#             direction_reading = self.empty_msg
#         if condition:
#             misalignments = [np.pi - circle_distance(angle, self.directions(self.sensor_owner.orientation[-1])[direction])\
#                 for angle in self.directions(obj.orientation[-1])]
#             tx_sensor = np.argmin(misalignments)
#             phi = misalignments[tx_sensor]
#             signal_strength = self.propagation(rho, phi) #! we use this rho for the moment
#             if signal_strength > direction_reading['signal']:
#                 my_pos = self.get_position(self.sensors_idx[direction]) + np.r_[0, 0, 0.02]
#                 #! Cambiar esto
#                 tar_pos = obj.sensors['IR_receiver'].get_position(self.sensors_idx[tx_sensor]) + np.r_[0, 0, 0.02]
#                 ray_res = p.rayTest(my_pos, tar_pos, physicsClientId=self.sensor_owner.physics_client)[0][0]
#                 # print(obj.id, ray_res, direction, tx_sensor)
#                 if ray_res == obj.id or ray_res == -1: # No beam collisions 
#                     sending_angle = self.directions(0.)[tx_sensor]
#                     receiving_angle = self.directions(0.)[direction]
#                     direction_reading['sending_direction'] = np.r_[np.cos(sending_angle), np.sin(sending_angle)]
#                     direction_reading['receiving_direction'] = np.r_[np.cos(receiving_angle), np.sin(receiving_angle)]
#                     direction_reading['receiving_direction'][np.abs(direction_reading['receiving_direction']) < 1e-5] = 0.0
#                     direction_reading['msg'] = np.array(obj.actuators['IR_transmitter'].frame['msg'])
#                     direction_reading['signal'] = np.array([signal_strength])
#                     direction_reading['priority'] = np.array([obj.actuators['IR_transmitter'].frame['priority']])
#                     direction_reading['destination'] = np.array([obj.actuators['IR_transmitter'].frame['destination']])
#                     direction_reading['sender'] = np.array([obj.id]) if obj.actuators['IR_transmitter'].frame['state'] \
#                                                 else obj.actuators['IR_transmitter'].frame['sender']
#                     direction_reading['n_hops'] = obj.actuators['IR_transmitter'].frame['n_hops']
#                     if direction_reading['n_hops'] > 1:
#                         direction_reading['sending_direction'] = obj.actuators['IR_transmitter'].frame['sending_direction']
#         return direction_reading

#     def reset(self):
#         joints = np.array([p.getJointInfo(self.sensor_owner.id, i, physicsClientId=self.sensor_owner.physics_client)[:2]\
#                 for i in range(p.getNumJoints(self.sensor_owner.id, physicsClientId=self.sensor_owner.physics_client))])
#         self.sensors_idx = {i : np.where(np.array(joints) == bytes('base_to_IR'+str(i), 'utf-8'))[0][0]\
#                 for i in range(self.n_sectors)}

#     def get_position(self, idx):
#         return np.array(p.getLinkState(self.sensor_owner.id, idx,\
#                 physicsClientId=self.sensor_owner.physics_client)[0])
