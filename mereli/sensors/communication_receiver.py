from numpy.lib.function_base import select
from mereli.utils.utils import remove_duplicates
import numpy as np
from .base_sensor import DirectionalSensor
from mereli.register import sensor_registry
from mereli.objects import Robot
from mereli.utils import compute_angle, angle_diff, issubclass_of_any, circle_distance
from .utils.propagation import ExpDecayPropagation
from mereli.communication import IRFrame, EmptyFrame
from mereli.globals import global_states




@sensor_registry(name='IRCommRX')
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
    :param int n_sectors: number of sectors of the sensor.  :param int msg_length: number of components of the message. Must be the same as in the Transmitter definition.
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

        self.aperture = 0.55 #1.5 * np.pi / self.n_sectors
        dist_coef = -np.log(0.01)/(self.range)
        phi_coef = 0  # -np.log(0.01)/ 2 * self.aperture 
        self.propagation = ExpDecayPropagation(rho_att=dist_coef, phi_att=phi_coef) # DS=

        self.contact_points = None
        self.reading = np.zeros(8)
        self.t = 0


    def step(self):
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
        #* Ids of all robots (used later)
        # robot_ids = [ent.id for ent in neighborhood if self.target_filter(ent)]
        frames = [self.empty_frame for _ in range(self.n_sectors)]
        signal_strengths = np.zeros(8)
        g_ids = [self.sensor_owner.physics_client.physical_sensors['distance_sensor'][i]['ghost_link_idx'] for i in range(8)]
        self.contact_points = self.sensor_owner.physics_client.get_contact_points(self.sensor_owner.id, ghost_ids=g_ids)
        oris = self.directions(self.sensor_owner.orientation[-1])
        for i in range(8):
            ori = oris[i]
            tar_ents = [pt[0] for pt in self.contact_points if pt[1] == g_ids[i]\
                and pt[0] not in self.sensor_owner.physics_client.luminous_objects and pt[0] != 0]
            signal_strength_ds = 0.0
            if len(tar_ents) > 0:
                origin = self.get_sensor_position(i)
                ray_angles = np.linspace(-self.aperture/2, self.aperture/2, 3)
                ray_dests = [self.range*np.r_[np.cos(ang), np.sin(ang), -0.05] + origin for ang in ori + ray_angles]
                # for o, d in zip([origin]*len(ray_dests), ray_dests):
                #     import pybullet as p
                #     p.addUserDebugLine(o, d, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0.5)
                ray_res, ray_positions = self.sensor_owner.physics_client.ray_cast([origin]*len(ray_dests), ray_dests)
                ray_res = np.array(ray_res)
                if any(ray_res != -1):
                    rhos, phis = zip(*[(np.linalg.norm(pos - origin), phi) for idx, pos, phi in zip(ray_res, ray_positions, ray_angles) if idx != -1])
                    #* Seize the sensor execution and compute the distance sensor reading as well.
                    signal_strength_ds = np.mean([self.propagation(rho, phi) for rho, phi in zip(rhos, phis)])
                    #* Only if there are robots.
                    robot_ids = self.physics_client.robot_ids
                    comm_conditions = [True if idx in robot_ids else False for idx in ray_res if idx != -1]
                    if any(comm_conditions):
                        rhos = np.array(rhos)[comm_conditions]
                        ray_res = ray_res[ray_res != -1]
                        # Compute the sector from where the frame was transmitted by the sender.
                        tmp_idx = np.argmin(rhos)
                        tx_idx = ray_res[np.where(comm_conditions)[0][tmp_idx]]
                        if tx_idx not in self.physics_client.link_parameters:
                            continue
                        rx_sensor_pos = self.physics_client.get_sensor_position(self.sensor_owner.id, sensor_name='IRCommRX', sector=0)[0]
                        tx_sector_dists = [np.linalg.norm(rx_sensor_pos - self.physics_client.get_sensor_position(tx_idx, sensor_name='IRCommRX', sector=i)[0]) for i in range(8)]
                        tx_sector = np.argmin(tx_sector_dists)
                        # comm_rho = 

                        comm_rho = tx_sector_dists[tx_sector]
                        # comm_phi = angle_diff(compute_angle(tx_sensor_pos[:2] - rx_sensor_pos[:2]), ori)

                        #* Compute both signal strength of distance sensor and communication reception (to optimize simulation).
                        # signal_strength_comm = self.propagation(comm_rho, comm_phi)
                        sector_id = self.physics_client.physical_sensors['distance_sensor'][tx_sector]['idx']
                        rx_frame = self.physics_client.link_parameters[tx_idx][sector_id]['frame']
                        rx_frame.rx_sector = i
                        rx_frame.tx_sector = tx_sector
                        rx_frame.tx_ori = self.directions(0.)[tx_sector] 
                        rx_frame.rx_ori = self.directions(0.)[i] 
                        rx_frame.signal_strength = signal_strength_ds
                        rx_frame.receiver =  self.sensor_owner.id
                        frames[i] = rx_frame
                        # print('Actual frame detected!')
            signal_strengths[i] += signal_strength_ds
        # self.reading += (0.2) * (np.array(signal_strengths) - self.reading)
        # if any(not fr.is_null for fr in frames):
        #     __import__('pdb').set_trace()
        #* Select a single frame from all possible sectors according to the given selection scheme.
        selected_frame = {
            'cyclic' : self.cyclic_selection(frames),
            'random' : self.random_selection(frames),
            'full' : self.full_selection(frames)
        }[self.selection_scheme]

        self.reading = selected_frame
        # If enabled, update also the distance_sensor
        # OJO FULL
        if 'distance_sensor' in self.sensor_owner.sensors:
            self.sensor_owner['distance_sensor'].reading = np.array(signal_strengths)

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
        # frames = [frame if frame.n_hops < self.max_hops and frame.sender is not None else self.empty_frame for frame in frames ]
        #* Select only a direction
        signal_strengths = np.zeros(len(frames)).astype(float)
        null_frames = [False] * len(frames)
        n_hops = np.zeros_like(signal_strengths)
        senders = np.zeros_like(n_hops)
        for i in range(len(frames)):
            fr = frames[i]
            signal_strengths[i] = fr.signal_strength
            null_frames[i] = fr.is_null
            n_hops[i] = fr.n_hops
            senders[i] = fr.sender
        if all(null_frames): 
            return frames[0]
        candidates = np.logical_and(~np.array(null_frames), n_hops < 10, senders != self.sensor_owner.id)
        sel_idx= np.argmax(signal_strengths[candidates])
        selected = np.r_[frames][candidates][sel_idx] 
        return selected 

        # signal_strengths = np.hstack([frame.signal_strength for frame in frames])
        # senders = np.hstack([frame.sender for frame in frames])
        # selected_direction = np.argmax(signal_strengths)
        # selection_condition = np.logical_and(senders != self.sensor_owner.id, senders != None, signal_strengths > 0)
        # if any(selection_condition):
        #     elements = np.where(selection_condition)[0]
        #     selected_direction = np.random.choice(elements,)
        # else:
        #     selected_direction = 0
        #     frames = [self.empty_frame]
        # return frames[selected_direction]

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


    def full_selection(self, frames):
        """ Selects the message of all directions at once (no selection). Thus, the resulting message has the dimension of 
        n_sectors * msg_length. 

        :param list frames: list of frame dicts (to be changed to frame objects) that are candidate to be selected.

        :returns: a single selected frame.
        """
        return [fr for fr in frames if not fr.is_null]
        # selected_frame = IRFrame(msg_len=frames[0].msg_len)
        # selected_frame.msg = np.hstack([fr.msg for fr in frames])
        # selected_frame.signal_strength = np.hstack([fr.signal_strength for fr in frames])
        # return selected_frame


    def reset(self):
        """ Reset method of the sensor. """
        # super().reset()
        self.reading = np.zeros(8)
        self.contact_points = None  
        self.t = 0
        self.current_direction = 0


    @property
    def empty_frame(self):
        """ Returns an empty frame ``dict``. """
        return EmptyFrame(msg_len=self.msg_length)

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
                my_pos = self.get_sensor_position(direction)# + np.r_[0, 0, 0.1] #+ np.r_[0, 0, 0.017]
                tx_positions = np.vstack([self.sensor_owner.physics_client.get_link_state(obj.id, i)[0]\
                            for i in range(self.n_sectors)])
                tx_sector = np.argmin(np.linalg.norm(tx_positions - my_pos, axis=1))
                ray_res = self.sensor_owner.physics_client.ray_cast(my_pos, tx_positions[tx_sector])
                # import pybullet as p
                # p.addUserDebugLine(my_pos, tx_positions[tx_sector], lineColorRGB=[1, 0, 0], lineWidth=2.0, lifeTime=0.25)
                if ray_res == obj.id:
                    received_frame = obj.actuators['IR_transmitter'].frame
                    sending_direction = tx_sector
                    received_frame.tx_ori = self.directions(0.)[sending_direction] #!
                    received_frame.rx_ori = self.directions(0.)[direction]
                    received_frame.receiver = self.sensor_owner.id
                    received_frame.signal_strength = signal_strength
                    direction_reading = received_frame
        return direction_reading
