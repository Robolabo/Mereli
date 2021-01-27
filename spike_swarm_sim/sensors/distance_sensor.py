import numpy as np
import numpy.linalg as LA
import pybullet as p

# from shapely.geometry import LineString
from spike_swarm_sim.sensors import DirectionalSensor
from spike_swarm_sim.register import sensor_registry
from spike_swarm_sim.utils import compute_angle, angle_diff
from .utils.propagation import ExpDecayPropagation


@sensor_registry(name='distance_sensor')
class DistanceSensor(DirectionalSensor):
    """ Directional distance sensor class. It mimics the IR distance sensor. 
    The sensor is partitioned into multiple sector that provide measurements 
    solely of their sector coverage. 
    """
    def __init__(self, *args, **kwargs):
        super(DistanceSensor, self).__init__(*args, **kwargs)
        self.propagation = ExpDecayPropagation(rho_att=1/200, phi_att=1)

    def _step_direction(self, rho, phi, direction_reading, *args, **kwargs):
        """ Step the sensor of a sector. For a detailed explanation of 
        this method see DirectionalSensor._step_direction.
        """
        condition = (kwargs['obj'] is not None\
                    and rho <= self.range\
                    and phi <= np.pi / self.n_sectors + 0.001)
        if direction_reading is None:
            direction_reading = 0.0
        if condition:
            # signal_strength = np.exp(-rho/120)
            signal_strength = self.propagation(rho, phi)
            if signal_strength > direction_reading:
                direction_reading = signal_strength
        return direction_reading

    def _target_filter(self, obj):
        """ Filtering of potential target WorldObjects. """
        return obj.tangible


@sensor_registry(name='distance_sensor3D')
class DistanceSensor3D(DirectionalSensor):
    """ Directional distance sensor class. It mimics the IR distance sensor. 
    The sensor is partitioned into multiple sector that provide measurements 
    solely of their sector coverage.
    """
    def __init__(self, *args, **kwargs):
        super(DistanceSensor3D, self).__init__(*args, **kwargs)
        self.propagation = ExpDecayPropagation(rho_att=1/200, phi_att=1)

        joints = np.array([p.getJointInfo(self.sensor_owner.id, i, physicsClientId=self.sensor_owner.physics_client)[:2]\
            for i in range(p.getNumJoints(self.sensor_owner.id, physicsClientId=self.sensor_owner.physics_client))])

        self.sensors_idx = {i : np.where(np.array(joints) == bytes('base_to_IR'+str(i), 'utf-8'))[0][0]\
                for i in range(self.n_sectors)}

        self.aperture = 2 * np.pi / self.n_sectors

    def _step_direction(self, rho, phi, direction_reading, *args, **kwargs):
        """ Step the sensor of a sector. For a detailed explanation of 
        this method see DirectionalSensor._step_direction.
        """
        
        condition = (kwargs['obj'] is not None\
                    and rho <= self.range\
                    and phi <= self.aperture)
        if direction_reading is None:
            direction_reading = 0.0
        if condition:
            signal_strength = self.propagation(rho, phi)
            if signal_strength > direction_reading:
                # my_pos = self.get_positions()[args[0]] * np.array([1, 1, 0.0]) + np.array([0, 0, 0.13])
                # ray_res = p.rayTest(my_pos, kwargs['obj'].position*np.array([1, 1, 0.0])+ np.array([0, 0, 0.08]),\
                #                 physicsClientId=self.sensor_owner.physics_client)
                my_pos = self.get_position(self.sensors_idx[args[0]]) + np.array([0., 0, -0.02])
                ray_res = p.rayTest(my_pos, kwargs['obj'].position,\
                            physicsClientId=self.sensor_owner.physics_client)
                # print(rho, ray_res[0][0], 'IR'+str(args[0]))
                if ray_res[0][0] == kwargs['obj'].id:
                    direction_reading = signal_strength
            # else:
            #     import pdb; pdb.set_trace()
        return direction_reading
    
    def get_position(self, idx):
        return np.array(p.getLinkState(self.sensor_owner.id, idx, physicsClientId=self.sensor_owner.physics_client)[0])
      
    def _target_filter(self, obj):
        """ Filtering of potential target WorldObjects. """
        return obj.tangible