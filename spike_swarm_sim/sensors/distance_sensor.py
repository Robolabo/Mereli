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
class DistanceSensor3D(DistanceSensor):
    """ Directional distance sensor class. It mimics the IR distance sensor. 
    The sensor is partitioned into multiple sector that provide measurements 
    solely of their sector coverage.
    """
    def __init__(self, *args, **kwargs):
        super(DistanceSensor3D, self).__init__(*args, **kwargs)

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
            # tar_pos = self.sensor_owner.position + kwargs['diff_vector']  + np.array([0,0,0.1])
            my_pos = self.get_positions()[args[0]] * np.array([1, 1, 0.0])  + np.array([0,0,0.15])
            
            ray_res = p.rayTest(my_pos, kwargs['obj'].position * np.array([1, 1, 0.0])  + np.array([0, 0, 0.11]),)
            # print(rho,ray_res[0][0], )
            if ray_res[0][0] == kwargs['obj'].id:
                signal_strength = self.propagation(rho, phi)
                if signal_strength > direction_reading:
                    direction_reading = signal_strength
            # else:
            #     import pdb; pdb.set_trace()
        return direction_reading
    
    def get_positions(self):
        return [self.sensor_owner.position + 0.012 * np.r_[np.cos(ang), np.sin(ang), 0.0]\
            for ang in self.directions(self.sensor_owner.orientation[-1])]

    #TODO QUITAR DE AQUI
    def directions(self, theta):
        """
        Returns the vector of sensing directions of the sectors based on the robot orientation.
        - Args:
            theta [float] -> orientation of the robots using the sensor.
        - Returns:
            Numpy Array with the absolute directions of each sensor (starting from theta).
        """
        return np.array([theta + i * (2 * np.pi / self.n_sectors) for i in range(self.n_sectors)])