import numpy as np
import numpy.linalg as LA
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