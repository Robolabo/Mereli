import numpy as np
from spike_swarm_sim.register import sensor_registry
from spike_swarm_sim.utils import compute_angle
from spike_swarm_sim.sensors import DirectionalSensor
from .utils.propagation import ExpDecayPropagation

@sensor_registry(name='light_sensor')
class LightSensor(DirectionalSensor):
    """ Directional ambient light sensor that enables the sensing 
    of the light intensity resulting from the emission of luminous 
    WorldObjects (e.g. LightSource).
    """
    def __init__(self, *args, color='red', **kwargs):
        super(LightSensor, self).__init__(*args, **kwargs)
        self.color = color
        self.aperture = 3 * np.pi / self.n_sectors
        self.propagation = ExpDecayPropagation(rho_att=1/200, phi_att=1)

    def _step_direction(self, rho, phi, direction_reading, *args, **kwargs):
        """ Step the sensor of a sector. For a detailed explanation of 
        this method see DirectionalSensor._step_direction.
        """
        condition = kwargs['obj'] is not None\
                    and rho <= kwargs['obj'].range\
                    and kwargs['obj'].color == self.color
                    #and phi <= self.aperture #<= 3*np.pi/self.n_sectors
        if direction_reading is None:
            direction_reading = 0.0
        if condition:
            signal_strength = self.propagation(rho, phi)
            direction_reading += signal_strength
            direction_reading = np.clip(direction_reading, a_min=0, a_max=1)
        return direction_reading

    def _target_filter(self, obj):
        """ Filtering of potential target WorldObjects. 
        #TODO Support for more luminous objects.
        """
        return type(obj).__name__ == 'LightSource'