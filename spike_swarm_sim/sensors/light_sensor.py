import numpy as np
import pybullet as p
from spike_swarm_sim.register import sensor_registry
from spike_swarm_sim.sensors import DirectionalSensor
from .utils.propagation import ExpDecayPropagation

@sensor_registry(name='light_sensor')
class LightSensor(DirectionalSensor):
    """ Directional ambient light sensor that enables the sensing 
    of the light intensity resulting from the emission of luminous 
    WorldObjects (e.g. ``LightSource``).
    The sensor is partitioned into multiple sectors that provide measurements 
    solely of their sector coverage and about the corresponding sensing orientation. 
    The reading of each sector is an scalar bounded in [0, 1] that estimates the received 
    light intensity within the correspoding sector. 
    
    **Reference Name**: ``light_sensor``.

    :param Robot sensor_owner: robot object owning and reading from the sensor.
    :param float range: range of coverage of the sensor.
    :param float noise_sigma: std. dev. of the white noise attached to the measurement.
    :param int n_sectors: number of sectors of the sensor.
    :param str color: color of the light to which the sensor is sensitive.

    :var float aperture: aperture in radians of each sector of the sensor.
    :var ExpDecayPropagation propagation: propagation model to map ``rho`` and ``phi`` into the 
        distance estimation bounded in [0, 1]. 
    """ 
    def __init__(self, *args, color='red', **kwargs):
        super(LightSensor, self).__init__(*args, **kwargs)
        self.color = color
        self.aperture = 3 * np.pi / self.n_sectors
        self.propagation = ExpDecayPropagation(rho_att=0.1, phi_att=1)# TFM
        # self.propagation = ExpDecayPropagation(rho_att=0.15, phi_att=1)
    
    def target_filter(self, obj):
        """ Method devoted to filtering the world objects that should be targeted for a particular sensor.
        In this case it filters out, among all neighboring objects, only the light sources.

        .. todo::
            #TODO Support for more luminous objects apart from light sources.

        :param WorldObject obj: Potential world object to be sensed.
        
        :returns: Boolean response revealing whether the obj should be explored by the sensor or not.
        """
        return type(obj).__name__ == 'LightSource'

    def step_direction(self, rho, phi, direction_reading, *args, **kwargs):
        """ Method that specifies the particular behavior of a directional sensor in each sensing direction.
        It must return the reading of the current direction. Only objects that are within the range and 
        aperture are considered.

        :param float rho: Eucliden distance between the object sensing and the object (obj) sensed.
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
        condition = kwargs['obj'] is not None\
                    and rho <= kwargs['obj'].range\
                    # and kwargs['obj'].color == self.color
                    #and phi <= self.aperture #<= 3*np.pi/self.n_sectors
        if direction_reading is None:
            direction_reading = np.random.randn() * self.noise_sigma if self.noise_sigma > 0 else 0.
        if condition:
            my_pos = self.get_sensor_position(args[0]) + np.r_[0,0,0.02]
            tar_post = kwargs['obj'].position
            ray_res = p.rayTest(my_pos, tar_post, physicsClientId=self.sensor_owner.physics_client.client)
            signal_strength = self.propagation(rho, phi)
            if ray_res[0][0] == -1:
                direction_reading += signal_strength
                if self.noise_sigma > 0:
                    direction_reading += np.random.randn() * self.noise_sigma
                direction_reading = np.clip(direction_reading, a_min=0, a_max=1)
               
        return direction_reading
    
class ColoredLightSensor(LightSensor):
    def __init__(self, color, *args, **kwargs):
        super(ColoredLightSensor, self).__init__(*args, **kwargs)
        self.color = color

    def target_filter(self, obj):
        """ Filtering of potential target WorldObjects. """
        return super().target_filter(obj) and obj.color == self.color

@sensor_registry(name='blue_light_sensor')
class BlueLightSensor(ColoredLightSensor):
    def __init__(self, *args, **kwargs):
        super(BlueLightSensor, self).__init__('blue', *args, **kwargs)

@sensor_registry(name='yellow_light_sensor')
class YellowLightSensor(ColoredLightSensor):
    def __init__(self, *args, **kwargs):
        super(YellowLightSensor, self).__init__('yellow', *args, **kwargs)
    
@sensor_registry(name='red_light_sensor')
class RedLightSensor(ColoredLightSensor):
    def __init__(self, *args, **kwargs):
        super(RedLightSensor, self).__init__('red', *args, **kwargs)

@sensor_registry(name='green_light_sensor')
class GreenLightSensor(ColoredLightSensor):
    def __init__(self, *args, **kwargs):
        super(GreenLightSensor, self).__init__('green', *args, **kwargs)
