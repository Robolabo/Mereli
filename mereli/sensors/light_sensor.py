import numpy as np
import pybullet as p
from mereli.register import sensor_registry
from mereli.sensors import DirectionalSensor
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
        self.aperture = 0.785 + .2
        self.propagation = ExpDecayPropagation(rho_att=0.5, phi_att=0.7)# TFM
        self.contact_points = None
        self.reading = {color + '_light_sensor' : np.zeros(8) for color in ['red', 'yellow', 'blue', 'green']}
        self.t = 0


    def step(self, neighborhood):
        """
        
        :param list neighborhood: list of world entities. This parameter is not used at all in this sensor, but it is 
            kept as a parameter because other sensors may need to use it.

        :returns: np.ndarray with the reading of each sector. 
        """
        g_ids = [self.physics_client.physical_sensors['light_sensor'][i]['ghost_link_idx'] for i in range(8)]
        reading = {'red' : np.zeros(8), 'yellow': np.zeros(8), 'blue' : np.zeros(8), 'green' : np.zeros(8)}
        oris = self.directions(self.sensor_owner.orientation[-1])
        # List the entities that overlap with the robot ghost cones.
        if self.contact_points is None or self.t % 10 == 0:
            self.contact_points = self.physics_client.get_contact_points(self.owner_id, ghost_ids=g_ids)
        for i in range(len(g_ids)):
            ori = oris[i]
            luminous_ents = [pt[0] for pt in self.contact_points if pt[1] == g_ids[i] \
                        and pt[0] in self.physics_client.luminous_objects]
            signal_strength = {'red' : 0., 'yellow': 0., 'blue' : 0., 'green' : 0.}
            if len(luminous_ents) == 0:
                continue
            origin = self.get_sensor_position(i) # Coordinates of the physical link of sensor (in the i-th sector).
            # Cast a ray for each luminous object in the sector cone.
            for ent_id in luminous_ents:
                # Query target entity position
                tar_pos = self.physics_client.get_body_position(ent_id, -1)
                
                # Cast a ray between the sensor position and the target entity position.
                ray_res, ray_position = self.sensor_owner.physics_client.ray_cast([origin], [tar_pos])
                # p.addUserDebugLine(origin, tar_pos, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0.0)
                if ray_res == ent_id: # If no obstacles (aside from target)
                    # p.addUserDebugLine(origin, tar_pos, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0.07)
                    # Compute distance to light.
                    rho = np.linalg.norm(np.array(ray_position) - origin)
                    if rho > self.range: # Exit if distance greater than range (may be redundant with ghost cones)
                        continue
                    # Compute misalignment as the angle between the following vectors: the vector of the already
                    # casted ray and the vector pointing at the maximum sensitivity orientation of the sensor in 
                    # i-th sector.
                    vector1 = np.r_[np.cos(ori), np.sin(ori), 0]
                    vector2 = np.array(ray_position) - origin
                    phi = np.arccos(vector1.dot(vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))
                    # Compute actual reading wrt the target light using the fixed propagation model.
                    light_color = self.physics_client.luminous_objects[ent_id]['color']
                    if light_color in signal_strength:
                        signal_strength[light_color] += self.propagation(rho, phi)
                    else:
                        signal_strength[light_color] = self.propagation(rho, phi)
            for color in signal_strength:
                reading[color][i] += np.round(signal_strength[color], 4)
        reading = {color + '_light_sensor' : vec for color, vec in reading.items()}
        # #* Convert dict to the type {color : vector}, where vector is the measurement of all sectors (dim=n_sectors)
        # #* taking into account only the color set by the key.
        # reading = {color : np.array([x[color] for x in reading]) for color in reading[0].keys()}
        # #* Rename keys according to state notation (e.g. red_light_sensor).
        # reading = {color + '_light_sensor' : vec for color, vec in reading.items()}
        # reading['light_sensor'] = np.sum([x for x in reading.values()], 0)
        self.t += 1
        for color in reading:
            # reading[color] += np.random.randn() * 0.0
            self.reading[color] += (0.2) * (reading[color]  - self.reading[color])
        return self.reading

    def step_fast(self, neighborhood):
        pass


    def reset(self):
        self.t = 0
        self.reading = {color + '_light_sensor' : np.zeros(8) for color in ['red', 'yellow', 'blue', 'green']}
        self.contact_points = None



@sensor_registry(name='blue_light_sensor')
class BlueLightSensor(LightSensor):
    def __init__(self, *args, **kwargs):
        super(BlueLightSensor, self).__init__(*args, color='blue', **kwargs)
    
    def step(self, *args):
        #* Get readings from all colors (using parent class step method).
        readings = super().step(*args)
        #* Return just blue reading 
        return readings['blue']

@sensor_registry(name='yellow_light_sensor')
class YellowLightSensor(LightSensor):
    def __init__(self, *args, **kwargs):
        super(YellowLightSensor, self).__init__(*args, color='yellow', **kwargs)
        
    def step(self, *args):
        #* Get readings from all colors (using parent class step method).
        readings = super().step(*args)
        #* Return just yellow reading 
        return readings['yellow']
    
@sensor_registry(name='red_light_sensor')
class RedLightSensor(LightSensor):
    def __init__(self, *args, **kwargs):
        super(RedLightSensor, self).__init__(*args, **kwargs)
    
    def step(self, *args):
        #* Get readings from all colors (using parent class step method).
        readings = super().step(*args)
        #* Return just red reading 
        return readings['red']

@sensor_registry(name='green_light_sensor')
class GreenLightSensor(LightSensor):
    def __init__(self, *args, **kwargs):
        super(GreenLightSensor, self).__init__(*args, color='green', **kwargs)

    def step(self, *args):
        #* Get readings from all colors (using parent class step method).
        readings = super().step(*args)
        #* Return just green reading 
        return readings['green']

# class ColoredLightSensor(LightSensor):
#     def __init__(self, color, *args, **kwargs):
#         super(ColoredLightSensor, self).__init__(*args, **kwargs)
#         self.color = color

#     def target_filter(self, obj):
#         """ Filtering of potential target WorldObjects. """
#         return super().target_filter(obj) and obj.color == self.color