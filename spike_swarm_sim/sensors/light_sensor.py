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
        self.aperture = 0.785
        self.propagation = ExpDecayPropagation(rho_att=0.1, phi_att=1)# TFM
    
    def step(self, neighborhood):
        phy = self.sensor_owner.physics_client
        reading = []
        g_ids = [self.sensor_owner.physics_client.physical_sensors['light_sensor'][i]['ghost_link_idx'] for i in range(8)]
        contact_points = self.sensor_owner.physics_client.get_contact_points(self.sensor_owner.id, ghost_ids=g_ids)
        for i, ori in enumerate(self.directions(self.sensor_owner.orientation[-1])): 
            tar_ents = [pt[0] for pt in contact_points if pt[1] == g_ids[i]]
            signal_strength = 0.0
            if len(tar_ents) > 0 and any(ent_i in phy.luminous_objects for ent_i in tar_ents):
                if any(phy.luminous_objects.get(ent_i, {}).get('color') == self.color for ent_i in tar_ents):
                    origin = self.get_sensor_position(i)
                
                    th_sp, phi_sp = np.linspace(-self.aperture/2, self.aperture/2, 4), np.linspace(-self.aperture/2, self.aperture/2, 5)
                    th_mat, phi_mat = np.meshgrid(np.pi/2 - 0.4 + th_sp, ori + phi_sp)
                    X = self.range * np.cos(phi_mat) * np.sin(th_mat)
                    Y = self.range * np.sin(phi_mat) * np.sin(th_mat)
                    Z = self.range * np.cos(th_mat)
                    ray_dests = [origin + np.r_[x, y, z] for x, y, z in zip(X.flatten(), Y.flatten(), Z.flatten())]
                    for o, d in zip([origin]*len(ray_dests), ray_dests):
                        p.addUserDebugLine(o, d, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0.)
                    import pdb; pdb.set_trace()
                    ray_res, ray_positions = phy.ray_cast([origin]*len(ray_dests), ray_dests) 
                    if any(ent_i in phy.luminous_objects for ent_i in ray_res):
                        ref_vec = np.r_[np.cos(ori), np.sin(ori), 0] - origin
                        angles = [np.arccos(np.r_[ x, y, z].dot(ref_vec) / (np.linalg.norm(ref_vec)*self.range))\
                                    for x, y, z in zip(X.flatten(), Y.flatten(), Z.flatten())]
                        rhos, phis = zip(*[(np.linalg.norm(pos - origin), phi) for idx, pos, phi in zip(ray_res, ray_positions, angles) if idx != -1])
                        signal_strength = np.mean([self.propagation(rho, phi) for rho, phi in zip(rhos, phis)])
            reading.append(signal_strength)
        if len(reading) != 8: import pdb; pdb.set_trace()
        return np.array(reading)
    

@sensor_registry(name='blue_light_sensor')
class BlueLightSensor(LightSensor):
    def __init__(self, *args, **kwargs):
        super(BlueLightSensor, self).__init__(*args, color='blue', **kwargs)

@sensor_registry(name='yellow_light_sensor')
class YellowLightSensor(LightSensor):
    def __init__(self, *args, **kwargs):
        super(YellowLightSensor, self).__init__(*args, color='yellow', **kwargs)
    
@sensor_registry(name='red_light_sensor')
class RedLightSensor(LightSensor):
    def __init__(self, *args, **kwargs):
        super(RedLightSensor, self).__init__(*args, color='red', **kwargs)

@sensor_registry(name='green_light_sensor')
class GreenLightSensor(LightSensor):
    def __init__(self, *args, **kwargs):
        super(GreenLightSensor, self).__init__(*args, color='green', **kwargs)


# class ColoredLightSensor(LightSensor):
#     def __init__(self, color, *args, **kwargs):
#         super(ColoredLightSensor, self).__init__(*args, **kwargs)
#         self.color = color

#     def target_filter(self, obj):
#         """ Filtering of potential target WorldObjects. """
#         return super().target_filter(obj) and obj.color == self.color