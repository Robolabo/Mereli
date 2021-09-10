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
        self.aperture = 0.785 + .2
        self.propagation = ExpDecayPropagation(rho_att=0.1, phi_att=1)# TFM
    


    def step(self, neighborhood):
        """
        
        :param list neighborhood: list of world entities. This parameter is not used at all in this sensor, but it is 
            kept as a parameter because other sensors may need to use it.

        :returns: np.ndarray with the reading of each sector. 
        """
        reading = []
        g_ids = [self.physics_client.physical_sensors['light_sensor'][i]['ghost_link_idx'] for i in range(8)]
        # List the entities that overlap with the robot ghost cones.
        contact_points = self.physics_client.get_contact_points(self.owner_id, ghost_ids=g_ids)
        for i, ori in enumerate(self.directions(self.sensor_owner.orientation[-1])): 
            luminous_ents = [pt[0] for pt in contact_points if pt[1] == g_ids[i] \
                        and pt[0] in self.physics_client.luminous_objects]
            signal_strength = 0.0
            origin = self.get_sensor_position(i) # Coordinates of the physical link of sensor (in the i-th sector).
            # Cast a ray for each luminous object in the sector cone.
            for ent_id in luminous_ents: 
                # Query target entity position
                tar_pos = self.physics_client.get_body_position(ent_id, -1)
                # Cast a ray between the sensor position and the target entity position.
                ray_res, ray_position = self.sensor_owner.physics_client.ray_cast([origin], [tar_pos])
                # p.addUserDebugLine(origin, tar_pos, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0.07)
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
                    signal_strength += self.propagation(rho, phi)
              
            reading.append(signal_strength)
        if len(reading) != 8: import pdb; pdb.set_trace()
        return np.array(reading)

    def ___step_heavy(self, neighborhood):

        """ WARNING: This class method is an old implementation.

        Step method of the light sensor that estimates the distances to nearby light sources at the current time instant. 
        It returns a numpy array of length equal to ``n_sectors`` with the reading of each independent sector. 
        The main steps of the reading are the following:

        1. The identifiers of the ghost links bonded to the sensor sectors are collected as a list. Using these ghost link ids, 
           it is requested to the physics engine to compute the contant points between the ghost links of the robot and any other 
           entity. This will return a list with the identifier of all the objects that overlap with any of the robot ghost links. In 
           turn, if an entity overlaps with a ghost cone, then it implies that the entity is within the sector sensing area. 

        2. We iterate through the different sectors of the sensor (8 in this case). The loop provides both the index of the sector (from 0 to N-1) 
           and the corresponding sector orientation (only scalar yaw for the moment). Within the first lines inside the loop, the list ``tar_ents`` 
           is filtered out so that only overlapping objects that emit light of the requested color are kept.           
           If this new filtered list is empty, the measured signal strength is zero (no entities within the sensing area of this sector). 
           Otherwise, if there are luminous entities within the sector area, then the signal strength reading is calculated (see below).

        3. Provided that ``tar_ents`` is not empty, the computation of the distance and misalignment estimation to light sources is accomplished as follows. 
           The core idea is to cast a batch of rays, all of them with the same origin coordinates (sensor position) and with destination at equispaced points 
           within an spherical sector determined by the sensor aperture and range.  The following screenshot displays the mentioned 
           ray batchs of a sector:
            
           .. raw:: html

                <img src="../../_static/demo_LS_rays.png" style="width:70%;text-align: center;">
        
        :param list neighborhood: list of world entities. This parameter is not used at all in this sensor, but it is 
            kept as a parameter because other sensors may need to use it.

        :returns: np.ndarray with the reading of each sector. 
        """
        phy = self.sensor_owner.physics_client
        reading = []
        g_ids = [self.physics_client.physical_sensors['light_sensor'][i]['ghost_link_idx'] for i in range(8)]
        contact_points = self.physics_client.get_contact_points(self.owner_id, ghost_ids=g_ids)
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
                    # for o, d in zip([origin]*len(ray_dests), ray_dests):
                    #     p.addUserDebugLine(o, d, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0.)
                    # import pdb; pdb.set_trace()
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