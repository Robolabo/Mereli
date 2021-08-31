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
    """ Directional distance sensor class. It mimics a IR based distance sensor. 
    The sensor is partitioned into multiple sectors that provide measurements 
    solely of their sector coverage and about the corresponding sensing orientation. 
    The reading of each sector is an scalar bounded in [0, 1] that estimates the distance 
    to the closest solid entity within the correspoding sector. A reading of 1 means that 
    the target object is very close and a measurement near 0 indicates that there are no obstacles 
    in that direction.
    
    **Reference Name**: ``distance_sensor``.

    :param Robot sensor_owner: robot object owning and reading from the sensor.
    :param float range: range of coverage of the sensor.
    :param float noise_sigma: std. dev. of the white noise attached to the measurement.
    :param int n_sectors: number of sectors of the sensor.

    :var float aperture: aperture in radians of each sector of the sensor.
    :var ExpDecayPropagation propagation: propagation model to map ``rho`` and ``phi`` into the 
        distance estimation bounded in [0, 1]. 
    """ 
    def __init__(self, *args, **kwargs):
        super(DistanceSensor, self).__init__(*args, **kwargs)
        self.propagation = ExpDecayPropagation(rho_att=0.7, phi_att=1.) # DS=
        # self.propagation = ExpDecayPropagation(rho_att=0.5, phi_att=1.) # DS=
        # self.sensors_idx = None
        self.aperture = 0.61 #1.5 * np.pi / self.n_sectors

    def step(self, neighborhood):
        import pybullet as p  
        phy = self.sensor_owner.physics_client
        reading = []
        g_ids = [phy.physical_sensors['distance_sensor'][i]['ghost_link_idx'] for i in range(8)]
        for idx in g_ids:
            p.setCollisionFilterGroupMask(self.sensor_owner.id, idx, 0b01, 0b01)
        p.performCollisionDetection()
        contact_points = p.getContactPoints(self.sensor_owner.id,)
        for idx in g_ids:
            p.setCollisionFilterGroupMask(self.sensor_owner.id, idx, 0b0, 0b0) 
        for i, ori in enumerate(self.directions(self.sensor_owner.orientation[-1])): 
            tar_ents = [pt[2] for pt in contact_points if pt[3] == g_ids[i]]
            signal_strength = 0.0
            if len(tar_ents) > 0:
                origin = self.get_sensor_position(i)
                # ray_angles = np.linspace(-self.aperture/2, self.aperture/2, 5)
                # ray_dests = [self.range*np.r_[np.cos(ang), np.sin(ang), 0] + origin for ang in ori + ray_angles]
                th_sp, phi_sp = np.linspace(-self.aperture/2, 0, 3), np.linspace(-self.aperture/2, self.aperture/2, 5)
                th_mat, phi_mat = np.meshgrid(np.pi/2 + th_sp, ori + phi_sp)
                X = self.range * np.cos(phi_mat) * np.sin(th_mat)
                Y = self.range * np.sin(phi_mat) * np.sin(th_mat)
                Z = self.range * np.cos(th_mat)
                ray_dests = [origin + np.r_[x, y, z] for x, y, z in zip(X.flatten(), Y.flatten(), Z.flatten())]
                ray_angles = np.maximum(np.abs(th_mat.flatten()), np.abs(phi_mat.flatten()))

                
                # for o, d in zip([origin]*len(ray_dests), ray_dests):
                #     p.addUserDebugLine(o, d, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0.)
                # import pdb; pdb.set_trace()
                ray_res, ray_positions = phy.ray_cast([origin]*len(ray_dests), ray_dests)
                if any(np.array(ray_res) != -1):
                    rhos, phis = zip(*[(np.linalg.norm(pos - origin), phi) for idx, pos, phi in zip(ray_res, ray_positions, ray_angles) if idx != -1])
                    signal_strength = np.mean([self.propagation(rho, phi) for rho, phi in zip(rhos, phi_mat.flatten())])
            reading.append(signal_strength)
        if len(reading) != 8: import pdb; pdb.set_trace()
        return np.array(reading)
            

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
        condition = (kwargs['obj'] is not None\
                    and rho <= self.range\
                    and phi <= self.aperture)
        if direction_reading is None:
            direction_reading = np.random.randn() * self.noise_sigma if self.noise_sigma > 0 else 0.
        
        if condition:
            signal_strength = self.propagation(rho, phi)
            if signal_strength > direction_reading:
                
                my_pos = self.get_sensor_position(args[0])#+ np.r_[0, 0, 0.1] #+ np.r_[0, 0, 0.017]
                if type(kwargs['obj']).__name__ in ['Map', 'Wall']:
                    tar_pos = self.sensor_owner.physics_client.get_closest_point(self.sensor_owner.id, kwargs['obj'].id,  
                                    linkA=self.get_sensor_idx(args[0]), max_dist=self.range)
                else:
                    tar_pos = kwargs['obj'].position + np.r_[0, 0, 0.07] # my_pos[2]]
                # Cast a ray between my_pos y tar_pos to verify if there are obstacles
                
                ray_res = self.sensor_owner.physics_client.ray_cast([my_pos], [tar_pos])[0]
                # print(args[0], rho, ray_res, kwargs['obj'].id, tar_pos)
                if ray_res == kwargs['obj'].id: 
                    # print('IR'+str(args[0]), type(kwargs['obj']).__name__, rho)
                    direction_reading = signal_strength
                    if self.noise_sigma > 0:
                        direction_reading += np.random.randn() * self.noise_sigma
        return direction_reading

    def target_filter(self, obj):
        """ Method devoted to filtering the world objects that should be targeted for a particular sensor.
        In this case it filters out, among all neighboring objects, only the tangible (solid) entities.

        :param WorldObject obj: Potential world object to be sensed.
        
        :returns: Boolean response revealing whether the obj should be explored by the sensor or not.
        """
        return obj.tangible