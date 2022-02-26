import numpy as np
import numpy.linalg as LA
import pybullet as p

from mereli.sensors import DirectionalSensor
from mereli.register import sensor_registry
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
        self.aperture = 0.61 #1.5 * np.pi / self.n_sectors
        self.contact_points = None
        self.reading = np.zeros(8)
        self.t = 0

    def step(self, neighborhood):
        r""" Step method of the distance sensor that estimates the distances to nearby entities at the current time instant. 
        It returns a numpy array of length equal to ``n_sectors`` with the reading of each independent sector. 
        The main steps of the reading are the following:

        1. The identifiers of the ghost links bonded to the sensor sectors are collected as a list. Using these ghost link ids, 
           it is requested to the physics engine to compute the contant points between the ghost links of the robot and any other 
           entity. This will return a list with the identifier of all the objects that overlap with any of the robot ghost links. In 
           turn, if an entity overlaps with a ghost cone, then it implies that the entity is within the sector sensing area.
        
        2. We iterate through the different sectors of the sensor (8 in this case). The loop provides both the index of the sector (from 0 to N-1) 
           and the corresponding sector orientation (only scalar yaw for the moment). Within the first lines inside the loop, the list ``tar_ents``, 
           filters out the identifiers of the overlapping entities within the sector ghost cone of the i-th sector. If this new list is empty, the 
           measured signal strength is zero (no entities within the sensing area of this sector). Otherwise, if there are entities within the sector area, 
           then the signal strength reading is calculated (see below).

        3. Provided that ``tar_ents`` is not empty, the computation of the distance and misalignment estimation to solid objects is accomplished as follows. 
           The core idea is to cast a batch of rays, all of them with the same origin coordinates (sensor position) and with destination at equispaced points 
           of a sector area of a total angle given by the sensor aperture and a radius given by the sensor range. The following screenshot displays the mentioned 
           ray batchs of a sector:

           .. raw:: html

                <img src="../../_static/demo_DS_rays.png" style="width:70%;text-align: center;">

           In terms of code, the destination of the rays are computed as follows:
        
                * Using the function ``np.linspace(-self.aperture/2, self.aperture/2, 5)`` we obtain 5 equispaced angle points inside the interval :math:`[-A/2,\, A/2]` rad, 
                  where :math:`A` is the aperture of the sector in radians. Even though in the code we use a total of 5 rays, generically speaking lets denote :math:`N` to the 
                  total number of rays casted from each sensor's sector.  
        
                * Provided that :math:`R` stands for the range in meters of the sensor and :math:`\mathbf{o}` is the position of the physical sensor, the destination positions are:

                  .. math::
                      :nowrap:
                  
                      \[ \mathbf{d} = \mathbf{o} + R \left(\begin{array}{c}\cos(\alpha_i)\\ \sin(\alpha_i) \end{array}\right) \]
  
                  where :math:`\alpha_i, \, \forall i\in\{1,\,\dots,\,N\}` are the angles previously computed using ``np.linspace``.
    
        The cast of the rays is accomplished by the physics engine, returning both a list of the ids of the fist entity intersecting each ray (or -1 is no obj was hitted) and the 
        position of the intersection to the first intersecting solid object. 
        


        :param list neighborhood: list of world entities. This parameter is not used at all in this sensor, but it is 
            kept as a parameter because other sensors may need to use it.

        :returns: np.ndarray with the reading of each sector.
        """
        
        g_ids = [self.sensor_owner.physics_client.physical_sensors['distance_sensor'][i]['ghost_link_idx'] for i in range(8)]
        reading = np.zeros(len(g_ids))
        if self.contact_points is None or self.t % 15 == 0:
            self.contact_points = self.sensor_owner.physics_client.get_contact_points(self.sensor_owner.id, ghost_ids=g_ids)
        # return np.zeros(8)
        oris = self.directions(self.sensor_owner.orientation[-1])
        
        for i in range(8):
            ori = oris[i]
            tar_ents = [pt[0] for pt in self.contact_points if pt[1] == g_ids[i] and pt[0] not in self.sensor_owner.physics_client.luminous_objects and pt[0] != 0]
            signal_strength = 0.0
            if len(tar_ents) > 0:
                origin = self.get_sensor_position(i)
                ray_angles = np.linspace(-self.aperture/2, self.aperture/2, 4)
                ray_dests = [self.range*np.r_[np.cos(ang), np.sin(ang), -0.05] + origin for ang in ori + ray_angles]
                
                # for o, d in zip([origin]*len(ray_dests), ray_dests):
                #     p.addUserDebugLine(o, d, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=15)
                ray_res, ray_positions = self.sensor_owner.physics_client.ray_cast([origin]*len(ray_dests), ray_dests)
                ray_positions = [np.round(ps, 5) for ps in ray_positions]
                if any(np.array(ray_res) != -1):
                    
                    rhos, phis = zip(*[(np.linalg.norm(np.round(pos, 5) - np.round(origin, 5)), phi) for idx, pos, phi in zip(ray_res, ray_positions, ray_angles) if idx != -1 and idx != 0])
                    signal_strength = np.mean([self.propagation(rho, phi) for rho, phi in zip(rhos, ray_angles.flatten())])
                    # if self.t == 171:import pdb; pdb.set_trace()
            reading[i] += np.round(signal_strength, 4)# + np.random.randn() * 0.0
        self.t += 1
        self.reading += (0.2) * (np.array(reading) - self.reading)
        # if self.t > 171:import pdb; pdb.set_trace()
        return self.reading

    def reset(self):
        self.reading = np.zeros(8)
        self.contact_points = None  
        self.t = 0