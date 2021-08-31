import numpy as np
import numpy.linalg as LA
import pybullet as p
from spike_swarm_sim.utils import compute_angle, angle_diff, key_of
from spike_swarm_sim.register import sensors

class Sensor:
    """ Base class of robot sensors.

    :param Robot sensor_owner: robot object owning and reading from the sensor.
    :param float range: range of coverage of the sensor.
    :param float noise_sigma: std. dev. of the white noise attached to the measurement.  
    """
    def __init__(self, sensor_owner, range=2, noise_sigma=0.0):
        self.sensor_owner = sensor_owner
        self.noise_sigma = noise_sigma
        self.range = range
        self.reading = None
        # self.sensor_idx = {}#!

    def step(self, neighborhood):
        raise NotImplementedError

class DirectionalSensor(Sensor):
    """ Base class for directional sensors. Directional or sectorized sensors are 
    sensors that can acquire the orientation from where the measurement was sensed.
    They have multiple sensor (one per sector), normally equally distributed around the
    robot perimeter, and can read the local surroundings of each sector area. In this 
    case, the sensor of each sector has a limited range and aperture in radians. Some 
    examples of directional sensors are the ``light_sensor`` and the ``distance_sensor``.
    Excluding some exceptions such as the IR communication receiver, the reading of the 
    sensor is an numpy array with length equal to the number of sectors.
    
    This class is a base class for directional sensors. It only implements the ``step`` 
    method, which is, at first, common to every directional sensor. Nonetheless, when inheriting 
    from this sensor, the class methods ``target_filter`` and ``step_direction`` have to be 
    filled to implement the precise sensor. In brief, ``target_filter`` filters out the entites that 
    are targeted by the sensor (e.g. ``LightSource`` in the case of the ``LightSensor``). 
    ``step_direction`` implements the reading of the sensor for a single direction.

    :param Robot sensor_owner: robot object owning and reading from the sensor.
    :param float range: range of coverage of the sensor.
    :param float noise_sigma: std. dev. of the white noise attached to the measurement.
    :param int n_sectors: number of sectors of the sensor.

    :var float aperture: aperture in radians of each sector of the sensor.
    """
    def __init__(self, *args, n_sectors=8, **kwargs):
        super(DirectionalSensor, self).__init__(*args, **kwargs)
        self.n_sectors = n_sectors
        self.aperture = np.pi / self.n_sectors

    def target_filter(self, obj):
        """ Method devoted to filtering the world objects that should be targeted for a particular sensor.
        For example: a light sensor will filter out only luminous objects.
        This method must be overwritten in each directional sensor that inherits from DirectionalSensor 
        in order to particularize its functioning.
        
        :param WorldObject obj: Potential world object to be sensed.
        
        :returns: Boolean response revealing whether the obj should be explored by the sensor or not.
        """
        raise NotImplementedError

    def step_direction(self, rho, phi, direction_reading, direction, obj=None, diff_vector=None):
        """ Method that specifies the particular behavior of a directional sensor in each sensing direction.
        It must return the reading of the current direction.
        This method must be overwritten in each directional sensor that inherits from DirectionalSensor 
        in order to particularize its functioning.

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
        raise NotImplementedError
    
    def step(self, neighborhood):
        """ Main method for steping the sensor and capturing nearby environment events.
        It senses the environment independently in each of the sensing sectors. The 
        particular behavior of each sensor inheriting this class should be specified in 
        ``step_direction`` method.

        :param list neighborhood: List of target neighboring entities (``WorldObject`` types) (excluding the robot 
            reading the sensor).
        
        :returns: numpy array with the measurement in each direction. In exceptional cases 
            it may return a list of python dictionaries (see ``CommunicationReceiver``).
        """
        # Initial zero readings. This will the reading in the cases with no sensing candidates.
        readings = [self.step_direction(0, 0, None, 0, obj=None) for _ in range(self.n_sectors)]
        # Orientation of the robot using the sensor.
        orientation = self.sensor_owner.orientation
        # Entities that are, at first, candidates to be sensed by the robot sensor. The candidates 
        # are filtered out using the target_filter method.
        featured_objects = [obj for obj in neighborhood \
                            if self.target_filter(obj) and obj.id != self.sensor_owner.id]
        #! Improve code
        for obj in featured_objects:
            # If Wall, detect the distance to the closest point to the wall (CoM would not work).
            if type(obj).__name__ in ['Map', 'Wall']:
                # closest_points = p.getClosestPoints(self.sensor_owner.id, obj.id, 200,\
                #         linkIndexA=-1, linkIndexB=-1, physicsClientId=self.sensor_owner.physics_client.client)
                closest_pt_pos = self.sensor_owner.physics_client.get_closest_point(self.sensor_owner.id, obj.id,  
                                    linkA=-1, linkB=-1, max_dist=self.range + 0.11)
                if len(closest_pt_pos) == 0: 
                    continue
                # closest_pt_pos = np.array(closest_points[0][6])
                v = closest_pt_pos - self.sensor_owner.position
            else:
                # Provisional solution: For the rest of objects (generally small objects) compute the distance 
                # to the CoM of the target object.
                v = obj.position - self.sensor_owner.position #!OJO
            orientation = self.sensor_owner.orientation[-1]
            rho = LA.norm(v)
            # Discard distant entities.
            if rho >= self.range:
                continue
            # Angle difference between sensor directions and ang(v).
            # Compute the directions with candidate entities to be perceived by the sensor. 
            phi_values = np.array([angle_diff(compute_angle(v[:2]), theta) for theta in self.directions(orientation)])
            featured_sensors = np.where(phi_values <= self.aperture)[0]
            phi_values = phi_values[featured_sensors]
            # For each sector with candidate targets to perceive, read the sector's sensor. 
            for k, phi in zip(featured_sensors, phi_values):
                readings[k] = self.step_direction(rho, phi, readings[k], k, obj=obj, diff_vector=v)
        return np.array(readings) if not isinstance(readings[0], dict) else readings

    def directions(self, theta):
        """ Returns the vector of sensing directions of the sectors based on the robot heading orientation.

        :param float theta: orientation of the robots using the sensor.
        
        :returns: numpy Array with the absolute directions of each sensor (starting from theta).
        """
        sensor_name = [ref_name for ref_name, sens_cls in sensors.items() if isinstance(self,sens_cls)][0]
        return np.array([theta - self.sensor_owner.physics_client.get_sensor_orientation(self.sensor_owner.id, 
                    sensor_name=sensor_name, sector=i) for i in range(self.n_sectors)])
        # return np.array([theta + i * (2 * np.pi / self.n_sectors) for i in range(self.n_sectors)])

    def get_sensor_position(self, sector):
        """ Gets the position of the sensor of a sector. Each sector is represented by a small 3D model 
        used to cast rays and compute the readings wrt it. 
        
        .. todo::
            TODO: For the moment only available in 3D. Create method in 2D engine.

        :param int sector: sector of a the sectorized sensor to be requested.

        :returns: the numpy array position of the sensor within the robot model.
        """
        sensor_name = [ref_name for ref_name, sens_cls in sensors.items() if isinstance(self,sens_cls)][0]
        return self.sensor_owner.physics_client.get_sensor_position(self.sensor_owner.id, 
                    sensor_name=sensor_name, sector=sector)[0]
    
    def get_sensor_idx(self, sector):
        """ Gets the position of the sensor of a sector. Each sector is represented by a small 3D model 
        used to cast rays and compute the readings wrt it. 
        
        .. todo::
            TODO: For the moment only available in 3D. Create method in 2D engine.

        :param int sector: sector of a the sectorized sensor to be requested.

        :returns: the numpy array position of the sensor within the robot model.
        """
        sensor_name = [ref_name for ref_name, sens_cls in sensors.items() if isinstance(self,sens_cls)][0]
        return self.sensor_owner.physics_client.get_sensor_position(self.sensor_owner.id, 
                    sensor_name=sensor_name, sector=sector)[1]