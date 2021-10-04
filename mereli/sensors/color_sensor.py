import numpy as np
import pybullet as p
from mereli.register import sensor_registry
from mereli.utils import compute_angle
from mereli.sensors import Sensor, DirectionalSensor
from .utils.propagation import ExpDecayPropagation


@sensor_registry(name='color_sensor')
class ColorSensor(DirectionalSensor):
    """ Directional color sensor that detects objects of a certain color in the local surroundings.
    The sensor is partitioned into multiple sectors that provide measurements 
    solely of their sector coverage and about the corresponding sensing orientation. 
    The reading of each sector is an binary reading inticatin whether an object of the color was detected or 
    not in the sector coverage. The color sensor is a high level sensor that simplyfies the use of 
    a camara plus a postprocessing unit. The analogous postprocessing would be to detect the amount of 
    pixels of the color and detect an object of that color provided that the number of detected pixels 
    surpasses a certain threshold. 
    
    **Reference Name**: ``color_sensor``.

    :param Robot sensor_owner: robot object owning and reading from the sensor.
    :param float range: range of coverage of the sensor.
    :param float noise_sigma: std. dev. of the white noise attached to the measurement.
    :param int n_sectors: number of sectors of the sensor.
    :param str color: color of the light to which the sensor is sensitive.

    :var float aperture: aperture in radians of each sector of the sensor.
    :var ExpDecayPropagation propagation: propagation model to map ``rho`` and ``phi`` into the 
        distance estimation bounded in [0, 1]. 
    """
    def __init__(self, *args, color='blue', **kwargs):
        super(ColorSensor, self).__init__(*args, **kwargs)
        # self.color = color
        self.aperture = 1.5 * np.pi / self.n_sectors
        self.propagation = ExpDecayPropagation(rho_att=0.2, phi_att=1)
        self.color = color

    def target_filter(self, obj):
        """ Method devoted to filtering the world objects that should be targeted for a particular sensor.
        In this case it filters out, among all neighboring objects, only the solid objects of the sentitive color.

        :param WorldObject obj: Potential world object to be sensed.
        
        :returns: Boolean response revealing whether the obj should be explored by the sensor or not.
        """
        # return type(obj).__name__ in ['Cube'] and not obj.is_grasped
        return obj.tangible and (hasattr(obj, 'color') and obj.color == self.color) and not obj.is_grasped

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
                    and rho <= self.range\
                    and phi <= self.aperture #<= 3*np.pi/self.n_sectors
        if direction_reading is None:
            direction_reading = 0.
        # import pdb; pdb.set_trace()
        if condition and direction_reading == 0.0:
            my_pos = self.get_sensor_position(args[0]) + np.r_[0, 0, 0.1] #+ np.r_[0, 0, 0.017]
            tar_pos = kwargs['obj'].position + np.r_[0, 0, 0.07] # my_pos[2]]
            # ray_res = p.rayTest(my_pos, tar_pos, physicsClientId=self.sensor_owner.physics_client)[0][0]
            ray_res = self.sensor_owner.physics_client.ray_cast(my_pos, tar_pos)
            if ray_res == kwargs['obj'].id:
                direction_reading = 1.
        return direction_reading
    

@sensor_registry(name='object_grasped_sensor')
class ObjectGraspedSensor(Sensor):
    def __init__(self, *args, **kwargs):
        super(ObjectGraspedSensor, self).__init__(*args, **kwargs)
    
    def step(self, neighborhood):
        reading = int(self.sensor_owner.actuators['grasp_actuator'].cube_grasped is not None)
        return np.array([reading])