import numpy as np
import numpy.linalg as LA
from mereli.register import sensor_registry
from mereli.sensors import Sensor

@sensor_registry(name='own_orientation_sensor')
class OwnOrientationSensor(Sensor):
    def __init__(self, *args, **kwargs):
        super(OwnOrientationSensor, self).__init__(*args, **kwargs)

    def step(self, neighborhood):
        # ang = (self.sensor_owner.orientation, self.sensor_owner.orientation + 2 * np.pi)[self.sensor_owner.orientation < 0]
        ang = self.sensor_owner.orientation
        return ang / (2*np.pi)