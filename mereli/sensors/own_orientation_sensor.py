import numpy as np
import numpy.linalg as LA
from mereli.register import sensor_registry
from mereli.sensors import Sensor


@sensor_registry(name='own_orientation_sensor')
@sensor_registry(name='compass')
class Compass(Sensor):
    def __init__(self, *args, **kwargs):
        super(Compass, self).__init__(*args, **kwargs)

    def step(self):
        # ang = (self.sensor_owner.orientation, self.sensor_owner.orientation + 2 * np.pi)[self.sensor_owner.orientation < 0]
        ang = self.sensor_owner.orientation[-1]
        self.reading = ang / (2*np.pi)
