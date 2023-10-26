import numpy as np
import numpy.linalg as LA
from mereli.register import sensor_registry
from mereli.sensors import Sensor

@sensor_registry(name='gps')
@sensor_registry(name='own_position_sensor')
class GPS(Sensor):
    def step(self):
        self.reading = self.sensor_owner.position[:2]
