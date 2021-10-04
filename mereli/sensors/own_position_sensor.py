import numpy as np
import numpy.linalg as LA
from mereli.register import sensor_registry
from mereli.sensors import Sensor

@sensor_registry(name='own_position_sensor')
class OwnPositionSensor(Sensor):
    def __init__(self, *args, **kwargs):
        super(OwnPositionSensor, self).__init__(*args, **kwargs)

    def step(self, neighborhood):
        return self.sensor_owner.pos