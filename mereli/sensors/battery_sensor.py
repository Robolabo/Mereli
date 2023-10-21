import numpy as np
from mereli.sensors import Sensor
from mereli.register import sensor_registry

@sensor_registry(name='battery_sensor')
class BatterySensor(Sensor):
    """ 
    """
    def __init__(self, *args, **kwargs):
        super(BatterySensor, self).__init__(*args, **kwargs)

    def step(self):
        if 'battery_sensor' in self.sensor_owner.sensors:
            self.reading = self.sensor_owner.battery.level
        else:
            self.reading = np.array([1.])
            print('WARNING: Using battery sensor but robot has no battery enabled!')

    def reset(self):
        self.reading = np.array([1.]) 
