import numpy as np
from mereli.sensors import Sensor
from mereli.register import sensor_registry

@sensor_registry(name='battery_sensor')
class BatterySensor(Sensor):
    """ 
    """
    """def __init__(self, *args, **kwargs):
        super(BatterySensor, self).__init__(*args, **kwargs)

    def step(self):
        if 'battery_sensor' in self.sensor_owner.sensors:
            self.reading = self.sensor_owner.battery.level
        else:
            self.reading = np.array([1.])
            print('WARNING: Using battery sensor but robot has no battery enabled!')

    def reset(self):
        self.reading = np.array([1.]) 
"""
    def __init__(self, *args, color='red', **kwargs):
        super(BatterySensor, self).__init__(*args, **kwargs)
        self.reading = np.zeros(len(self.sensor_owner.battery_colors))

    def step(self):
        for idx, color in enumerate(self.sensor_owner.battery_colors):
            self.reading[idx] = self.sensor_owner.battery.level[idx]        
        return self.reading

    def reset(self):
       self.reading = np.zeros(len(self.sensor_owner.battery_colors))
       
@sensor_registry(name='blue_battery_sensor')
class BlueBatterySensor(BatterySensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if 'blue' in self.sensor_owner.battery_colors:
            self.blue_idx = self.sensor_owner.battery_colors.index('blue')
        else:
            self.blue_idx = None
    
    def step(self, *args):
        if self.blue_idx is not None:
            return np.array([self.reading[self.blue_idx]])
        else:
            # Si no hay batería azul, devolvemos 1.0 por defecto
            return np.array([1.0])


@sensor_registry(name='yellow_battery_sensor')
class YellowBatterySensor(BatterySensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, color='yellow', **kwargs)
        if 'yellow' in self.sensor_owner.battery_colors:
            self.yellow_idx = self.sensor_owner.battery_colors.index('yellow')
        else:
            self.yellow_idx = None

    def step(self, *args):
        if self.yellow_idx is not None:
            return np.array([self.reading[self.yellow_idx]])
        else:
            return np.array([1.0])
