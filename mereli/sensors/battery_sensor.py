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
        self.reading = np.zeros(len(self.sensor_owner.battery_colors)) #num readings = num battery colors

    def step(self):

        for idx, color in enumerate(self.sensor_owner.battery_colors): #update reading for each battery color
            self.reading[idx] = self.sensor_owner.battery.level[idx]       
        

    def reset(self):
       self.reading = np.zeros(len(self.sensor_owner.battery_colors))
       
@sensor_registry(name='blue_battery_sensor')
class BlueBatterySensor(BatterySensor):
    def __init__(self, *args, **kwargs):
        super(BlueBatterySensor,self).__init__(*args, **kwargs)
        
        self.blue_idx = None
        # Sobreescribir la lectura para que sea tamaño 1
        self.reading = np.array([1.0])
    
    def step(self, *args):

        # Ahora que hemos reducido el tamaño de self.reading
        # Leer los niveles de batería completos directamente del robot.
        all_battery_levels = self.sensor_owner.battery.level

        if 'blue' in self.sensor_owner.battery_colors: #retrieve index of blue battery
            self.blue_idx = self.sensor_owner.battery_colors.index('blue')
            print("ÍNDICE BATERÍA AZUL:", self.blue_idx)
        else:
            self.blue_idx = None    

        if self.blue_idx is not None:
            blue_value = all_battery_levels[self.blue_idx]
            print("VALOR BATERÍA AZUL:", blue_value)
            self.reading[0] = blue_value #extracts only the blue battery level
        
        else:
            # Si no hay batería azul, devolvemos 1.0 por defecto
            self.reading[0] = 1.0
        


@sensor_registry(name='yellow_battery_sensor')
class YellowBatterySensor(BatterySensor):
    def __init__(self, *args, **kwargs):
        super(YellowBatterySensor,self).__init__(*args, color='yellow', **kwargs)

        self.yellow_idx = None

        self.reading = np.array([1.0])

    def step(self, *args):
        
        all_battery_levels = self.sensor_owner.battery.level

        if 'yellow' in self.sensor_owner.battery_colors:
            self.yellow_idx = self.sensor_owner.battery_colors.index('yellow')
        else:
            self.yellow_idx = None

        if self.yellow_idx is not None:
            self.reading[0] = all_battery_levels[self.yellow_idx]
        else:
            self.reading[0] = 1.0
