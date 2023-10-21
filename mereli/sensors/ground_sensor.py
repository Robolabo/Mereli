import numpy as np
from mereli.register import sensor_registry
from mereli.sensors import DirectionalSensor, Sensor

@sensor_registry(name='ground_sensor')
class GroundSensor(Sensor):
    """ Sensor that detects if there is a ground area underneath the robot (binary reading). 
    Additionally, it only detects ground areas of a particular color.
    
    **Reference Name**: ``ground_sensor``.

    :param Robot sensor_owner: robot object owning and reading from the sensor.
    :param float noise_sigma: TODO change to Bernoulli noise.  
    :param str color: color of the light to which the sensor is sensitive.
    """
    def __init__(self, *args, **kwargs):
        super(GroundSensor, self).__init__(*args, **kwargs)
        self.coding = {"black" : 0.5, "grey" : 1.0}
        self.reading = np.array([0.0])

    def step(self, neighborhood):
        """ Step method for reading the ground sensor. 
        
        :param list neighborhood: ``list`` of entities in the surroundings of the robot.

        :returns: numpy array of length 1 with the binary reading. A reading of 1.0 means that a ground 
            area has been detected and a value of 0.0 means that no ground areas was detected.
        """
        self.reading = np.array([0.0]) 
        for ground_area in filter(lambda x: type(x).__name__ == 'GroundArea', neighborhood):
            if np.linalg.norm(self.sensor_owner.position[:2] - ground_area.position[:2]) <= ground_area.radius:
                self.reading = np.array([self.coding.get(ground_area.color, 0.0)])

    def reset(self, seed=None):
        self.reading  = np.array([0.])

@sensor_registry(name='memory_ground_sensor')
class MemoryGroundSensor(Sensor):
    """ 
    """
    def __init__(self, *args, **kwargs):
        super(MemoryGroundSensor, self).__init__(*args, **kwargs)
        self.reading = np.array([0.0])

    def step(self, neighborhood):
        for ground_area in filter(lambda x: type(x).__name__ == 'GroundArea', neighborhood):
            if np.linalg.norm(self.sensor_owner.position[:2] - ground_area.position[:2]) <= ground_area.radius:
                if ground_area.color == 'grey' and self.reading == 0:
                    self.reading = np.array([1.0])
                elif self.reading == 1. and ground_area.color == 'black':
                    self.reading = np.array([0.0])

    def reset(self, seed=None):
        self.reading  = np.array([0.])
