import numpy as np
import numpy.linalg as LA
from mereli.register import sensor_registry
from mereli.sensors import Sensor

""" ALSO INCLUDES NEIGHBORHOOD ORIENTATIONS """
@sensor_registry(name='neighborhood_pos_sensor')
@sensor_registry(name='neighborhood_gps')
class NeighborhoodPositionSensor(Sensor):
    def __init__(self, *args, **kwargs):
        super(NeighborhoodPositionSensor, self).__init__(*args, **kwargs)
        self.range = 5000

    def step(self, neighborhood):
        neighborhood_pos = []
        for obj in self.sensor_owner.neighbors: 
            if self.sensor_owner.id != obj.id and obj.controllable:
                dist = LA.norm(obj.position - self.sensor_owner.position)
                neighborhood_pos.append(np.hstack((obj.position, obj.orientation)))
        self.readings = np.array(neighborhood_pos)


