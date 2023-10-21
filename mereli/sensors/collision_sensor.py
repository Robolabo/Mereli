import numpy as np
import numpy.linalg as LA
import pybullet as p
from mereli.sensors import Sensor
from mereli.register import sensor_registry
from mereli.utils import isinstance_of_any

@sensor_registry(name='collision_sensor')
@sensor_registry(name='contact_sensor')
class CollisionSensor(Sensor):
    """ 
    """
    def __init__(self, *args, **kwargs):
        super(CollisionSensor, self).__init__(*args, **kwargs)

    def step(self):
        is_collision = 0
        collisions = self.sensor_owner.physics_client.get_contact_points(self.sensor_owner.id)
        # Filter out collision with earth plane
        is_collision = any([coll[0] > 0 for coll in collisions])
        self.reading = int(is_collision)

    def reset(self):
        self.reading = 0
       
