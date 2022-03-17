import numpy as np
import numpy.linalg as LA
import pybullet as p
from mereli.sensors import Sensor
from mereli.register import sensor_registry
from mereli.utils import isinstance_of_any

@sensor_registry(name='collision_sensor')
class CollisionSensor(Sensor):
    """ 
    """
    def __init__(self, *args, **kwargs):
        super(CollisionSensor, self).__init__(*args, **kwargs)

    def step(self, hierarchy):
        is_collision = 0
        for obj in hierarchy:
            if obj.tangible:#type(obj).__name__ in ['Wall', 'Robot3D']:
                collision_info = p.getContactPoints(self.sensor_owner.id, obj.id, physicsClientId=self.sensor_owner.physics_client.client)
                if len(collision_info) > 0:
                        is_collision = 1
            if is_collision:
                return is_collision
        return is_collision  
       