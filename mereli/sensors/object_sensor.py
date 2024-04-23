
import numpy as np
import numpy.linalg as LA
from mereli.register import sensor_registry
from mereli.sensors import Sensor

@sensor_registry(name='object_sensor')
class ObjectSensor(Sensor):

    def step(self):
        R = 0.6 
        objs = self.sensor_owner.physics_client.geometry_objects
        own_pos = self.sensor_owner.position[:2]
        clst_obj = None
        clst_dist = 0
        for obj in objs.values():
            dist_obj = np.linalg.norm(obj['position'] - own_pos) - obj['size'] / 2  
            if dist_obj < R: 
                if clst_obj is None or dist_obj < clst_dist:
                    clst_obj = obj 
                    clst_dist = dist_obj
        # if self.reading is not None:
        #     __import__('pdb').set_trace()
        self.reading = clst_obj
