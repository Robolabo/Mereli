import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from mereli.register import sensor_registry
from mereli.sensors import Sensor

@sensor_registry(name='camera')
class Camera(Sensor):
    """ 
    
    **Reference Name**: ``camera``.

    """
    def __init__(self, *args, hpx=60, wpx=60, **kwargs):
        self.w = wpx 
        self.h = hpx 
        super(Camera, self).__init__(*args, **kwargs)

    def step(self):
        """ Step method for reading the camera sensor. 
        
        :param list neighborhood: ``list`` of entities in the surroundings of the robot.

        :returns: 
        """
        camPos = self.sensor_owner.physics_client.get_link_state(self.sensor_owner.id, 4)[0] 
        ori = self.sensor_owner.orientation[2]
        vcam = np.r_[np.cos(ori), np.sin(ori), camPos[-1]]
        tarPos = camPos + 2 * vcam
        camPos += 0.05 * vcam 

        viewMatrix = p.computeViewMatrix(camPos, tarPos, (0,0,1),physicsClientId=self.sensor_owner.physics_client.client)
        projectionMatrix = p.computeProjectionMatrixFOV(fov=45, aspect=1, nearVal=.05, farVal=5.1)
        img_arr = p.getCameraImage(self.w,
                               self.h,
                               viewMatrix=viewMatrix,
                               projectionMatrix=projectionMatrix,
                               shadow=1,
                               lightDirection=[1, 1, 1], physicsClientId=self.sensor_owner.physics_client.client)
        self.reading = img_arr[2] 
    
    def reset(self):
        self.reading = np.zeros([self.w, self.h])
       
