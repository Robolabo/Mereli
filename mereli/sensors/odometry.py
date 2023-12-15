import numpy as np
from mereli.register import sensor_registry
from mereli.sensors import Sensor, Encoder

@sensor_registry(name='odometry')
class OdometrySensor(Sensor):
    """ Robot Sensor that estimates the current position and orientation of the robot using basic odometry.
    The estimates assume some initial values for position and orientation. 
    IMPORTANT: This odometry sensor is only valid for two-wheeled robots (depending on the robot dimensions R and b should be adapted). 
    """
    def __init__(self, *args, initial_position=[0,0], initial_orientation=0.0,  **kwargs):
        super(OdometrySensor, self).__init__(*args, **kwargs)
        self.initial_position = initial_position
        self.initial_orientation = initial_orientation
        self.running_error = {'position' : np.zeros(2), 'orientation' : 0.0}
        self.b = 0.12 # Dist between wheels
        self.R = 0.0390625 # Wheel radius

    def step(self):
        enc = self.sensor_owner.sensors['encoder'].reading
        dt = 10 * self.physics_client.dt
        dUl = self.R * enc[1] * dt
        dUr = self.R * enc[0] * dt
        w = (dUr - dUl) / self.b
        V = (dUr + dUl) / 2
        dx = np.cos(self.reading['orientation']) * V
        dy = np.sin(self.reading['orientation']) * V
        dTh = w 
        # Update new estimates
        self.reading['position'] += np.r_[dx, dy] 
        self.reading['orientation'] += dTh
        self.reading['orientation'] = self.reading['orientation'] % (2*np.pi)
    
        # Compute running errors (debug)
        real_pos = self.sensor_owner.position[:2]
        real_ori = self.sensor_owner.orientation[-1]
        self.running_error['position'] = real_pos - self.reading['position']
        self.running_error['orientation'] = min(np.abs(self.reading['orientation']-real_ori), 2*np.pi - np.abs(self.reading['orientation']-real_ori))
    
    def reset(self):
        self.reading = {'position' : np.array(self.initial_position).astype(float), 'orientation' : self.initial_orientation} 
        self.running_error = {'position' : np.zeros(2), 'orientation' : 0.0}
       
