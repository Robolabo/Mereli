
import numpy as np
from mereli.sensors import Sensor
from mereli.register import sensor_registry

@sensor_registry(name='goal_coordinates_sensor')
class GoalCoordinatesSensor(Sensor):
    """ 
    """
    def __init__(self, *args, goal_coordinates='random', xrange=(-1,1), yrange=(-1,1), **kwargs):
        super(GoalCoordinatesSensor, self).__init__(*args, **kwargs)
        self.goal_coordinates = goal_coordinates if goal_coordinates != 'random' else None 
        self.random_generation = goal_coordinates == 'random'
        self.xrange = yrange
        self.yrange = xrange

    def step(self):
        self.reading = (self.sensor_owner.position[:2] - self.goal_coordinates) /2  
    

    def set_goal_coordinates(self, new_coords):
        self.goal_coordinates = new_coords
        self.reading = new_coords

    def reset(self):
        # self.reading = self.goal_coordinates
        if self.random_generation:
            self.goal_coordinates = np.random.uniform(low=[self.xrange[0], self.yrange[0]],high=[self.xrange[1], self.yrange[1]], size=2)
            
