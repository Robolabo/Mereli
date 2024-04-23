import numpy as np
from mereli.objects import WorldObject
from mereli.register import world_object_registry

@world_object_registry(name='cube')
class Cube(WorldObject):
    """ Cube object of a certain mase, color and side length. If 2D then it is a
    square (TODO) and if 3D it is a cube.
    
    **Reference Name**: ``cube``

    :param float side_len: length of the sides [metres] of the cube.
    :param str color: color of the cube.
    :param float mass: total mass of the ball.

    :var bool is_grasped: flag indicating if a robot has grasped the object through its 
        high level grasp and drop sensor.
    """
    def __init__(self,  position, orientation, *args, color='blue', mass=1., side_len=0.3, **kwargs):
        # position = list(position)
        # position[-1] = side_len / 2 - 0.1
        # __import__('pdb').set_trace()
        if color == 'random': 
            # cluster = np.random.choice([0,1,2])
            # self.color = n))p.random.normal(scale=0.07)
            self.color = np.random.uniform(low=0, high=1, size=3)
        else:
            self.color = color
        self.mass = mass
        self.size = None
        if isinstance(side_len, str) and  "random" in side_len:
            rnd_dist = side_len.split(':')[1]
            if rnd_dist[0] == 'N':
               rnd_dist_params = rnd_dist.split('(')[1].split(')')[0]
               dist_mean = float(rnd_dist_params.split(',')[0])
               dist_std = float(rnd_dist_params.split(',')[1])
               np.clip(np.random.normal(dist_mean, dist_std), a_min=0.05, a_max=0.5) 

            self.size = np.random.uniform(low=0.05, high=0.5)
        else:
            self.size = side_len
        
        self.scaling = self.size 
        position[-1] = self.scaling / 2 - .1
        super(Cube, self).__init__('entities/cube/cube', position, orientation, *args, **kwargs)
        self.is_grasped = False # Whether a robot is grasping the cube or not.

    def render(self):
        super().render()
        # self.id = self.physics_client.create_box(A=self.size, B=self.size, H=self.size, mass=self.mass) 
        self.physics_client.geometry_objects.update({self.id : {
                            'shape' : 'cube',
                            'position' : self.position[:2], 
                            'color' : self.color, 
                            'size' : self.size}})
    
    def initialize_state(self):
        super().initialize_state()
        self.physics_client.geometry_objects.update({self.id : {
                            'shape' : 'cube',
                            'position' : self.position[:2], 
                            'color' : self.color, 
                            'size' : self.size}})



    def step(self, world_dict):
        pass

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.is_grasped = False

