from mereli.objects import WorldObject
from mereli.register import world_object_registry

@world_object_registry(name='ball')
class Ball(WorldObject):
    """ Ball object of a certain radius, color and mass. If 2D then it is a
    circle (TODO) and if 3D it is a sphere.
    
    **Reference Name**: ``ball``

    :param float radius: radius [metres] of the ball.
    :param str color: color of the ball.
    :param float mass: total mass of the ball.
    """
    def __init__(self, position, orientation, *args, 
                    radius=0.3, color='red', mass=1., **kwargs):
        super(Ball, self).__init__('entities/ball/ball', position, orientation,\
                        *args, **kwargs)
        self.color = color
        self.mass = mass
        self.radius = radius
        self.scaling = radius
 
    def step(self, world_dict):
        pass

    def reset(self, seed=None):
        pass