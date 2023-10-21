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
    def __init__(self, position, orientation, *args, color='blue', mass=1., side_len=0.3, **kwargs):
        position = list(position)
        position[-1] = side_len / 2 - 0.1
        super(Cube, self).__init__('entities/cube/cube', position, orientation,\
                        *args, **kwargs)
        self.color = color
        self.mass = mass
        self.scaling = side_len
        self.is_grasped = False # Whether a robot is grasping the cube or not.

    def step(self, world_dict):
        pass

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.is_grasped = False

