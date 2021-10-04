from mereli.objects import WorldObject
from mereli.register import world_object_registry

@world_object_registry(name='ground_area')
class GroundArea(WorldObject):
    """ Circular ground area radius and color. It is an intangible objects that robots 
    can detect through their ground_sensor, provided that it is underneath the robot body.
    
    **Reference Name**: ``ground_area``.

    :param float radius: radius [metres] of the ground area.
    :param str color: color of the ground area.
    :param float: total mass of the ball.
    """
    def __init__(self, position, orientation, *args, color='grey', radius=1., **kwargs):
        super(GroundArea, self).__init__('entities/ground_area/ground_area', position, orientation,\
                        *args, tangible=False, **kwargs)
        self.color = color
        self.radius = radius
        self.scaling = radius

    def step(self):
        pass

    def reset(self, seed=None):
        pass