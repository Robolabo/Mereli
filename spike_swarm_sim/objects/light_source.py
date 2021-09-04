from spike_swarm_sim.objects import WorldObject
from spike_swarm_sim.register import world_object_registry

@world_object_registry(name='light_source')
class LightSource(WorldObject):
    """ Class of the light source entity. It is an intangible entity that isotropically emits 
    light of a certain color and up to a defined range. Even though they are represented as 3D or 2D 
    balls, it is just displayed in that way for visualization purposes. 
    They are actually point lights without mass. The emitted light can be sensed by robots through 
    their ``light_sensors``.

    :param str color: color of the light.
    :param float range: coverage range of the light.
    :param int z_offset: offset of the z-axis position of the point light.  
    """
    def __init__(self, position, orientation, *args, color='red', range=1., z_offset=0., **kwargs):
        super(LightSource, self).__init__('entities/light_source/light', position, orientation, z_offset=z_offset,\
                        static=False, luminous=True, tangible=False, *args, **kwargs)
        self.range = range
        self.color = color
        self.reset()

    def step(self, neighborhood):
        """ Step method of the light sou\rce. Even though lights are not technically controlled, they can 
        have a 'virtual controller' for allowing custom behaviours such as mobile lights or preys in the 
        predator and prey game. The controller can be implemented just as in the robots. 
        
        :param list neighborhood: list filled with the neighboring entities. 
        """
        if self.controllable:
            if type(self.controller).__name__ == 'PreyController':
                robot_pos = [robot.position for robot in neighborhood]
                self.position = self.controller.step(self.position, robot_pos)
            else:
                self.position = self.controller.step(self.position)
        return (0, 0)

    def reset(self, seed=None):
        if self.controller is not None:
            self.controller.reset()