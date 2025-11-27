from matplotlib import colors
from mereli.objects import WorldObject
from mereli.register import world_object_registry

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
    def __init__(self, position, orientation, *args, color='red', is_on = True, range=1., z_offset=0., **kwargs):
        super(LightSource, self).__init__('entities/light_source/light', position, orientation, z_offset=z_offset,\
                        static=False, luminous=True, tangible=False, *args, **kwargs)
        self.range = range
        self.color = color
        self.scaling = 0.25
        self.is_on = is_on
        # self.reset()
    def render(self):
        super().render()
        self.physics_client.engine.setCollisionFilterGroupMask(self.id, -1, 0b00, 0b00)

    def step(self, neighborhood):
        """ Step method of the light source. Even though lights are not technically controlled, they can 
        have a 'virtual controller' for allowing custom behaviours such as mobile lights or preys in the 
        predator and prey game. The controller can be implemented just as in the robots. 
        """
        if self.controllable:
            if type(self.controller).__name__ == 'PreyController':
                robot_pos = [robot.position for robot in neighborhood]
                self.position = self.controller.step(self.position, robot_pos)
            else:
                self.position = self.controller.step(self.position)
        return (0, 0)

    def change_color(self, new_color):
        self.color = new_color
        self.physics_client.set_color(self.id, -1, self.color, opacity=1)

    def switch(self):
        if self.is_on:
            self.turn_off()
        else:
            self.turn_on()
        # self.is_on = not self.is_on
        # self.physics_client.set_color(self.id, -1, self.color, opacity=(.3, 1)[self.is_on])

    
    def turn_on(self):
        self.is_on = True 
        self.physics_client.set_color(self.id, -1, self.color, opacity=1)
        self.physics_client.luminous_objects[self.id]['is_on'] = True

    def turn_off(self):
        self.is_on = False
        self.physics_client.set_color(self.id, -1, self.color, opacity=.3)
        self.physics_client.luminous_objects[self.id]['is_on'] = False 

    def reset(self, seed=None):
        super().reset(seed=seed)
        if self.is_on:
            self.turn_on()
        else:
            self.turn_off()
        if self.controller is not None:
            self.controller.reset()
        # self.physics_client.get_contact_points(2, ghost_ids=[-1])
        # __import__('pdb').set_trace()
