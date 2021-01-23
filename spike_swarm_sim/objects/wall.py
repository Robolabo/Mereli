
from spike_swarm_sim.objects import WorldObject

class Wall(WorldObject):
    def __init__(self, xpos, ypos, h, w):
        super(Wall, self).__init__(xpos=xpos, ypos=ypos, static=True, 
             controllable=False, tangible=True, luminous=False,
             bounding_shape='rect')
        self.h = h
        self.w = w

    def initialize_render(self, canvas):
        x, y = tuple(self.pos)
        canvas.create_rectangle(x - self.h/2, y- self.w/2,\
                        x + self.h/2, y + self.w/2, fill='red') 
        return canvas 

    def reset(self):
        pass