import numpy as np            
from matplotlib import colors
import matplotlib.pyplot as plot
from spike_swarm_sim.objects import WorldObject
from spike_swarm_sim.register import world_object_registry

class IsotropicEmitter(WorldObject):
    def __init__(self, pos, color='red', range=150, static=False, controller=None):
        super(IsotropicEmitter, self).__init__(pos=pos,
            static=static, tangible=False, luminous=True, controller=controller,
            shape='circ',)
        self.controller = controller
        self.range = range
        self.color = color
        self.reset()
    
    def step(self, world_dict):
        if self.controllable:
            if type(self.controller).__name__ == 'PreyController':
                robot_pos = [robot.pos for robot in world_dict]
                self.pos = self.controller.step(self.pos, robot_pos)
            else:
                self.pos = self.controller.step(self.pos)
        return (0, 0)

    def reset(self):
        if self.controller is not None:
            self.controller.reset()

    def initialize_render(self, canvas):
        x, y = tuple(self.pos)
        coverage_shadow_id = canvas.create_oval(x-10, y-10, x + 10, y + 10, fill=self.color)
        render_id = canvas.create_oval(x-10, y-10, x + 10, y + 10, fill=None)
        self.render_dict = {
            'body' : render_id,
            'shadow' : coverage_shadow_id,
        }
        return canvas 

    def render(self, canvas):
        x, y = tuple(self.pos)
        canvas.coords(self.render_dict['body'],
                        x-10, y-10,
                        x + 10, y + 10)
        canvas.itemconfig(self.render_dict['body'], fill=self.color)
        # #coverage shadow
        canvas.coords(self.render_dict['shadow'],
                x-10-self.range, y-10-self.range,
                x + 10+self.range, y + 10+self.range)
        canvas.itemconfig(self.render_dict['shadow'], fill=self.color)
        canvas.tag_lower(self.render_dict['body'])
        canvas.tag_lower(self.render_dict['shadow'])
        return canvas

@world_object_registry(name='light_source')
class LightSource(IsotropicEmitter):
    def __init__(self, *args, **kwargs):
        super(LightSource, self).__init__(*args, **kwargs)

@world_object_registry(name='food_area')
class FoodArea(IsotropicEmitter):
    def __init__(self, *args, **kwargs):
        super(FoodArea, self).__init__(*args, color='green', **kwargs)

@world_object_registry(name='nest')
class Nest(IsotropicEmitter):
    def __init__(self, *args, **kwargs):
        super(Nest, self).__init__(*args, color='black', **kwargs)
        self._food_items = 0
    
    def reset(self):
        super().reset()
        self._food_items = 0

    @property
    def food_items(self):
        return self._food_items
    
    def increase_food(self):
        self._food_items += 1
    
    def decrease_food(self):
        if self._food_items > 0:
            self._food_items = 1
    
    def initialize_render(self, canvas):
        x, y = tuple(self.pos)
        coverage_shadow_id = canvas.create_oval(x-10, y-10, x + 10, y + 10, fill=self.color)
        render_id = canvas.create_oval(x-10, y-10, x + 10, y + 10, fill=None)
        self.render_dict = {
            'body' : render_id,
            'shadow' : coverage_shadow_id,
        }
        return canvas 

    def render(self, canvas):
        x, y = tuple(self.pos)
        canvas.coords(self.render_dict['body'],
                        x-10, y-10,
                        x + 10, y + 10)
        canvas.itemconfig(self.render_dict['body'], fill=self.color)
        #coverage shadow
        canvas.coords(self.render_dict['shadow'],
                x-10-self.range, y-10-self.range,
                x + 10+self.range, y + 10+self.range)
        canvas.itemconfig(self.render_dict['shadow'], fill=self.color)
        canvas.tag_lower(self.render_dict['body'])
        canvas.tag_lower(self.render_dict['shadow'])
        return canvas


# #! TO BE MOVED 

# class FoodObserver:
#     def __init__(self, food_droppers, food_depots):
#         self.food_droppers = food_droppers
#         self.food_depots = food_depots

#     def notify_depots(self):
#         pass

#     def add_dropper(self, food_dropper):
#         self.food_droppers.append(food_dropper)