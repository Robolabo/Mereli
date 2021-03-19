import numpy as np            
from matplotlib import colors
import matplotlib.pyplot as plot
import pybullet as p
from spike_swarm_sim.objects import WorldObject2D, WorldObject3D
from spike_swarm_sim.register import world_object_registry



@world_object_registry(name='ball')
class Ball(WorldObject3D):
    def __init__(self, position, orientation, *args, color='red', range=1., **kwargs):
        super(Ball, self).__init__('ball', position, orientation,\
                        *args, **kwargs)
        self.color = color

    def add_physics(self, physics_client):
        super().add_physics(physics_client)
        color = list(colors.to_rgb(self.color)) + [1.]
        # import pdb; pdb.set_trace()
        p.changeVisualShape(self.id, -1, rgbaColor=color, physicsClientId=physics_client)
    
    def step(self, world_dict):
        pass

    def reset(self):
        pass


@world_object_registry(name='cube')
class Cube(WorldObject3D):
    def __init__(self, position, orientation, *args, color='red', range=1., **kwargs):
        super(Cube, self).__init__('cube', position, orientation,\
                        *args, **kwargs)
        self.color = color

    def add_physics(self, physics_client):
        super().add_physics(physics_client)
        color = list(colors.to_rgb(self.color)) + [1.]
        # import pdb; pdb.set_trace()
        p.changeVisualShape(self.id, -1, rgbaColor=color, physicsClientId=physics_client)
    
    def step(self, world_dict):
        pass

    def reset(self):
        pass

@world_object_registry(name='ground_area')
class GroundArea(WorldObject3D):
    def __init__(self, position, orientation, *args, color='red', radius=1., **kwargs):
        super(GroundArea, self).__init__('ground_area', position, orientation,\
                        *args, **kwargs)
        self.color = color
        self.radius = radius

    def add_physics(self, physics_client):
        super().add_physics(physics_client)
        color = list(colors.to_rgb(self.color)) + [1.]
        p.changeVisualShape(self.id, -1, rgbaColor=color, physicsClientId=physics_client)

    def reset(self):
        pass