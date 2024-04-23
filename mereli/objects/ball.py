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
                    radius=0.3, color='red', mass=0.1, **kwargs):
        super(Ball, self).__init__('entities/ball/ball.urdf', position, orientation,\
                        *args, **kwargs)
        self.color = color
        self.mass = mass
        self.radius = radius
        self.scaling = radius

 
    def step(self, world_dict):
        pass

    def render(self):
        super().render()
        # self.id = self.physics_client.create_box(A=self.size, B=self.size, H=self.size, mass=self.mass) 
        self.physics_client.geometry_objects.update({self.id : {
                            'shape' : 'ball',
                            'position' : self.position[:2], 
                            'color' : self.color, 
                            'size' : self.radius}})

    
    def initialize_state(self):
        super().initialize_state()
        self.physics_client.geometry_objects.update({self.id : {
                            'shape' : 'ball',
                            'position' : self.position[:2], 
                            'color' : self.color, 
                            'size' : self.radius}})

    def reset(self, seed=None):
        super().reset(seed=seed)
        # self.physics_client.change_dynamics(self.id, mass=10, restitution=1.0, linearDamping=0, angularDamping=0, rollingFriction=0.001, spinningFriction=0.001)

@world_object_registry(name='pyramid')
class Pyramid(WorldObject):
    def __init__(self, position, orientation, *args, 
                    size=0.1, color='red', mass=0.1, **kwargs):
        super(Pyramid, self).__init__('entities/pyramid/pyramid.urdf', position, orientation,\
                        *args, **kwargs)
        self.color = color
        self.mass = mass
        self.size =size  
        self.scaling =size 
        position[-1] = self.scaling / 2 - .1

 
    def step(self, world_dict):
        pass

    def render(self):
        super().render()
        # self.id = self.physics_client.create_box(A=self.size, B=self.size, H=self.size, mass=self.mass) 
        self.physics_client.geometry_objects.update({self.id : {
                            'shape' : 'pyramid',
                            'position' : self.position[:2], 
                            'color' : self.color, 
                            'size' : self.size}})

    
    def initialize_state(self):
        super().initialize_state()
        self.physics_client.geometry_objects.update({self.id : {
                            'shape' : 'pyramid',
                            'position' : self.position[:2], 
                            'color' : self.color, 
                            'size' : self.size}})

    def reset(self, seed=None):
        super().reset(seed=seed)
        # self.physics_client.change_dynamics(self.id, mass=10, restitution=1.0, linearDamping=0, angularDamping=0, rollingFriction=0.001, spinningFriction=0.001)
