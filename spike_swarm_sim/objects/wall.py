from spike_swarm_sim.objects import WorldObject3D

class Wall(WorldObject3D):
    #TODO Meter H y W variables.
    def __init__(self, position, orientation):
        super(Wall, self).__init__('wall', position, orientation, static=True,\
            controller=None, tangible=True, luminous=False)

    def reset(self):
        pass