from spike_swarm_sim.objects import WorldObject 
from spike_swarm_sim.register import world_object_registry

class Map(WorldObject):
    """ 

    :param position:
    :param float height: height in metres of the wall.
    :param float width: width in metres of the wall.   
    """
    def __init__(self, map_file, *args, **kwargs):
        super(Map, self).__init__(map_file, *args, static=True,\
            controller=None, tangible=True, luminous=False, **kwargs)
    
    def step(self):
        pass

    def reset(self, seed=None):
        pass