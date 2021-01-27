import xml.etree.cElementTree as ET
from spike_swarm_sim.objects import WorldObject3D

class Wall(WorldObject3D):
    #TODO Meter H y W variables.
    def __init__(self, *args, height=2, width=20, **kwargs):
        self.height = height
        self.width = width
        self.resize_wall()
        super(Wall, self).__init__('wall', *args, static=True,\
            controller=None, tangible=True, luminous=False, **kwargs)
    
    def add_physics(self, physics_client):
        self.resize_wall()
        super().add_physics(physics_client) #! NOT WORKING

    def resize_wall(self):
        tree = ET.parse("spike_swarm_sim/objects/urdf/wall.urdf")
        root = tree.getroot()
        # aa = root.get('link').get('link')
        root.findall(".//link/visual/geometry/box")[0].attrib['size'] = '{} {} 2'.format(self.width, self.height)
        root.findall(".//link/collision/geometry/box")[0].attrib['size'] = '{} {} 2'.format(self.width, self.height)
        tree.write(open("spike_swarm_sim/objects/urdf/wall.urdf", 'wb'))

    def reset(self):
        pass