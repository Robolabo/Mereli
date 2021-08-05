import os
import json
import xml.etree.cElementTree as ET
from spike_swarm_sim.objects import WorldObject 
from spike_swarm_sim.register import world_object_registry

@world_object_registry(name='wall')
class Wall(WorldObject):
    """ Class of wall objects, with parametrizable height and width. The mass is large enough 
    so that robots cannot move them. Its main use is to constrain the environment arena, but 
    they can be also instantiated as regular walls in arbitrary positions.

    :param position:
    :param float height: height in metres of the wall.
    :param float width: width in metres of the wall.   
    
    """
    def __init__(self, *args, height=2, width=20, **kwargs):
        self.height = height
        self.width = width
        # Tmp solution
        self.resize_wall()
        self.resize_wall2D()
        super(Wall, self).__init__('tmp/wall_{}x{}x2'.format(width, height), *args, static=True,\
            controller=None, tangible=True, luminous=False, **kwargs)
    
    def resize_wall(self):
        if not os.path.isfile("spike_swarm_sim/objects/urdf/tmp/wall_{}x{}x2.urdf".format(self.width, self.height)): 
            tree = ET.parse("spike_swarm_sim/objects/urdf/wall.urdf")
            root = tree.getroot()
            # aa = root.get('link').get('link')
            root.findall(".//link/visual/geometry/box")[0].attrib['size'] = '{} {} 2'.format(self.width, self.height)
            root.findall(".//link/collision/geometry/box")[0].attrib['size'] = '{} {} 2'.format(self.width, self.height)
            tree.write(open("spike_swarm_sim/objects/urdf/tmp/wall_{}x{}x2.urdf".format(self.width, self.height), 'wb'))
    
    def resize_wall2D(self):
        file_tmp = "spike_swarm_sim/objects/urdf/tmp/wall_{}x{}x2.json".format(self.width, self.height)
        if not os.path.isfile(file_tmp): 
            with open("spike_swarm_sim/objects/urdf/wall.json") as json_file:
                obj_dict = json.load(json_file)
            vertices = [[-100*(self.width/2), 100*(-self.height/2)], [100*(self.width/2), 100*(-self.height/2)], 
                        [100*(self.width/2), 100*(self.height/2)], [100*(-self.width/2), 100*(self.height/2)]]
            obj_dict['links'][0]['moment']['params']['vertices'] = vertices
            obj_dict['shapes'][0]['params']['vertices'] = vertices
            with open(file_tmp, 'w') as outfile:
                json.dump(obj_dict, outfile)

    def step(self):
        pass

    def reset(self, seed=None):
        pass