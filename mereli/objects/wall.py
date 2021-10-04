import os
import json
import xml.etree.cElementTree as ET
from mereli.objects import WorldObject 
from mereli.register import world_object_registry

@world_object_registry(name='wall')
class Wall(WorldObject):
    """ Class of wall objects, with parametrizable height and width. The mass is large enough 
    so that robots cannot move them. Its main use is to constrain the environment arena, but 
    they can be also instantiated as regular walls in arbitrary positions.

    :param position:
    :param float height: height in metres of the wall.
    :param float width: width in metres of the wall.   
    
    """
    def __init__(self, *args, height=1, width=5, **kwargs):
        self.height = height
        self.width = width
        # Tmp solution
        self.resize_wall()
        self.resize_wall2D()
        super(Wall, self).__init__('tmp/wall_{}x{}x{}'.format(width, height, 1), *args, static=True,\
            controller=None, tangible=True, luminous=False, **kwargs)
    
    def resize_wall(self):
        if not os.path.isfile("mereli/models/tmp/wall_{}x{}x{}.urdf".format(self.width, self.height, 1)): 
            tree = ET.parse("mereli/models/entities/wall/wall.urdf")
            root = tree.getroot()
            # aa = root.get('link').get('link')
            root.findall(".//link/visual/geometry/box")[0].attrib['size'] = '{} {} 1'.format(self.width, self.height)
            root.findall(".//link/collision/geometry/box")[0].attrib['size'] = '{} {} 1'.format(self.width, self.height)
            tree.write(open("mereli/models/tmp/wall_{}x{}x{}.urdf".format(self.width, self.height,1), 'wb'))
    
    def resize_wall2D(self):
        file_tmp = "mereli/models/tmp/wall_{}x{}x2.json".format(self.width, self.height)
        if not os.path.isfile(file_tmp): 
            with open("mereli/models/entities/wall/wall.json") as json_file:
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