import numpy as np

#TODO

# def modify_cone_generatix(obj_file, generatrix):

#     with open(file, "r") as f:
#         lines = f.readlines()
#         vertices = []
#         for line in lines:
#             elems = line.rstrip('\n').split(' ')
#             if elems[0] == 'v':
#                 vert = np.array(elems[1:]).astype(float)
#                 vertices.append(vert)
#         vertices = np.vstack(vertices)
#         old_rads = np.unique(np.sqrt(vertices[:,0] ** 2 + vertices[:,2] ** 2).round(3))
#         assert len(old_rads) == 2
#         scaling = self.radius / old_rads.min()
#         vertices[:,[0,2]] *= scaling
#     with open(file, "r+") as f:
#         lines = f.readlines()
#         f.seek(0)
#         vert_iter = iter(vertices)
#         lines = ['v {} {} {}\n'.format(*tuple(next(vert_iter))) if line.split(' ')[0] == 'v' else line for line in lines ]
#         f.writelines(lines)
#         f.truncate()