# import numpy as np
# from ../.. import Flatworld 
# from mereli.physics_engines import PybulletEngine
# from mereli.objects import Epuck



# USE_API = False  # Whether to add entities using world API or config dict. 

# # Create physics engine with 0.02sec of discretization.
# phy_engine = PybulletEngine(dt=0.01)
# # Create empty world with physics Engine
# world = Flatworld(phy_engine)

# position = [0,0,0]
# orientation =0.0 
# epk = Epuck(position, [0,0,orientation], controller=controller)
# world.register_entity('epuck0', epk, group='swarm')

# with world:
#     world.reset()
#     for t in range(timesteps):
#         world.step()





