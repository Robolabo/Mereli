import numpy as np
from spike_swarm_sim import World
from spike_swarm_sim.physics_engines import PybulletEngine
from spike_swarm_sim.objects import Epuck
from spike_swarm_sim.utils.initializers import RandomUniformInitializer, RandomGraphInitializer
from spike_swarm_sim.controllers import RobotController

""" EXAMPLE DESCRIPTION:

"""

import pybullet as p #!
class FlockingController(RobotController):
    def __init__(self, *args, **kwargs):
        super(FlockingController, self).__init__(self, *args, **kwargs)
        self.add_sensor('distance_sensor', {'n_sectors': 8, 'range' : 2.0})
        self.add_actuator("joint_velocity_actuator",  {"joint_ids" : [0, 1], "max_velocity" : 4})
        
    
    def step(self, state, reward=0.0):
        ds_st = state['distance_sensor']
        print(ds_st)
        delta = 0.5

        phis = np.array([0.26179, 0.78539, 1.57079, 2.61799,  3.66519, 4.71238, 5.4977, 6.0213])
        V = np.array([[np.cos(phi), np.sin(phi)] for phi in phis])

        delta_pos = V.T.dot(ds_st - delta)
        delta_pos = delta_pos.tolist() + [0.]
        mod = np.linalg.norm(delta_pos)
        ang = np.array(delta_pos)[:2].dot([1, 0]) / mod
        if ang < 0 :
            action = mod * np.array([1, 1-ang])
        else:
            action = mod * np.array([1-ang, 1])
        return {"joint_velocity_actuator" : action/2}
        

n_robots = 3 # Number of robots.

# Create physics engine with 0.02sec of discretization.
phy_engine = PybulletEngine(dt=0.02)
# Create empty world with physics Engine
world = World(phy_engine)


ini_ori = RandomUniformInitializer(n_robots, low=0, high=6.28, size=1, engine='3D', variable='orientations')
ini_pos = RandomGraphInitializer(n_robots, max_rad=3, initial_pos=[0, 0], engine='3D',  variable='positions')
world.set_initializer('swarm', ini_pos, initializer_ori=ini_ori)
for i, (pos, ori) in enumerate(zip(ini_pos(), ini_ori())):
    robot = Epuck(pos, ori, controller=FlockingController())
    world.register_entity(f'swarm_{i}', robot, group='swarm')

# robot = Epuck([1,0,0], 0., controller=FlockingController())
# world.register_entity('swarm_1', robot, group='swarm')

with world:
    world.reset()
    for t in range(2000):
        st, ac = world.step()

        

