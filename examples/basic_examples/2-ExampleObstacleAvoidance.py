import time
import numpy as np
from mereli.globals import global_states
from mereli import SquareArena, CircularArena, CustomWorld
from mereli.physics_engines import PybulletEngine
from mereli.objects import Epuck, LightSource
from mereli.controllers import BasicObstacleAvoider
from mereli.utils.initializers import FixedInitializer, RandomUniformInitializer


global_states.set_states(render=True, debug=True)
#* Number of robots
n_robots = 5
#* Create physics engine and world
world = SquareArena(PybulletEngine(dt=0.04, T_control=0.04), height=3, width=3)

#* Position and orientation initializers
ini_ori = RandomUniformInitializer(n_robots, low=0, high=6.28, size=1, engine='3D', variable='orientations')
ini_pos = RandomUniformInitializer(n_robots, low=[-2, -2], high=[2, 2], size=2, engine='3D',  variable='positions')
world.set_initializer('swarm', ini_pos, initializer_ori=ini_ori)
#* Loop for creating and initalizing the robots.
for i, (pos, ori) in enumerate(zip(ini_pos(), ini_ori())):
    ctlr = BasicObstacleAvoider()
    ctlr.add_sensor("distance_sensor", {"n_sectors" : 8, "range" : 0.7})
    ctlr.add_actuator("joint_velocity_actuator",  {"joint_ids" : [0, 1], "max_velocity" : 4})
    ctlr.add_actuator("led_actuator",  {})
    ent = Epuck(pos, ori, controller=ctlr)
    world.register_entity('swarm_' + str(i), ent, group='swarm')

#* Simulation loop
with world:
    world.reset()
    t0 = time.time()
    for _ in range(1000):
        state, action = world.step()
print('\nSIMULATION DURATION: ', np.round(time.time() - t0, 4))