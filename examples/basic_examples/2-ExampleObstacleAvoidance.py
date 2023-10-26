import time
import numpy as np
from mereli.globals import global_states
from mereli import SquareArena, CircularArena, CustomWorld
from mereli.physics_engines import PybulletEngine
from mereli.objects import Epuck, LightSource
from mereli.controllers import BasicObstacleAvoider
from mereli.utils.initializers import FixedInitializer, RandomUniformInitializer


global_states.set_states(render=True, debug=False)

n_robots = 10 
# Create physics engine with 0.02sec of discretization and a period the robot
# control loop of 0.14sec.
world = SquareArena(PybulletEngine(dt=0.05, T_control=0.05), height=7, width=7)
for i in range(n_robots):
    ctlr = BasicObstacleAvoider()
    ctlr.add_sensor("distance_sensor", {"n_sectors" : 8, "range" : 0.8})
    ctlr.add_actuator("joint_velocity_actuator",  {"joint_ids" : [0, 1], "max_velocity" : 6})
    # ctlr.add_actuator("led_actuator",  {})
    robot = Epuck([0,0,0], [0,0,0], controller=ctlr)
    robot.pos_init_method = {"type" : "random", "low" : [-2, -2], "high" : [2,2]}
    robot.ori_init_method = 'random' 
    world.register_entity('swarm_' + str(i), robot, group='swarm')

with world:
    world.reset()
    t0 = time.time()
    for _ in range(10000):
        state, action = world.step()
print('\nSIMULATION DURATION: ', np.round(time.time() - t0, 4))
