import time
import numpy as np
from spike_swarm_sim.globals import global_states
from spike_swarm_sim.world import SquareArena, CircularArena, CustomWorld
from spike_swarm_sim.objects import Epuck, LightSource
from spike_swarm_sim.controllers import BasicObstacleAvoider
from spike_swarm_sim.utils.initializers import FixedInitializer, RandomUniformInitializer

""" EXAMPLE DESCRIPTION:
The experiments can be defined in three different ways. The first and most common one is via the command 
line execution of the main.py file with the reference to the corresponding config file previously designed 
and stored in spike_swarm_sim/config. This type of execution is not shown in this example. 
Besides there are two other manners of executing an experiment within a python program:
1) Config as dict: the experiment is defined as a python dict with the same fields as in a config file. This
dict is fed to a previously created world instance throughout the build_from_dict method.
2) API based: it uses the simulator API to create the experiment.

This example shows the basic use of both 1) and 2) ways of defining an experiment in a simple obstacle avoidance 
problem without optimization involved. 
With the variable USE_API you can switch between using 1) or 2).
"""

global_states.set_states(render=True, debug=True)
USE_API = True
n_robots = 5
# world = CustomWorld(map_file='simple_map_1/simple_map_1')
world = SquareArena(height=10, width=10)

if USE_API:
    ini_ori = RandomUniformInitializer(n_robots, low=0, high=6.28, size=1, engine='3D', variable='orientations')
    ini_pos = RandomUniformInitializer(n_robots, low=[-3, -3], high=[3, 3], size=2, engine='3D',  variable='positions')
    # ini_pos = FixedInitializer(n_robots, fixed_values=[[0, 1.5]], engine='3D',  variable='positions')
    # ini_ori = FixedInitializer(n_robots, fixed_values=[np.pi/2], engine='3D',  variable='orientations')
    world.set_initializer('swarm', ini_pos, initializer_ori=ini_ori)
    for i, (pos, ori) in enumerate(zip(ini_pos(), ini_ori())):
        ctlr = BasicObstacleAvoider()
        ctlr.add_sensor("distance_sensor", {"n_sectors" : 8, "range" : 1.5})
        # ctlr.add_sensor("camera", {})
        ctlr.add_actuator("joint_velocity_actuator",  {"joint_ids" : [0, 1], "max_velocity" : 13})
        # ctlr.add_sensors_from_dict({"distance_sensor" : {"n_sectors" : 4, "range" : 1}})
        # ctlr.add_actuators_from_dict({"joint_velocity_actuator" : {"joint_ids" : [0, 1], "max_velocity" : 13}})
        ent = Epuck(pos, ori, controller=ctlr)
        world.register_entity('swarm_' + str(i), ent, group='swarm')
    world.register_entity('ls' + str(i), LightSource([0,3,2], [0,0,0], color='red'), group='ls')    
else:
    world_cfg = {
        "engine" : "3D",
        "world_delay" : 1,
        "height": 10,
        "width":  10,
        "objects" : {
            "robotA" : {
                "type" : "robot",
                "num_instances" : n_robots,
                "controller" : "basic_obstable_avoider",
                "sensors" : {
                    "distance_sensor" : {"n_sectors" : 8, "range" : 3}
                },  
                "actuators" : {
                    "joint_velocity_actuator" : {"joint_ids" : [0, 1], "max_velocity" : 13}
                },
                "initializers" : {
                    "positions" : {"name" : "random_uniform",  "params" : {"low" : [-3, -3], "high" : [3, 3], "size" : 2}},
                    "orientations" : {"name" : "random_uniform",  "params" : {"low":0, "high" : 6.28, "size" : 1}}
                },
                "perturbations" : {
                },
                "params" : {"trainable" : True}
            }
        }
    }
    world.build_from_dict(world_cfg)


world.connect()
world.reset()
t0 = time.time()
for _ in range(1000):
    state, action = world.step()
    # import pdb; pdb.set_trace()
    # print(state[0]['distance_sensor'])
    # st1 = state[1]['distance_sensor']
    # import pdb; pdb.set_trace()
world.disconnect()
print('\nSIMULATION DURATION: ', np.round(time.time() - t0, 4))

