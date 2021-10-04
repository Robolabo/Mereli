import numpy as np
from mereli import SquareArena
from mereli.physics_engines import PybulletEngine
from mereli.objects import Epuck, LightSource, GroundArea, Ball
from mereli.utils.initializers import RandomUniformInitializer

""" EXAMPLE DESCRIPTION:
The experiments can be defined in three different ways. The first and most common one is via the command 
line execution of the main.py file with the reference to the corresponding config file previously designed 
and stored in mereli/config. This type of execution is not shown in this example. 
Besides there are two other manners of executing an experiment within a python program:
1) Config as dict: the experiment is defined as a python dict with the same fields as in a config file. This
dict is fed to a previously created world instance throughout the build_from_dict method.
2) API based: it uses the simulator API to create the experiment.

This example shows the basic use of both 1) and 2) ways of defining an experiment in a simple obstacle avoidance 
problem without optimization involved. 
With the variable USE_API you can switch between using 1) or 2).
"""

USE_API = False  # Whether to add entities using world API or config dict. 
n_robots = 5 # Number of robots.
n_balls = 3 # Number of small balls.

# Create physics engine with 0.02sec of discretization.
phy_engine = PybulletEngine(dt=0.02)
# Create empty world with physics Engine
world = SquareArena(phy_engine, height=10,  width=10)

if USE_API:
    # Create and add robots
    ini_ori = RandomUniformInitializer(n_robots, low=6.27, high=6.28, size=1, engine='3D', variable='orientations')
    ini_pos = RandomUniformInitializer(n_robots, low=[-3,-3], high=[3,1], size=2, engine='3D',  variable='positions')
    world.set_initializer('swarm', ini_pos, initializer_ori=ini_ori)
    for i, (pos, ori) in enumerate(zip(ini_pos(), ini_ori())):
        robot = Epuck(pos, ori, controller=None)
        world.register_entity('swarm_' + str(i), robot, group='swarm')

    # Create blue and yellow lights
    blue_ls = LightSource([0, 2, 1], 0, color='blue')
    world.register_entity('light_blue', blue_ls, group='lights')
    yellow_ls = LightSource([0, -2, 3], 0, color='yellow')
    world.register_entity('light_yellow', yellow_ls, group='lights')

    # Create ground area\
    grey_area = GroundArea([3.5,0,0], 0, radius=1, color='grey')
    world.register_entity('grey_area', grey_area, group='ground_areas')

    # Create balls
    ini_pos = RandomUniformInitializer(n_balls, low=[-3,1.3], high=[3,3], size=2, engine='3D',  variable='positions')
    world.set_initializer('balls', ini_pos, initializer_ori=None)
    for i, pos in enumerate(ini_pos()):
        ball = Ball(pos, np.zeros(3), radius=0.7, color='red')
        world.register_entity('ball_' + str(i), ball, group='balls')


else:
    world_cfg = {
        "engine" : "3D",
        "objects" : {
            "robotA" : {
                "type" : "epuck",
                "num_instances" : n_robots,
                "controller" : None, # In the JSON file, use null instead.
                "sensors" : {},  
                "actuators" : {},
                "initializers" : {
                    "positions" : {"name" : "random_uniform",  "params" : {"low" : [-3, -3], "high" : [3, 1], "size" : 2}},
                    "orientations" : {"name" : "random_uniform",  "params" : {"low":0, "high" : 6.28, "size" : 1}}
                },
                "perturbations" : {},
                "params" : {"trainable" : True}
            },
            "balls" : {
                "type" : "ball",
                "num_instances" : n_balls,
                "initializers" : {
                    "positions" : {"name" : "random_uniform",  "params" : {"low" : [-3, 1.3], "high" : [3, 3], "size" : 2}},
                },
                "params" : {"color" : 'red', 'radius' : .7}
            },
            "light_blue" : {
                "type" : "light_source",
                "num_instances" : 1,
                "initializers" : {
                    "positions" : {"name" : "fixed",  "params" : {"fixed_values" : [[0,2,0]]}},
                },
                "params" : {"color" : 'blue'}
            },
            "light_yellow" : {
                "type" : "light_source",
                "num_instances" : 1,
                "initializers" : {
                    "positions" : {"name" : "fixed",  "params" : {"fixed_values" : [[0,-2,0]]}},
                },
                "params" : {"color" : 'yellow'}
            },
            "ground_areas" : {
                "type" : "ground_area",
                "num_instances" : 1,
                "initializers" : {
                    "positions" : {"name" : "fixed",  "params" : {"fixed_values" : [[3.5,0,0]]}},
                },
                "params" : {"color" : 'grey', 'radius' : 1}
            },
        }
    }
    world.build_from_dict(world_cfg)


# Before executing the simulation for the first time, call the connect method.
world.connect()
# Simulate 3 independent times. It is important to reset the world at the beggining
# of every trial. Here we can clearly see the importance of entity initializers, as 
# they are run at the beggining of every trial.
for trial in range(3):
    print('STARTING TRIAL ', trial)
    world.reset()
    for t in range(200):
        state, action = world.step()

world.disconnect()