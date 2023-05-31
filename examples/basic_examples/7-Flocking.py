import numpy as np
from mereli import FlatWorld 
from mereli.physics_engines import PybulletEngine
from mereli.objects import Epuck
from mereli.utils.initializers import RandomUniformInitializer, RandomGraphInitializer
from mereli.controllers import RobotController

""" EXAMPLE DESCRIPTION:

"""

import pybullet as p #!
class FlockingController(RobotController):
    def __init__(self, *args, **kwargs):
        super(FlockingController, self).__init__(self, *args, **kwargs)
        self.add_sensor('distance_sensor', {'n_sectors': 8, 'range' : 2.0})
        self.add_actuator("joint_velocity_actuator",  {"joint_ids" : [0, 1], "max_velocity" : 4})
        self.eps_coll = 0.1
        self.eps_coh = 0.6 
        
    def step_collision_avoidance(self, st_ds):
        if any(st_ds[[0,1]] > self.eps_coll):
            # print('Turn Left')
            action = np.array([1., -1])
            avoid = True
        elif any(st_ds[[6,7]] > self.eps_coll):
            # print('Turn Right')
            action = np.array([-1, 1.])
            avoid = True
        else:
            # print('GO straight over')
            action = np.array([1,1])
            avoid = False
        return avoid, action

    def step_cohesion(self, st_ds):
        if any(st_ds[[0,1,2,3]] > self.eps_coh):
            # print('Turn Left')
            action = np.array([-1., 1])
        elif any(st_ds[[6,7, 4, 5]] > self.eps_coh):
            # print('Turn Right')
            action = np.array([1, -1.])
        else:
            # print('GO straight over')
            action = np.array([1, 1])
            action = np.random.uniform(low=0.1, high=1, size=2)
        return  action / 2.5

    def step(self, state, reward=0.0):
        st_ds = state['distance_sensor']
        avoid, action1 = self.step_collision_avoidance(st_ds)
        if not avoid:
            # Cohesion
            action = self.step_cohesion(st_ds)
        else:
            action = action1
        return {"joint_velocity_actuator" : action}
        

n_robots = 20 # Number of robots.

# Create physics engine with 0.02sec of discretization.
phy_engine = PybulletEngine(dt=0.01)
# Create empty world with physics Engine
world = FlatWorld(phy_engine)


ini_ori = RandomUniformInitializer(n_robots, low=0, high=6.28, size=1, engine='3D', variable='orientations')
ini_pos = RandomGraphInitializer(n_robots, max_rad=1, initial_pos=[0, 0], engine='3D',  variable='positions')
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

        

