import time
import numpy as np
from spike_swarm_sim.globals import global_states
from spike_swarm_sim import CircularArena, Engine3D
from spike_swarm_sim.objects import Epuck, LightSource
from spike_swarm_sim.controllers import Braitenberg2Controller


# global_states.set_states(render=True)
# Create physics engine with 0.02sec of discretization and a period the robot
# control loop of 0.14sec.
phy_engine = Engine3D(dt=0.02, T_control=0.14)
world = CircularArena(phy_engine, radius=7)

ctlr = Braitenberg2Controller()
ctlr.add_sensor("light_sensor", {"n_sectors" : 8, "range" : 10})
ctlr.add_actuator("joint_velocity_actuator",  {"joint_ids" : [0, 1], "max_velocity" : 7})
ent = Epuck([0,0,0], [0,0,0], controller=ctlr)
world.register_entity('swarm_0', ent, group='swarm')

ls = LightSource([2,2,1.5], [0,0,0], color='red', range=5.)
world.register_entity('light_red', ls, group='light_sources')
# ls = LightSource([-2,2,1.5], [0,0,0], color='red', range=5.)
# world.register_entity('light_red2', ls, group='light_sources')
# ls = LightSource([0,-1,1], [0,0,0], color='red', range=10.)
# world.register_entity('light_red3', ls, group='light_sources')

world.connect()
world.reset()
t0 = time.time()
for _ in range(2000):
    state, action = world.step()
world.disconnect()
print('\nSIMULATION DURATION: ', np.round(time.time() - t0, 4))