import numpy as np
import matplotlib.pyplot as plt
from spike_swarm_sim.globals import global_states
from spike_swarm_sim.world import SquareArena, CircularArena
from spike_swarm_sim.objects import Epuck
from spike_swarm_sim.controllers import BasicObstacleAvoider


global_states.set_states(render=True)
world = SquareArena(height=5, width=5)
# world = CircularArena()

ctlr = BasicObstacleAvoider()
ctlr.add_sensor("distance_sensor", {"n_sectors" : 8, "range" : 1})
ctlr.add_sensor("joint_velocity_sensor", {"joints" : [0, 1]})
ctlr.add_actuator("joint_velocity_actuator",  {"joint_ids" : [0, 1], "max_velocity" : 5})
ent = Epuck([0,0,0], [0,0,0], controller=ctlr)
world.register_entity('swarm_0', ent, group='swarm')

world.connect()
world.reset()
odom_positions = [np.zeros(2)]
odom_orients = [0.0]
r, l = 0.05, .1
for t in range(2000):
    state, action = world.step()
    # print(state[0]['distance_sensor'])
    wh_vel = state[0]['joint_velocity_sensor']
    last_pos = odom_positions[-1]
    last_th = odom_orients[-1]
    
    delta_pos = np.array([
        [np.cos(last_th), -np.sin(last_th)],
        [np.sin(last_th), np.cos(last_th)]
    ]).dot(np.array([r*np.sum(wh_vel)/20, 0]))
    delta_th = r * (wh_vel[0] - wh_vel[1])/(20*l) 
    odom_positions.append(odom_positions[-1] + delta_pos)
    odom_orients.append(odom_orients[-1] + delta_th)

odom_positions = np.vstack(odom_positions)
plt.plot(odom_positions[:,0], odom_positions[:,1])
plt.ylim([-10, 10])
plt.show()
import pdb; pdb.set_trace()