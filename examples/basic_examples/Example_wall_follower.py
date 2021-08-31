import numpy as np
from spike_swarm_sim.globals import global_states
from spike_swarm_sim.world import SquareArena, CircularArena, CustomWorld
from spike_swarm_sim.objects import Epuck
from spike_swarm_sim.controllers import RobotController, BasicObstacleAvoider

""" EXAMPLE DESCRIPTION:
"""


class WallFollower(RobotController):
    def __init__(self, *args, **kwargs):
        super(WallFollower, self).__init__(*args, **kwargs)

    def step(self, state, reward=0.0):
        st_ds = state['distance_sensor']
        sens = 0.15
        
        action = [0, 0]
        print(st_ds[7])
        if st_ds[7] > sens:
            action = [-1, 1]
        elif st_ds[5] > sens:
            action = [1, 1]
        elif all(st_ds[[5, 7]] < sens):
            action = [1, -1]
            
        return {'joint_velocity_actuator' : np.array(action)}

global_states.set_states(render=True, debug=True)

world = CustomWorld(map_file='complex_maze_1/complex_maze_1')
# world = SquareArena()

ctlr = WallFollower()
ctlr.add_sensor("distance_sensor", {"n_sectors" : 8, "range" : 1.5})
ctlr.add_actuator("joint_velocity_actuator",  {"joint_ids" : [0, 1], "max_velocity" : 5})
ent = Epuck([1,-14.5,0], [0, 0, 1.57], controller=ctlr)
world.register_entity('swarm_0', ent, group='swarm')

world.connect()
world.reset()
world.set_camera_focus(ent, distance=5.0)
while(True): 
    state, action = world.step()
