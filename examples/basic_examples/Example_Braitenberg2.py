from spike_swarm_sim.globals import global_states
from spike_swarm_sim.world import CircularArena
from spike_swarm_sim.objects import Epuck, LightSource
from spike_swarm_sim.controllers import Braitenberg2bController, Braitenberg2Controller


# global_states.set_states(render=True)
world = CircularArena(radius=4)

ctlr = Braitenberg2Controller()
ctlr.add_sensor("red_light_sensor", {"n_sectors" : 8, "range" : 10})
ctlr.add_actuator("joint_velocity_actuator",  {"joint_ids" : [0, 1], "max_velocity" : 13})
ent = Epuck([0,0,0], [0,0,0], controller=ctlr)
world.register_entity('swarm_0', ent, group='swarm')

ls = LightSource([1,1,0], [0,0,0], color='red', range=10.)
world.register_entity('light_red', ls, group='light_sources')
ls = LightSource([-1,1,0], [0,0,0], color='red', range=10.)
world.register_entity('light_red2', ls, group='light_sources')
# ls = LightSource([0,-1,1], [0,0,0], color='red', range=10.)
# world.register_entity('light_red3', ls, group='light_sources')

world.connect()
world.reset()
while(True):
    state, action = world.step()
