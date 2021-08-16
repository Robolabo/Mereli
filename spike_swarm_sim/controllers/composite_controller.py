import numpy as np
from spike_swarm_sim.controllers import RobotController
from spike_swarm_sim.register import controller_registry, controllers
from spike_swarm_sim.utils import flatten_dict, key_of, increase_time, RegexpDict

@controller_registry(name='cascade_controller')
class CascadeController(RobotController):

    def __init__(self, *args, **kwargs):
        super(CascadeController, self).__init__(*args, **kwargs)
    

    def step(self, state, reward=0.0):
        pass


    def reset(self):
        pass