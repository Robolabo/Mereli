import numpy as np
from spike_swarm_sim.controllers import RobotController
from spike_swarm_sim.register import controller_registry


@controller_registry(name='random_walk')
class RandomMovementController(RobotController):
    def __init__(self, *args, **kwargs):
        super(RandomMovementController, self).__init__(*args, **kwargs)
        
    def step(self, state, reward=0.0):
        return {'wheel_actuator' : np.r_[-.5, -.5]} # np.random.choice([-.1, 1], size=2)}