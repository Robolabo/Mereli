import numpy as np
from spike_swarm_sim.controllers import RobotController
from spike_swarm_sim.register import controller_registry

@controller_registry(name='Braitenberg2')
class Braitenberg2Controller(RobotController):
    def __init__(self, *args, **kwargs):
        super(Braitenberg2Controller, self).__init__(*args, **kwargs)

    def step(self, state, reward=0.0):
        action = np.zeros(2)
        action[0] = np.max(state['red_light_sensor'][[4, 5, 6, 7]])
        action[1] = np.max(state['red_light_sensor'][[0, 1, 2, 3]])
        if np.max(state['red_light_sensor']) == 0.0: # 
            action = np.array([.2, .2])
        return {'joint_velocity_actuator' : 2*np.clip(action, a_min=-1, a_max=1)}


@controller_registry(name='Braitenberg2b')
class Braitenberg2bController(RobotController):
    def __init__(self, *args, **kwargs):
        super(Braitenberg2bController, self).__init__(*args, **kwargs)

    def step(self, state, reward=0.0):
        action = np.zeros(2)
        action[0] = np.max(state['red_light_sensor'][[0, 1, 2, 3]])
        action[1] = np.max(state['red_light_sensor'][[4, 5, 6, 7]])
        if np.max(state['red_light_sensor']) == 0.0:
            action = np.array([1., 1.])
        return {'joint_velocity_actuator' : np.clip(2 * action, a_min=-1, a_max=1)}