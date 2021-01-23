from spike_swarm_sim.register import actuator_registry
from .base_actuator import Actuator

@actuator_registry(name='led_actuator')
class LedActuator(Actuator):
    """ LED actuator that turns on or off the LED depending on 
    the action. """
    def __init__(self, *args, **kwargs):
        super(LedActuator, self).__init__(*args, **kwargs)
        self.on = 0

    def step(self, action):
        self.on = action