from spike_swarm_sim.register import actuator_registry

@actuator_registry(name='led_actuator')
class LedActuator:
    """ LED actuator that turns on or off the LED depending on 
    the action. """
    def __init__(self):
        self.on = 0
    def step(self, action):
        self.on = action