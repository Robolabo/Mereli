import numpy as np
from matplotlib import colors
import pybullet as p
from spike_swarm_sim.register import actuator_registry
from .base_actuator import Actuator

@actuator_registry(name='led_actuator')
class LedActuator(Actuator):
    """ LED actuator that turns on or off the LED depending on 
    the action. """
    def __init__(self, *args, color_on='blue', color_off='white', **kwargs):
        super(LedActuator, self).__init__(*args, **kwargs)
        self.color_on = [1,0,0]
        self.color_off = [1,1,1]
        self.color_fault = [1,0,0]
        self.on = 0
        self.fault = False
        self.prev_action = None

    def step(self, action):
        # if len(action) == 1:TODO
        if action is None:
            return
        action = action if not self.fault else np.zeros_like(action)
        if self.prev_action is None:
            self.prev_action = np.zeros_like(action)

        for i, (led_ac, prev_led_ac) in enumerate(zip(action, self.prev_action)):
            if led_ac == prev_led_ac: # Used to optimize code (quite slow otherwise)
                continue
            color = (self.color_on if led_ac else self.color_off) if not self.fault else self.color_fault
            if 0 < led_ac < 1:
                color = [led_ac, 0, 0]
            led_idx = self.physics_client.get_actuator_position(self.actuator_owner.id, 'led_actuator', sector=i)[1]
            self.actuator_owner.physics_client.set_color(self.actuator_owner.id, led_idx, color, opacity=0.6)

        self.prev_action = action
        
    @property
    def physics_client(self):
        return self.actuator_owner.physics_client

    def reset(self):
        self.on = 0
        self.fault = False
        self.prev_action = None

