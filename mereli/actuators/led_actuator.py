import numpy as np
from matplotlib import colors
import pybullet as p
from mereli.globals import global_states
from mereli.register import actuator_registry
from .base_actuator import Actuator

@actuator_registry(name='led')
class LedActuator(Actuator):
    """ LED actuator that turns on or off the LED depending on 
    the action. """
    def __init__(self, *args, num_colors=2, **kwargs):
        super(LedActuator, self).__init__(*args, **kwargs)
        self.colors = [[1,1,1], [1,0,0],  [0,0,1], [0,1,0], [0,1,1], [1,0,1], [1,1,0]]
        self.color_on = [1,0,0]
        self.color_off = [1,1,1]
        self.color_fault = [1,0,0]
        self.on = 0
        self.fault = False
        self.prev_action = None
        self.opacity = 0.6

    def step(self):
        if self.action is None:
            return
        if isinstance(self.action, int) or len(self.action) == 1:
            self.action = self.action * np.ones(8)
        if not global_states.RENDER:
            return

        self.action = self.action if not self.fault else np.zeros_like(self.action)

        for i in range(8):
            led_a = self.action[i] 
            if self.prev_action is not None and led_a == self.prev_action[i]: # Used to optimize code (quite slow otherwise)
                continue
            color = [1,1,1]
            if self.fault:
                color = self.color_fault
            else:
                color = self.colors[int(led_a)]


            color = ('orange', 'b', 'b', 'g', 'g', 'r', 'r', 'purple')[self.actuator_owner.gid]

            # elif 0 < led_a < 1:
            #     color = [led_a, 0, 0]
            led_idx = self.physics_client.get_actuator_position(self.actuator_owner.id, 'led_actuator', sector=i)[1]
            self.actuator_owner.physics_client.set_color(self.actuator_owner.id, led_idx, color, opacity=self.opacity)
        self.prev_action = self.action

        
    @property
    def physics_client(self):
        return self.actuator_owner.physics_client

    def reset(self):
        self.on = 0
        self.fault = False
        self.action = 0 
        self.prev_action = None 

