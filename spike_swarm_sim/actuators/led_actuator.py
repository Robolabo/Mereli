from matplotlib import colors
import pybullet as p
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

@actuator_registry(name='led_actuator_3D')
class LedActuator3D(Actuator):
    """ LED actuator that turns on or off the LED depending on 
    the action. """
    def __init__(self, *args, color_on='blue', color_off='white', **kwargs):
        super(LedActuator3D, self).__init__(*args, **kwargs)
        self.color_on = color_on
        self.color_off = color_off
        self.on = 0

    def step(self, action):
        self.on = action
        #! Check color with colors.is_color_like
        color = colors.to_rgba(self.color_on) if self.on else colors.to_rgba(self.color_off)
        # In epuck 3 is the led piece
        p.changeVisualShape(self.actuator_owner.id, 3, rgbaColor=color,\
            physicsClientId=self.actuator_owner.physics_client)