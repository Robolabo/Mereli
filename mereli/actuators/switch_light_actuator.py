
import numpy as np
from mereli.register import actuator_registry
from .base_actuator import Actuator

@actuator_registry(name='switch_light')
class SwitchLightActuator(Actuator):
    def __init__(self, *args, action_range=0.5, **kwargs):
        super(SwitchLightActuator, self).__init__(*args, **kwargs)
        self.action_range = action_range

    def step(self):
        # action=1 means switch action=0 stay.
        lights = list(self.actuator_owner.static_neighbors.values())
        clst_light_idx = np.argmin([np.linalg.norm(ent.position[:2] - self.actuator_owner.position[:2]) for ent in lights])
        clst_light = lights[clst_light_idx] 
        if self.action == 1:
            if np.linalg.norm(clst_light.position[:2] - self.actuator_owner.position[:2]) <= self.action_range:
                clst_light.switch()

    def reset(self):
        self.action = 1

@actuator_registry(name='switch_light_color')
class SwitchLightColorActuator(Actuator):
    def __init__(self, *args, action_range=0.5, **kwargs):
        super(SwitchLightColorActuator, self).__init__(*args, **kwargs)
        self.action_range = action_range

    def step(self):
        if self.action is None: return
        lights = list(self.actuator_owner.static_neighbors.values())
        clst_light_idx = np.argmin([np.linalg.norm(ent.position[:2] - self.actuator_owner.position[:2]) for ent in lights])
        clst_light = lights[clst_light_idx] 
        if self.action != clst_light.color:
            if np.linalg.norm(clst_light.position[:2] - self.actuator_owner.position[:2]) <= self.action_range:
                clst_light.change_color(self.action)


