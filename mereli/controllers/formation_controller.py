import numpy as np
from mereli.controllers import RobotController
from mereli.register import controller_registry

@controller_registry(name='formation_controller')
class FormationController(RobotController):
    def __init__(self, *args, tar_points=[], **kwargs):
        super(FormationController, self).__init__(*args, **kwargs)

    def step(self, state, reward=0.0):
        closest_tar = state['closest_target']
        closest_state = state['closest_state']
        own_state = state['own_state']
        if np.linalg.norm(closest_state - own_state) < 0.01:
            force = np.random.uniform(-1,1,2)
        else:
            forceA = (closest_tar - own_state) 
            forceB = (closest_state - own_state) if np.linalg.norm(closest_state - own_state) < 0.1 else np.array([0.,0.])
            force = forceA + forceB 
        # force = np.random.uniform(-1,1,2)
        return {'stateful_tx' : 50 * force} 
