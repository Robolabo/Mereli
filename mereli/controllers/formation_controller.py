import numpy as np
from mereli.controllers import RobotController
from mereli.register import controller_registry

@controller_registry(name='formation_controller')
class FormationController(RobotController):
    def __init__(self, *args, tar_points=[], **kwargs):
        super(FormationController, self).__init__(*args, **kwargs)
        self.attractors = [[0.75, 0.75], [-0.75, -0.75], [0.75, -.75], [-.75, .75]]

    def step(self, state, reward=0.0):
        closest_tar = state['closest_target']
        closest_state = state['closest_state']
        own_state = state['own_state']
        closest_attr = self.attractors[np.argmin([np.linalg.norm(attr - own_state) for attr in self.attractors])]

        fa = (closest_attr - own_state)
        fa_norm = fa / np.linalg.norm(fa)
        fb = (closest_attr - own_state)
        fb_norm = fb / np.linalg.norm(fb)
        A = 1 - np.linalg.norm(fa) / 2
        B = 1 - np.linalg.norm(fb) / 2
        force = A*fa + B*fb
        # if np.linalg.norm(own_state - closest_state) < 0.2:
        #     force += 2 * (own_state - closest_state)

        # if np.linalg.norm(closest_state - own_state) < 0.01:
        #     force = np.random.uniform(-1,1,2)
        # else:
        #     forceA = (closest_tar - own_state) 
        #     forceB = (closest_state - own_state) if np.linalg.norm(closest_state - own_state) < 0.1 else np.array([0.,0.])
        #     force = forceA + forceB 
        # force = np.random.uniform(-1,1,2)
        return {'stateful_tx' : 0.5 *  force} 
