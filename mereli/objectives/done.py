import numpy as np
import numpy.linalg as LA
from mereli.register import done_registry


@done_registry(name='time_elapsed')
class TimeElapsed:
    def __init__(self, timesteps=100):
        self.timesteps=timesteps
        self.t = 0

    def __call__(self, entities):
        self.t += 1
        if self.t >= self.timesteps:
            return True
        return False

    def reset(self):
        self.t = 0

class TaskCompleted:
    def __init__(self, ):
        self.timesteps = timesteps
        self.t = 0

    def __call__(self, entities):
        self.t += 1
        if self.t >= self.timesteps:
            return True
        return False

    def reset(self):
        self.t = 0