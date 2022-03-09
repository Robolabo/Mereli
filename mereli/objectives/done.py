import numpy as np
import numpy.linalg as LA
from mereli.register import done_registry
from mereli.objects import Robot

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

class LightReached:
    def __init__(self, color='red'):
        # self.tasks = tasks
        self.color = color

    def __call__(self, entities):
        lights = [obj for obj in entities.values() if type(obj).__name__ == 'LightSource' and obj.color == self.color]
        robots = [obj for obj in entities.values() if issubclass(type(obj), Robot)]
        done = True
        for robot in robots:
             = []
            for light in lights:
                if LA.norm(robot.position - light.position) < 1.5:

        import pdb; pdb.set_trace()
        return False

    def reset(self):
        self.t = 0


class TaskCompleted:
    def __init__(self):
        self.task_dones = [LightReached(color='red'), LightReached(color='yellow')]
        self.t = 0

    def __call__(self, entities):
        task = entities['task_scheduler_0'].current_task
        return self.task_dones[task](entities) 
        return False

    def reset(self):
        self.t = 0