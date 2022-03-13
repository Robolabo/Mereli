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

@done_registry(name='light_reached')
class LightReached:
    def __init__(self, color='red'):
        # self.tasks = tasks
        self.color = color

    def __call__(self, entities):
        lights = [obj for obj in entities.values() if type(obj).__name__ == 'LightSource' and obj.color == self.color]
        robots = [obj for obj in entities.values() if issubclass(type(obj), Robot)]
        for robot in robots:
            distances = []
            for light in lights:
                distances.append(LA.norm(robot.position - light.position))
            if not any(np.array(distances) < .5):
                return False
        return True

    def reset(self):
        self.t = 0

@done_registry(name='task_completed')
class TaskCompleted:
    def __init__(self):
        self.task_dones = [LightReached(color='red'), LightReached(color='yellow')]
        self.t = 0

    def __call__(self, entities):
        task = entities['task_scheduler_0'].current_task
        done =  self.task_dones[task](entities) 
        return done

    def reset(self):
        self.t = 0


# @done_registry(name='task_completed')
# class SequentialTasks:
#     def __init__(self):
#         self.task_dones = [LightReached(color='red'), LightReached(color='yellow')]
#         self.t = 0

#     def __call__(self, entities):
#         task = entities['task_scheduler_0'].current_task
#         done =  self.task_dones[task](entities) 
#         return done

#     def reset(self):
#         self.t = 0