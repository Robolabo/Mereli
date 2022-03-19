import numpy as np


class Task:
    def __init__(self, duration=1000):
        self.duration = duration
        self.t = 0
        self._done = False
        self._reward = 0

    def __call__(self):
        self.t += 1
        self._reward

    def reward_generator(self, entities):
        pass

    @property
    def is_done(self):
        pass

    @property
    def reward(self):
        pass

    def reset(self):
        self.t = 0
        self._reward = 0
        self._done = False

class GotoLightTask(Task):
    def __init__(self, *args, range=0.5, color='red', **kwargs):
        super(GotoLightTask,self).__init__(*args, **kwargs)
        self.color = color

    @property
    def is_done(self):
        pass


class TaskManager:
    def __init__(self, tasks, time_props, duration=1000, rand_order=True):
        self.tasks = tasks
        self.duration = duration
        self.time_props = time_props
        if self.time_props == 'same': 
            self.time_props = [1/len(tasks)] * len(tasks)
        self.task_durations = [int(prop * self.duration) for prop in self.time_props]
        self.rand_order = rand_order
        self.current_task = self.task_order[0]
        self.block = 0
        self.t = 0

    def __call__(self, entities):
        if self.task_done:
            self.block += 1
        self.t += 1

    @property
    def is_done(self):
        pass

    @property
    def current_task_idx(self):
        return self.task_order[self.block]

    
    @property
    def current_task(self):
        return self.task[self.block]

    @property
    def task_done(self):
        return self.tasks[self.current_task].is_done
    
    @property
    def reward(self):
        return self.tasks[self.current_task].reward

    def reset(self):
        self.block = 0
        self.t = 0
        self.task_order = np.random.choice(self.num_tasks, size=self.num_slots, replace=False)


