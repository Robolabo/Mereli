import numpy as np
from mereli.objects import Robot, LightSource

class Task:
    def __init__(self, duration=1000):
        self.duration = duration
        self.t = 0
        self._done = False
        self._rewards = {} # Dict mapping robot names to rewards

    def __call__(self, entities):
        self.t += 1
        robot_names = [name for name, ent in entities.items() if issubclass(type(ent), Robot)]
        self._rewards = {self.reward_generator(entities, name) for name in robot_names}
        self._done = self.done_generator(entities)

    def reward_generator(self, entities, robot_name):
        raise NotImplementedError

    def done_generator(self, entities):
        raise NotImplementedError

    @property
    def is_done(self):
        return self._done

    @property
    def rewards(self):
        return self._rewards

    def reward(self, robot_name):
        return self._rewards[robot_name]

    def reset(self):
        self.t = 0
        self._reward = 0
        self._done = False

@task_registry(name="goto_light ")
class GotoLightTask(Task):
    def __init__(self, *args, range=0.5, color='red', **kwargs):
        super(GotoLightTask,self).__init__(*args, **kwargs)
        self.color = color

    def reward_generator(self, entities, robot_name):
        robot = entities[robot_name]
        lights = [ent for ent in entities.values() if isinstance(type(ent), LightSource) and ent.color == self.color]
        assert len(lights) > 0
        distances = [np.linalg.norm(robot.position - ls.position) for ls in lights]
        if min(distances) < self.range:
            return np.array([1 - min(distances)/self.range])
        else:
            return np.array([0])

    def done_generator(self, entities):
        lights = [ent for ent in entities.values() if isinstance(type(ent), LightSource) and ent.color == self.color]
        robots = [ent for ent in entities.values() if issubclass(type(ent), Robot)]
        for robot in robots:
            distances = []
            for light in lights:
                distances.append(np.linalg.norm(robot.position - light.position))
            if not any(np.array(distances) < self.range):
                return False
        return True

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


