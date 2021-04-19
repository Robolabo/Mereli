import numpy as np
from spike_swarm_sim.objects.world_object import WorldObject
from spike_swarm_sim.register import world_object_registry
from spike_swarm_sim.utils import increase_time

@world_object_registry(name='task_scheduler')
class TaskScheduler(WorldObject):
    def __init__(self, *args, total_timesteps=1000, num_tasks=2, num_slots=2, **kwargs):
        super(TaskScheduler, self).__init__(*args, tangible=False, **kwargs)
        self.total_timesteps = total_timesteps
        self.num_tasks = num_tasks
        self.num_slots = num_slots
        
        # self.min_slot_duration = int(total_timesteps
        # self.max_slot_duration = 
        self.task_order = None
        # self.task_switch = None
        self.t = 0


    def step(self, neighborhood):
        self.t += 1
        # print(self.t, self.current_task)

    def controllable(self):
        return True

    @property
    def current_task(self):
        assert self.task_order is not None
        return self.task_order[self.t // (self.total_timesteps // self.num_slots + 1)]

    def reset(self, seed=None):#! OJO seed
        if seed is not None:
            np.random.seed(seed)
        self.t = 0
        self.task_order = np.random.choice(self.num_tasks, size=self.num_slots, replace=False)
        # self.task_switch 
        if seed is not None:
            np.random.seed()
    
    def add_physics(self, engine):
        pass

    def position(self):
        pass
    
    def orientation(self):
        pass
    