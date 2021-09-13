import numpy as np
import numpy.linalg as LA
from spike_swarm_sim.utils import angle_mean, angle_diff, increase_time
from spike_swarm_sim.register import reward_registry



@reward_registry(name='many_lights')
class GoToLightReward:
    def __init__(self, color='red'):
        self.color=color

    def __call__(self, actions, states, robot, info=None):
        lights = [obj for obj in info if type(obj).__name__ == 'LightSource' and obj.color == self.color]
        distances_ls = np.array([np.linalg.norm(ls.position[:2] - robot.position[:2]) for ls in lights])
        return np.array([int(any(distances_ls < 1.0) if len(distances_ls) > 0 else 0.0)])
            
        # return  rew_obst + rew_ls
    def reset(self):
        pass


@reward_registry(name='task_switching_lights')
class TaskSwitchingLights:

    def __init__(self):
        self.tasks = [GoToLightReward(color='red'), GoToLightReward(color='yellow'),\
                    GoToLightReward(color='blue'), GoToLightReward(color='green')]
        # self.required_info = tuple(set(['task_scheduler:current_task']).union(*[set(tsk.required_info) for tsk in self.tasks]))
        self.buffered_fitnesses = []

    def __call__(self, actions, states, robot, info=None):
        task_scheduler = [obj for obj in info if type(obj).__name__ == 'TaskScheduler'][0]
        current_task = task_scheduler.current_task
        return self.tasks[current_task](actions, states, robot, info=info)
    
    def reset(self):
        pass