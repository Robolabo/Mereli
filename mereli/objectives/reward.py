import numpy as np
import numpy.linalg as LA
from mereli.utils import angle_mean, angle_diff, increase_time
from mereli.register import reward_registry



@reward_registry(name='many_lights')
class GoToLightReward:
    def __init__(self, color='red'):
        self.color=color

    def __call__(self, actions, states, robot, info=None):
        lights = [obj for obj in info if type(obj).__name__ == 'LightSource' and obj.color == self.color]
        distances_ls = np.array([np.linalg.norm(ls.position[:2] - robot.position[:2]) for ls in lights])
        if any(distances_ls < 1.5):
            return np.array([1])#np.array([1 - (distances_ls/15)**2])
        else:
            return np.array([0])
        # return np.array([1 if any(distances_ls < 2.0) else -1)]) #Antes 0 en vez de -1
            
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
        rews = [tsk(actions, states, robot, info=info) for tsk in self.tasks]
        rew = np.sum([rews[i]*(-1,1)[i == current_task]for i in range(len(rews))])
        # import pdb; pdb.set_trace()
        return rews[current_task].flatten()
        return np.array([rew])
    
    def reset(self):
        pass