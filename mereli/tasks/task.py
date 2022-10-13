import numpy as np
import pybullet as p
from mereli.globals import global_states
from mereli.objects import Robot, LightSource, GroundArea
from mereli.register import tasks, task_registry


class Task:
    def __init__(self, duration=1000, use_done=False):
        self.duration = duration
        self.use_done = use_done
        self.t = 0
        self._done = False
        self._rewards = {} # Dict mapping robot names to rewards

    def __call__(self, entities):
        self.t += 1
        robot_names = [name for name, ent in entities.items() if issubclass(type(ent), Robot)]
        self._rewards = {name : self.reward_generator(entities, name) for name in robot_names}
        for name in robot_names:
            self._rewards[name] = self.reward_generator(entities, name)
            entities[name].reward = self._rewards[name]
        self._done = self.done_generator(entities) or self.time_done() if self.use_done else self.time_done()

    def reward_generator(self, entities, robot_name):
        raise NotImplementedError

    def time_done(self):
        return self.t >= self.duration
        
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

@task_registry(name="dummy")
class DummyTask(Task):
    def __init__(self, *args, **kwargs):
        super(DummyTask,self).__init__(*args, **kwargs)

    def reward_generator(self, *args):
        return np.array([0.])
    
    def done_generator(self, *args):
        return False

@task_registry(name="goto_light")
class GotoLightTask(Task):
    def __init__(self, *args, range=0.5, color='red', **kwargs):
        super(GotoLightTask,self).__init__(*args, **kwargs)
        self.color = color
        self.range = range

    def reward_generator(self, entities, robot_name):
        robot = entities[robot_name]
        if robot.sensors['collision_sensor'].reading:
            return np.array([-1])
        lights = [ent for ent in entities.values() if isinstance(ent, LightSource) and ent.color == self.color]
        other_lights = [ent for ent in entities.values() if isinstance(ent, LightSource) and ent.color != self.color]
        assert len(lights) > 0
        distances = np.array([np.linalg.norm(robot.position[:2] - ls.position[:2]) for ls in lights])
        if min(distances) < self.range:
            return np.array([1])#np.array([1 - (min(distances) / self.range) ** 2])
        else:
            distances = np.array([np.linalg.norm(robot.position[:2] - ls.position[:2]) for ls in other_lights])
            if min(distances) < self.range:
                return np.array([-1])
        return np.array([0.])

    def done_generator(self, entities):
        lights = [ent for ent in entities.values() if isinstance(ent, LightSource) and ent.color == self.color]
        robots = [ent for ent in entities.values() if issubclass(type(ent), Robot)]
        for robot in robots:
            distances = []
            for light in lights:
                distances.append(np.linalg.norm(robot.position[:2] - light.position[:2]))
            if not any(np.array(distances) < self.range):
                return False
        return True

@task_registry(name="task_allocation")
class TaskAllocation(Task):
    def __init__(self, *args, num_tasks=5, agents_per_task=1, **kwargs):
        super(TaskAllocation,self).__init__(*args, **kwargs)
        self.num_tasks = num_tasks
        self.agents_per_task = agents_per_task
        self.prev_task = {}
        self.times_task = {}
        
    def reward_generator(self, entities, robot_name):
        if robot_name not in self.prev_task:
            self.prev_task[robot_name] = -1
            self.times_task[robot_name] = 0
        robot_led = int(entities[robot_name].actuators['led_actuator'].action[0])
        others_led = np.array([int(ent.actuators['led_actuator'].action[0])\
            for ent in entities.values() if issubclass(type(ent), Robot) and ent.id != entities[robot_name].id])
        # all_led = np.array([int(ent.actuators['led_actuator'].action[0])\
        #     for ent in entities.values() if issubclass(type(ent), Robot)])
        # led_res = [all(all_led[i] != all_led[j] for j in range(len(all_led)) if i != j) for i in range(len(all_led))]
        good_decision = all(robot_led != neigh_led for neigh_led in others_led) 
        reward = int(good_decision)
        # if all(led_res): 
        #     reward = np.array([10.])
        # else:
        #     reward = led_res #np.mean(led_res) #np.exp(np.mean(led_res)) - 1
        if good_decision: #all(led != robot_led for led in others_led):
            if robot_led == self.prev_task[robot_name]:
                self.times_task[robot_name] += 1
            else:
                self.times_task[robot_name] = 1
                self.prev_task[robot_name] = robot_led
        else:
            self.times_task[robot_name] = 0
            self.prev_task[robot_name] = -1 
        return reward * self.prev_task[robot_name] #*  min(self.times_task[robot_name], 50) / 50

    def done_generator(self, entities):
        return False
    
    def reset(self):
        super().reset()
        self.prev_task = {}
        self.times_task = {}


@task_registry(name="comm_formation")
class CommFormation(Task):
    def __init__(self, *args, threshold=0.4, points=[], **kwargs):
        super(CommFormation, self).__init__(*args, **kwargs)
        self.points = np.array(points)
        self.threshold = threshold
        
    def reward_generator2(self, entities, robot_name):
        my_state = entities[robot_name].sensors['stateful_rx'].state
        others_state = np.array([ent.sensors['stateful_rx'].state\
            for ent in entities.values() if issubclass(type(ent), Robot) and ent.id != entities[robot_name].id])
        closest = self.points[np.argmin([np.linalg.norm(pt - my_state) for pt in self.points])] 
        # free_spots = [*filter(lambda pt: all([np.linalg.norm(st - pt) > self.threshold for st in others_state]), self.points)]
        # closest = free_spots[np.argmin([np.linalg.norm(pt - my_state) for pt in free_spots])] 
        num_closest = np.sum([np.linalg.norm(st - closest) < self.threshold for st in others_state]) #Thresh before 0.2 
        inside_area = np.linalg.norm(my_state - closest) < self.threshold 

        alpha = 10
        # if inside_area:
        r1 = np.exp(-alpha * np.linalg.norm(my_state - closest)) 
        r2 = np.min([np.linalg.norm(oth_st - my_state) for oth_st in others_state]) / np.sqrt(8) 
        if inside_area and num_closest == 0:
            return r1
        else:
            return r1*r2
        return 0.0 
    
    def reward_generator3(self, entities, robot_name):
        sensor = 'ori_stateful_rx'
        my_state = entities[robot_name].sensors[sensor].state
        others_state = np.array([ent.sensors[sensor].state\
            for ent in entities.values() if issubclass(type(ent), Robot) and ent.id != entities[robot_name].id])
        closest = self.points[np.argmin([np.linalg.norm(pt - my_state) for pt in self.points])] 
        num_closest = np.sum([np.linalg.norm(st - closest) < self.threshold for st in others_state]) #Thresh before 0.2 
        # inside_area = np.linalg.norm(my_state - closest) < 0.2
        alpha = 2
        # if inside_area:
        # r1 = max(0, 1 - np.linalg.norm(my_state - closest) / self.threshold)
        r1 = np.exp(-alpha * np.linalg.norm(my_state - closest)) 
        r2 = np.min([np.linalg.norm(oth_st - my_state) for oth_st in others_state]) / np.sqrt(8) 
        if num_closest == 0:
            # r2 = np.exp(5 * r2 - 5)
            # __import__('pdb').set_trace()
            reward = r1 * r2 
        else:
            reward = 0 
        return reward

    def reward_generator(self, entities, robot_name):
        sensor = 'ori_stateful_rx'
        my_state = entities[robot_name].sensors[sensor].state
        others_state = np.array([ent.sensors[sensor].state\
            for ent in entities.values() if issubclass(type(ent), Robot) and ent.id != entities[robot_name].id])
        closest = self.points[np.argmin([np.linalg.norm(pt - my_state) for pt in self.points])] 
        num_closest = np.sum([np.linalg.norm(st - closest) < self.threshold for st in others_state]) #Thresh before 0.2 
        alpha = 1 
        dist_neigh = np.min([np.linalg.norm(oth_st - my_state) for oth_st in others_state])
        dist_tar = np.linalg.norm(my_state - closest) 
        
        # if dist_neigh < self.threshold:
        #     return - (1 - dist_neigh / self.threshold) 
        # else:
            # return max(0, 1 - dist_tar/self.threshold)
        return np.exp(-alpha * dist_tar) 

    def done_generator(self, entities):
        return False
    
    def reset(self):
        super().reset()

@task_registry(name="group_formation")
class GroupFormation(Task):
    def __init__(self, *args, num_members=4, radius=0.3, centroids=[], **kwargs):
        super(GroupFormation, self).__init__(*args, **kwargs)
        self.centroids= np.array(centroids)
        self.radius = radius
        self.num_members = num_members
    
    def reward_generator(self, entities, robot_name):
        my_state = entities[robot_name].sensors['stateful_rx'].state
        others_state = np.array([ent.sensors['stateful_rx'].state\
            for ent in entities.values() if issubclass(type(ent), Robot) and ent.id != entities[robot_name].id])
        centroid = self.centroids[np.argmin([np.linalg.norm(pt - my_state) for pt in self.centroids])] 
        inside_area = np.linalg.norm(centroid - my_state) <= self.radius
        num_others_inside = np.sum([np.linalg.norm(st - centroid) < self.radius for st in others_state]) #Thresh before 0.2 
        if inside_area:
            if num_others_inside < self.num_members - 1:
                return 1.0
            elif num_others_inside == self.num_members - 1:
                return 2.0
            else:
                return 0.0
        else:
            return 0.0
        
    def done_generator(self, entities):
        return False
    
    def reset(self):
        super().reset()


@task_registry(name="task_sequence")
class TaskSequence(Task):
    def __init__(self, *args, radius=0.3, centroids=[], **kwargs):
        super(TaskSequence, self).__init__(*args, **kwargs)
        self.centroids= np.array(centroids)
        self.radius = radius
        self.correct_sequence = np.arange(len(centroids))
        self.curr_tsk_idx = 0
        self.seq_done = False 
    
    def reward_generator(self, entities, robot_name):
        my_state = entities[robot_name].sensors['stateful_rx'].state
        others_state = np.array([ent.sensors['stateful_rx'].state\
            for ent in entities.values() if issubclass(type(ent), Robot) and ent.id != entities[robot_name].id])
        inside_area = np.linalg.norm(self.current_task - my_state) <= self.radius
        all_inside_area = np.all([np.linalg.norm(st - self.current_task) < self.radius for st in others_state]) 
        
        if inside_area and not all_inside_area:
            return self.curr_tsk_idx + 1
        elif inside_area and all_inside_area:
            self.curr_tsk_idx += 1
            if self.curr_tsk_idx == len(self.correct_sequence):
                self.seq_done = True
            return self.curr_tsk_idx + 1 
        else:
            return 0.0


    def done_generator(self, entities):
        return self.seq_done 
    
    @property
    def current_task(self):
        return self.centroids[self.curr_tsk_idx]

    def reset(self):
        super().reset()
        self.curr_tsk_idx = 0
        np.random.shuffle(self.correct_sequence)
        self.seq_done = False 

@task_registry(name="obstacle_avoidance")
class ObstacleAvoidance(Task):
    def __init__(self, *args, **kwargs):
        super(ObstacleAvoidance,self).__init__(*args, **kwargs)

    def reward_generator(self, entities, robot_name):
        robot = entities[robot_name]
        ds = robot.sensors['distance_sensor'].reading
        wheels = robot.actuators['joint_velocity_actuator'].action / robot.actuators['joint_velocity_actuator'].max_velocity
        rA = 0. if any(ds > 0.4) else 1.
        rB = max(1 - np.abs(wheels[0] - wheels[1]), 0) * np.linalg.norm(wheels)
        return rA * rB

    def done_generator(self, entities):
        return False


 
@task_registry(name="goto_nest")
class GotoNestTask(Task):
    def __init__(self, *args, color='grey', **kwargs):
        super(GotoNestTask,self).__init__(*args, **kwargs)
        self.color = color

    def reward_generator(self, entities, robot_name):
        robot = entities[robot_name]
        dist_robots = np.array([np.linalg.norm(ent.position - robot.position) for ent in entities.values()\
                    if issubclass(type(ent), Robot) if ent.id != robot.id])
        if any(dist_robots < 0.1):
            return np.array([-1])
        nests = [ent for ent in entities.values() if isinstance(ent, GroundArea) and ent.color == self.color]
        assert len(nests) > 0
        inside_nests = np.array([np.linalg.norm(robot.position[:2] - nest.position[:2]) < nest.radius for nest in nests])
        if any(inside_nests):
            return np.array([1])#np.array([1 - (min(distances) / self.range) ** 2])
        return np.array([0.])

    def done_generator(self, entities):
        return False

@task_registry(name="best_of_n")
class BestofN(Task):
    def __init__(self, *args, num_areas=3, **kwargs):
        super(BestofN,self).__init__(*args, **kwargs)
        self.num_areas = num_areas
        self.qualities = [1, 0.5, 0, 0, 0]
        np.random.shuffle(self.qualities)

    def reward_generator(self, entities, robot_name):
        robot = entities[robot_name]
        # dist_robots = np.array([np.linalg.norm(ent.position - robot.position) for ent in entities.values()\
        #             if issubclass(type(ent), Robot) if ent.id != robot.id])
        # if any(dist_robots < 0.1):
        #     return np.array([-1])
        # nests = [ent for ent in entities.values() if isinstance(ent, GroundArea)]
        nest = [ent for ent in entities.values() if isinstance(ent, GroundArea) and ent.color == 'green'][0]
        # other_nests = [ent for ent in entities.values() if isinstance(ent, GroundArea) and ent.color != 'green'] 
        inside_nest = np.linalg.norm(robot.position[:2] - nest.position[:2]) < nest.radius
        if inside_nest:
            return np.array([1.])#self.qualities[np.where(inside_nests)[0][0]]
        # else:
        #     inside_other = [np.linalg.norm(robot.position[:2] - other.position[:2]) < other.radius for other in other_nests]
        #     if any(inside_other):
        #         return np.array([-1])
        return np.array([0.])

    def done_generator(self, entities):
        return False

    def reset(self):
        super().reset()
        np.random.shuffle(self.qualities)

class TaskManager:
    def __init__(self, duration=1000, use_done=False, num_slots=2, rand_order=True):
        self.tasks = []
        self.duration = duration
        self.num_slots = num_slots
        self.use_done = use_done
        self.time_props = [1/len(tasks)] * len(tasks)
        self.task_durations = [int(prop * self.duration) for prop in self.time_props]
        self.rand_order = rand_order
        self.task_order = None
        self.block = 0
        self.t = 0

    def add_task(self, task_name, **task_params):
        task = tasks[task_name](**task_params, duration=self.duration//self.num_slots, use_done=self.use_done)
        self.tasks.append(task)

    def __call__(self, entities):
        self.t += 1
        # print(self.t,self.current_task_idx)
        if self.block >= self.num_slots:
            return
        self.current_task(entities)
        if self.task_done:
            self.block += 1
            if self.block >= self.num_slots:
                return
            self.current_task.reset()
        self.render_task()
        robot_names = [name for name, ent in entities.items() if issubclass(type(ent), Robot)]
        for robot in robot_names:
            entities[robot].task = np.array([self.current_task_idx / (self.num_tasks - 1)])
        

    def render_task(self):
        return
        if global_states.RENDER:
            if self.t == 1:
                self.label_id = p.addUserDebugText(str(self.current_task_idx), (0,0,0.1), 
                        textColorRGB=(0,0,0), textSize=2, )
            else:
                self.label_id = p.addUserDebugText(str(self.current_task_idx), (-0.5,0,0.1), 
                        textColorRGB=(0,0,0), textSize=2, replaceItemUniqueId=self.label_id)

    @property
    def num_tasks(self):
        return len(self.tasks)

    @property
    def is_done(self):
        return self.task_done and self.block == self.num_slots or self.t >= self.duration 

    @property
    def current_task_idx(self):
        return self.task_order[self.block] if self.block < self.num_slots else self.task_order[-1]

    @property
    def current_task(self):
        return self.tasks[self.current_task_idx]

    @property
    def task_done(self):
        return self.current_task.is_done
    
    @property
    def rewards(self):
        return self.current_task.rewards

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.block = 0
        self.t = 0
        self.task_order = np.random.choice(self.num_tasks, size=self.num_slots, replace=False)
        for tsk in self.tasks:
            tsk.reset()
        if seed is not None:
            np.random.seed()


