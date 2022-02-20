from itertools import combinations
from functools import reduce
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plot
from mereli.register import fitness_func_registry
from mereli.utils import (normalize, compute_angle, angle_diff,
                                    geom_mean, angle_mean, get_alphashape,
                                    convert2graph, disjoint_subgraphs, toroidal_difference)

@fitness_func_registry(name='identify_borderline')
class IdentifyBorderline:
    """Fitness function for the borderline identification task."""
    def __init__(self):
        self.required_info = ("robot_positions",)
        self.borderline = []

    def __call__(self, actions, states, info=None):
        """Computes the fitness function based on trial actions and states.
        Additionally, other useful variables can be used from info dict (if specified in init).
        =======================================================================================
        - Args:
            actions [list of dicts]: list of dictionaries with actuator names and the
                    corresponding action.
            states [list of dicts]: list of dictionaries with sensor names and the
                    corresponding measured states.
            info [dict or None]: dict of additional information.
        =======================================================================================
        """
        actions = np.stack(actions)
        #* Assume static robots
        positions = info['robot_positions'][0]

        #* Compute alpha-shape
        borderline_robots = []
        # Check if there are robot clusters
        subgraphs = disjoint_subgraphs(convert2graph(positions, max_dist=100))
        for subG in subgraphs: # Compute alpha shape for each subgraph
            borderline_subG, _ = get_alphashape(positions[subG].copy()/1000, alpha=15)#0.8)
            borderline_robots.extend(borderline_subG)
        #* Compute fitness
        target_actions = np.array([1 if k in borderline_robots else 0 for k in range(len(positions))])
        self.borderline = target_actions
        fitness = 0
        for _, timestep_actions in enumerate(actions):
            action_vector = np.array([ac['led_actuator'] for ac in timestep_actions])
            true_neg = np.sum([(ac == 0) and not target for ac, target in zip(action_vector, target_actions)])\
                     / np.sum(1 - target_actions)
            true_pos = np.sum([(ac == 1) and target for ac, target in zip(action_vector, target_actions)])\
                     / np.sum(target_actions)
            Fi = true_neg * true_pos
            fitness += Fi
        fitness /= len(actions)
        return fitness + 1e-5

@fitness_func_registry(name='identify_leader')
class IdentifyLeader:
    """Fitness function for the leader selection task."""
    def __init__(self):
        self.required_info = ("generation",)

    def __call__(self, actions, states, info=None):
        """Computes the fitness function based on trial actions and states.
        Additionally, other useful variables can be used from info dict (if specified in init).
        =======================================================================================
        - Args:
            actions [list of dicts]: list of dictionaries with actuator names and the
                    corresponding action.
            states [list of dicts]: list of dictionaries with sensor names and the
                    corresponding measured states.
            info [dict or None]: dict of additional information.
        =======================================================================================
        """
        actions = np.stack([[ac_robot['led_actuator'] for ac_robot in ac] for ac in actions])
        fitness = 0
        consecutive_leader = 0
        prev_leader = None
        init_timestep = 0
        for robot_actions in actions[init_timestep:]:
            if np.sum(robot_actions) == 1:
                new_leader = np.argmax(robot_actions)
                if prev_leader == new_leader:
                    consecutive_leader += 1
                    fitness += np.clip(0.1 * consecutive_leader, a_min=0, a_max=5)
                else:
                    consecutive_leader = 0
                prev_leader = new_leader
            else:
                prev_leader = None
                consecutive_leader = 0
            fitness = np.clip(fitness, a_min=0, a_max=None)
        return fitness / (len(actions) - init_timestep)

@fitness_func_registry(name='alignment')
class Alignment:
    """Fitness function for the orientation consensus task."""
    def __init__(self):
        self.required_info = ("robot_positions", "robot_orientations",)

    def __call__(self, actions, states, info=None):
        """Computes the fitness function based on trial actions and states.
        Additionally, other useful variables can be used from info dict (if specified in init).
        =======================================================================================
        - Args:
            actions [list of dicts]: list of dictionaries with actuator names and the
                    corresponding action.
            states [list of dicts]: list of dictionaries with sensor names and the
                    corresponding measured states.
            info [dict or None]: dict of additional information.
        =======================================================================================
        """
        robot_orientations = np.stack(info["robot_orientations"]).copy()
        fitness = 0
        initial_timestep = 0 # ignore previous timesteps for fitness computation
        for t, (thetas, action) in enumerate(zip(robot_orientations[initial_timestep:], \
                        np.array(actions)[initial_timestep:]), start=initial_timestep):
            #angle_errs = np.max([angle_diff(th1[-1], th2[-1])
            #                    for j, th1 in enumerate(thetas)
            #                    for i, th2 in enumerate(thetas) if i != j])
            angle_errs = np.mean([angle_diff(th1[-1], th2[-1])
                                for j, th1 in enumerate(thetas)
                                for i, th2 in enumerate(thetas) if i != j])
            #import pdb; pdb.set_trace()
            fA = np.clip(1 - (angle_errs / (0.5 * np.pi)), a_min=0, a_max=1)
            fB = np.mean([np.clip(1 - np.abs(ac['joint_velocity_actuator'][0]), a_min=0, a_max=1) for ac in action])
            fitness += (fA * fB)
        fitness /= (len(robot_orientations) - initial_timestep)
        return fitness + 1e-5

@fitness_func_registry(name='goto_light')
class GotoLight:
    """Fitness function for the light follower task."""
    def __init__(self, color='red'):
        self.color = color
        self.required_info = ("robot:position", "robot:orientation", "light_source:position@color="+color)

    def __call__(self, actions, states, info=None):
        """Computes the fitness function based on trial actions and states.
        Additionally, other useful variables can be used from info dict (if specified in init).
        =======================================================================================
        - Args:
            actions [list of dicts]: list of dictionaries with actuator names and the
                    corresponding action.
            states [list of dicts]: list of dictionaries with sensor names and the
                    corresponding measured states.
            info [dict or None]: dict of additional information.
        =======================================================================================
        """
        robot_positions = np.stack(info["robot:position"]).copy()
        light_positions = np.stack(info["light_source:position@color="+self.color]).copy()
        fitness = 0
        for t, (pos, light_pos, actions_t)  in enumerate(zip(robot_positions, light_positions, actions)):
            #* Considering only 1 light
            light_pos = light_pos.flatten()
            distances = LA.norm(pos[:, :2] - light_pos[:2], axis=1)
            #* Considering that there can be multiple lights
            # distances = np.min([LA.norm(pos[:, :2] - ls_pos[:2], axis=1) for ls_pos in light_pos], 0)
            # fA = (distances < 1).mean()
            fA = int(all(distances < 1))
            # if len(distances) > 1:
            #     fA *= (np.sum(distances < 1) > 1)
            if t < 100:#antes a 100
                rad_ball = -(3/100) * t + 3
                fA = np.clip(1 - (distances / rad_ball), a_max=1, a_min=0).mean()
            # fitness += 
            fitness += fA
        # return (fitness / len(states)) + 1e-5
        return (fitness / len(states)) + 1e-5

@fitness_func_registry(name='transport_cubes')
class TransportCubesFitness:
    """Fitness function for the light follower task."""
    def __init__(self):
        self.required_info = ("robot:position", "robot:orientation",
                            "light_source:position@color=yellow",
                            "cube:position", "ground_area:position",
                            "ground_area:radius", "cube:is_grasped")

    def __call__(self, actions, states, info=None):
        """Computes the fitness function based on trial actions and states.
        Additionally, other useful variables can be used from info dict (if specified in init).
        =======================================================================================
        - Args:
            actions [list of dicts]: list of dictionaries with actuator names and the
                    corresponding action.
            states [list of dicts]: list of dictionaries with sensor names and the
                    corresponding measured states.
            info [dict or None]: dict of additional information.
        =======================================================================================
        """

        robot_positions = np.stack(info["robot:position"]).copy()
        cube_positions = np.stack(info["cube:position"]).copy()
        ground_area_pos = info["ground_area:position"][0]
        ground_area_rad = info["ground_area:radius"][0][0]#Second index supposes all ground areas have same area.
        light_source_pos = info["light_source:position@color=yellow"][0].flatten()
        cubes_grasped = info["cube:is_grasped"][-1]
        #* Correct area is the one with light source above
        correct_area_idx = np.argmin(LA.norm(ground_area_pos[:, :2] - light_source_pos[:2], axis=1))
        correct_area = ground_area_pos[correct_area_idx]
        n_cubes_correct_start = np.sum([LA.norm(cube_pos - correct_area) <= ground_area_rad for cube_pos in cube_positions[0]])
        # wrong_areas = np.array([ground_area_pos[j] for j in range(len(ground_area_pos)) if j != correct_area_idx])
        n_cubes_correct = np.sum([not is_grasped and LA.norm(cube_pos - correct_area) <= ground_area_rad\
                            for is_grasped, cube_pos in zip(cubes_grasped, cube_positions[-1])])
        # n_cubes_wrong = np.sum([not is_grasped and any(LA.norm(cube_pos - wrong_areas, axis=1) <= ground_area_rad)\
        #                     for is_grasped, cube_pos in zip(cubes_grasped, cube_positions[-1])])
        mask_dist_moved = LA.norm(cube_positions[-1] - correct_area, axis=1) < LA.norm(cube_positions[0] - correct_area, axis=1)
        dist_moved = LA.norm(cube_positions[-1] - cube_positions[0], axis=1)
        dist_moved[dist_moved < 0.1] = 0.
        mean_dist_moved = (mask_dist_moved * dist_moved).mean() / 10
        fitness = max(0, n_cubes_correct + mean_dist_moved - n_cubes_correct_start) / cube_positions.shape[1]
        return fitness + 1e-5



@fitness_func_registry(name='task_switching3old')
class TaskSwitching3:
    """Fitness function for the exploration task."""
    def __init__(self):
        self.tasks = [GotoLight(), TransportCubesFitness()]
        #! Add current task info
        self.required_info = tuple(set(['task_scheduler:current_task', 'task_scheduler:num_slots']).union(*[set(tsk.required_info) for tsk in self.tasks]))

    def __call__(self, actions, states, info=None):
        tasks = np.array(info['task_scheduler:current_task']).flatten()
        n_slots = info['task_scheduler:num_slots'][0].item()
        n_collisions = np.sum([[st['collision_sensor'] for st in state_t] for state_t in states], 0)
        if any(n_collisions > 200):
            return 1e-5
        fitness_tasks = []
        for i in range(n_slots):
            t_init = int(i * len(actions) / n_slots)
            t_end = int((i + 1)  * len(actions) / n_slots)
            task_actions = np.array(actions)[t_init:t_end]
            task_states = np.array(states)[t_init:t_end]
            task_info = {key : np.array(values)[t_init:t_end]\
                if key != 'generation' else values for key, values in info.items()}
            task = self.tasks[tasks[t_init+1]]
            fitness_tasks.append(task(task_actions, task_states, info=task_info))
        fitness = np.prod(fitness_tasks) ** (1 / len(fitness_tasks)) #* Geom mean combination
        return np.log(fitness + 1e-5)

@fitness_func_registry(name='task_switching3')
class TaskSwitching4Lights:
    """Fitness function for the exploration task."""
    def __init__(self):
        # self.tasks = [GotoLight(color='red'), GotoLight(color='yellow'), GotoLight(color='blue'), GotoLight(color='green')]
        self.tasks = [GotoLight(color='red'), GotoLight(color='yellow')]
        #! Add current task info
        self.required_info = tuple(set(['task_scheduler:current_task', 'task_scheduler:num_slots']).union(*[set(tsk.required_info) for tsk in self.tasks]))

    def __call__(self, actions, states, info=None):
        tasks = np.array(info['task_scheduler:current_task']).flatten()
        n_slots = info['task_scheduler:num_slots'][0].item()
        fitness_tasks = []
        for i in range(n_slots):
            t_init = int(i * len(actions) / n_slots)
            t_end = int((i + 1)  * len(actions) / n_slots)
            task_actions = np.array(actions)[t_init:t_end]
            task_states = np.array(states)[t_init:t_end]
            task_info = {key : np.array(values)[t_init:t_end]\
                if key != 'generation' else values for key, values in info.items()}
            task = self.tasks[tasks[t_init+1]]
            fitness_tasks.append(task(task_actions, task_states, info=task_info))

        fitness = np.prod(fitness_tasks) ** (1 / len(fitness_tasks)) #* Geom mean combination
        return fitness + 1e-5


@fitness_func_registry(name='task_switching_B')
class TaskSwitchingB:
    """Fitness function for the exploration task."""
    def __init__(self):
        self.required_info = ['robot:reward']

    def __call__(self, actions, states, info=None):
        fA, fB = 0, 0
        for t, rew_t in enumerate(info['robot:reward']):
            # try:
            #     V_t = np.sum([np.clip(info['robot:reward'][k].flatten(), a_min=0, a_max=1) * 0.95 ** (k-t) for k in range(t, len(states))],0)
            # except:
            #     import pdb; pdb.set_trace()
            # F_tA = len(rew_t) / np.sum((np.clip(rew_t.flatten(), a_min=0, a_max=None)+1e-5)**-1)
            if True: #sum(rew_t.flatten() > 0) > 1:
                if t <= len(states) / 2: 
                    fA += np.mean(rew_t.flatten())
                else:
                    fB += np.mean(rew_t.flatten())
        fA /= 0.5 * len(states)
        fB /= 0.5 * len(states)
        return (fA + fB)/2 + 1e-5
        return np.sqrt(fA * fB) + 1e-5



@fitness_func_registry(name='grouping')
class Grouping:
    """Fitness function for the aggrupation task."""
    def __init__(self):
        self.required_info = ("robot_positions", "robot_orientations")

    def __call__(self, actions, states, info=None):
        """Computes the fitness function based on trial actions and states.
        Additionally, other useful variables can be used from info dict (if specified in init).
        =======================================================================================
        - Args:
            actions [list of dicts]: list of dictionaries with actuator names and the
                    corresponding action.
            states [list of dicts]: list of dictionaries with sensor names and the
                    corresponding measured states.
            info [dict or None]: dict of additional information.
        =======================================================================================
        """
        robot_positions = np.stack(info["robot_positions"]).copy()
        robot_orientations = np.stack(info["robot_orientations"]).copy()
        fitness = 0
        for _, (pos, thetas, action)  in enumerate(zip(robot_positions, robot_orientations, actions)):
            # angle_errs = np.mean([angle_diff(th1, th2)
            #                     for j, th1 in enumerate(thetas)
            #                     for i, th2 in enumerate(thetas) if i != j])
            distances_robots = [LA.norm(pos_i - pos_j)
                                for i, pos_i in enumerate(pos)
                                for j, pos_j in enumerate(pos) if i != j]
            distances = [LA.norm(pos_i - np.mean(pos, 0)) for pos_i in pos]
            fA = np.mean([ np.clip(1 - dist / 100, a_min=0, a_max=1) for dist in distances])
            fB = np.min(distances_robots) > 20
            fitness += fA * fB
        fitness /= len(robot_positions)
        return fitness + 1e-5




@fitness_func_registry(name='obstacle_avoidance')
class ObstacleAvoidance:
    """Fitness function for the Obstacle Avoidance task."""
    def __init__(self):
        self.required_info = ("robot:position",)

    def __call__(self, actions, states, info=None):
        """Computes the fitness function based on trial actions and states.
        Additionally, other useful variables can be used from info dict (if specified in init).
        =======================================================================================
        - Args:
            actions [list of dicts]: list of dictionaries with actuator names and the
                    corresponding action.
            states [list of dicts]: list of dictionaries with sensor names and the
                    corresponding measured states.
            info [dict or None]: dict of additional information.
        =======================================================================================
        """
        fitness = 0
        robot_positions = np.stack(info["robot:position"]).copy()
        for t, (states_t, actions_t, pos)  in enumerate(zip(states, actions, robot_positions)):
            # fA = np.mean([np.max(st['distance_sensor']) == 0.0 for st in states_t])
            fA = np.mean([0 if np.max(st['distance_sensor']) > 0.2 else 1. for st in states_t])
            # fB = np.mean([2*np.linalg.norm(pos - robot_positions[t-5, i]) for i, pos in enumerate(robot_positions)])
            fB =  np.mean([np.abs(ac['joint_velocity_actuator'][0] - ac['joint_velocity_actuator'][1]) < 0.1 for ac in actions_t])
            # fB = np.mean([0.5*LA.norm(ac['joint_velocity_actuator'], ord=1) * (1 - np.abs(np.diff(ac['joint_velocity_actuator'])) / 2) ** 2 for ac in actions_t])
            fitness += fA * fB
        fitness /= len(states)
        return fitness + 1e-5





@fitness_func_registry(name='comm_sync')
class CommSync:
    """ Consensus state communication"""
    def __init__(self):
        # self.symbols = np.array([
        #     [0,1,1,0],
        #     [1,0,0,1]
        # ])
        self.required_info = ()

    def __call__(self, actions, states, info=None):
        """Computes the fitness function based on trial actions and states.
        Additionally, other useful variables can be used from info dict (if specified in init).
        =======================================================================================
        - Args:
            actions [list of dicts]: list of dictionaries with actuator names and the
                    corresponding action.
            states [list of dicts]: list of dictionaries with sensor names and the
                    corresponding measured states.
            info [dict or None]: dict of additional information.
        =======================================================================================
        """
        fitness = 0
        target = np.mean([rob_st['own_state'] for rob_st in states[0]], 0)
        for states_t in states:
            robot_states = [rob_st['own_state'] for rob_st in states_t]
            # mean_swarm_st = np.mean(robot_states, 0)
            F_t = 1 - np.mean([np.linalg.norm(st - target)/np.sqrt(len(st)) for st in robot_states])
            fitness += F_t ** 3
        fitness /= len(states)
        return fitness + 1e-5