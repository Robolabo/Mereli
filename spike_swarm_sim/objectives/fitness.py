from itertools import combinations
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plot
from spike_swarm_sim.register import fitness_func_registry
from spike_swarm_sim.utils import (normalize, compute_angle, angle_diff,
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
        initial_timestep = 100 # ignore previous timesteps for fitness computation
        for t, (thetas, action) in enumerate(zip(robot_orientations[initial_timestep:], \
                        np.array(actions)[initial_timestep:]), start=initial_timestep):
            angle_errs = np.max([angle_diff(th1, th2)
                                for j, th1 in enumerate(thetas)
                                for i, th2 in enumerate(thetas) if i != j])

            fA = np.clip(1 - (angle_errs / (0.4*np.pi)), a_min=0, a_max=1)
            fB = np.mean([np.clip(1 - np.abs(ac['wheel_actuator'][0]), a_min=0, a_max=1) for ac in action])
            fitness += (fA*fB)
        fitness /= (len(robot_orientations)-initial_timestep)
        return fitness + 1e-5

@fitness_func_registry(name='goto_light')
class GotoLight:
    """Fitness function for the light follower task."""
    def __init__(self):
        self.required_info = ("generation", "robot_positions", "robot_orientations", "light_positions")

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
        light_positions = np.stack(info["light_positions"]).copy()
        fitness = 0
        for t, (pos, light_pos)  in enumerate(zip(robot_positions, light_positions)):
            distances = [LA.norm(toroidal_difference(pos_i, light_pos)) for i, pos_i in enumerate(pos)]
            # distances_robots = [LA.norm(pos_i - pos_j) for i, pos_i in enumerate(pos)  for j, pos_j in enumerate(pos) if i != j]
            fitness += np.mean(np.array([dist < 80 for dist in distances]))
        return fitness / len(states)

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

# @fitness_func_registry(name='line_formation')
# class LineFormation:
#     def __init__(self, eval_steps):
#         self.required_info = ("robot_positions", "robot_orientations",)

#     def __call__(self, actions, states, info=None):
#         fitness = 0 
#         from sklearn.linear_model import LinearRegression
#         robot_positions = np.stack(info["robot_positions"]).copy()
#         robot_orientations = np.stack(info["robot_orientations"]).copy()
#         for _, (pos, action)  in enumerate(zip(robot_positions, actions)):
#             lr = LinearRegression().fit(pos[:, 0][:, np.newaxis], pos[:, 1])
#             f_score = lr.score(pos[:, 0][:, np.newaxis], pos[:, 1]) ** 2
#             f_dist = float(np.min([LA.norm(pos_i - pos_j)\
#                             for i, pos_i in enumerate(pos)\
#                             for j, pos_j in enumerate(pos) if i != j]) > 50.)
#             fitness += (f_score * f_dist)
#         fitness /= robot_positions.shape[0]
#         return fitness + 1e-5

# @fitness_func_registry(name='simple_foraging')
# class SimpleForaging:
#     def __init__(self, positions, eval_steps):
#         self.required_info = ()

#     def __call__(self, actions, states, info=None):
#         fitness = 0
#         prev_food_state = np.zeros(len(actions[0]))
#         # import pdb; pdb.set_trace()
#         for t, (states_timestep, actions_timestep)  in enumerate(zip(states, actions)):
#             food_state = np.stack([state['food_sensor'][0] for state in states_timestep])
#             # if any(food_state > 0 ):import pdb; pdb.set_trace()
#             for robot_food, prev_robot_food in zip(food_state, prev_food_state):
#                 # if robot_food - prev_robot_food == 1: # reward for picking food
#                 #     fitness += 5.
#                 if robot_food - prev_robot_food == -1: # reward for dropping food
#                     fitness += 20.
#                 # elif (robot_food == prev_robot_food) and robot_food == 1: # penalize keeping food
#                 #     fitness -= 0.05
#                 fitness = np.clip(fitness, a_min=0, a_max=None)
#             prev_food_state = food_state.copy()
#         fitness /= 100#len(actions)
#         return fitness + 1e-5