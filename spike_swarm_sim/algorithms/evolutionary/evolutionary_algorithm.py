import time
import copy
import re
import multiprocessing
import logging
from collections import deque
from itertools import repeat, chain
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except:
    MPI_AVAILABLE = False
    logging.warning('MPI is not installed. Running without mpi4py.')
import numpy as np
import matplotlib.pyplot as plot
from spike_swarm_sim.algorithms.interfaces import GeneticInterface
from spike_swarm_sim.utils import flatten_dict, DataLogger, without_duplicates
from  spike_swarm_sim.sensors.utils import list_sensors
from  spike_swarm_sim.actuators.utils import list_actuators     
from spike_swarm_sim.globals import global_states          

def get_info(name, robots, world,):
    """
    Returns queried information about the world and its objects.
    #! Provisional implementation, will be improved in the future.
    ====================================
    - Args:
        name [str] -> name of the query.
        robots [dict] -> world robots.
        world [World] -> world under assessment.
    ====================================
    """
    return {
        'robot_positions' : np.stack([bot.pos for bot in robots]),
        'robot_orientations' : np.array([bot.theta for bot in robots]),
        'light_positions' : np.array([light.pos for light in world.lights.values()])
    }[name]

def _run_worker(env_id, populations, world, eval_steps, \
                num_evaluations, fitness_fn, seed, generation):
    """
    Worker function to evaluate an individual of the EA population and
    compute its fitness.
    =====================================================================
    - Args:
        env_id [int] -> if parallelized, the id of the worker.
        populations [dict] -> population dict storing all the subpopulations of the EA.
        world [World] -> world object to evaluate fitness.
        num_evaluations [int] -> number of eval repetitions or samples to average the fitness.
        fitness_fn [Fitness] -> fitness class or function to quantify evaluation performance.
        seed [int] -> random state to intialize the world equally for all individuals
                      in the population.
    - Returns:
        Tuple (env_id [int], fitness [float]) with the worker id that executed the evaluation
        and the resulting fitness.
    =====================================================================
    """
    world.reset()
    robots = [robot for robot in world.robots.values()]
    interfaces = [GeneticInterface(bot.controller.neural_network) for bot in robots]
    for interface in interfaces:
        for pop in populations.values():
            genotype_segment = pop.population[env_id]
            # import pdb; pdb.set_trace()
            interface.fromGenotype(pop.objects, genotype_segment, pop.min_vals, pop.max_vals)
    fitness = 0
    mean_survival_time = 0
    # Evaluate gentoype several times and average
    for rep in range(num_evaluations):
        seed += 1
        world.reset(seed=seed)
        actions_history = deque()
        states_history = deque()
        info = {n : deque() for n in fitness_fn.required_info}
        info['generation'] = generation
        survival_time = 0
        done = False
        while (not done and survival_time <= eval_steps):
            states, actions = world.step()
            for key, val in info.items():
                if isinstance(val, deque):
                    val.append(get_info(key, robots, world))
            actions_history.append(actions)
            states_history.append(states)
            survival_time += 1
            # if done:
            #     break
        mean_survival_time += survival_time
        fitness += fitness_fn(actions_history, states_history, info=info)
    mean_survival_time /= num_evaluations
    fitness = (fitness / num_evaluations)
    return (env_id, fitness)

class EvolutionaryAlgorithm:
    """ Base class for evolutionary algorithms """
    def __init__(self, populations, world,
                 n_generations=100,
                 population_size=100,
                 eval_steps=500,
                 num_evaluations=3,
                 n_processes=1,
                 fitness_fn=None,
                 checkpoint_name='chk',
                 resume=False):
        self.world = world
        self.populations = populations
        self.n_generations = n_generations
        self.population_size = population_size
        self.eval_steps = eval_steps
        self.num_evaluations = num_evaluations
        self.n_processes = n_processes
        self.checkpoint_name = checkpoint_name
        # print('Running with ', self.n_processes, ' cores')
        self.fitness_fn = fitness_fn
        if fitness_fn is None:
            raise Exception('Error: Specify fitness function.')
        self.fitness = [0 for _ in range(self.population_size)]
        self.evolution_history = {stat : [] for stat in ['mean', 'max', 'min']}
        self.init_generation = 0
        if resume:
            self.load_population()
        else:
            robots = [copy.deepcopy(robot) for robot in world.robots.values()]
            for pop in self.populations.values():
                pop.initialize(GeneticInterface(robots[0].controller.neural_network))

    def run(self):
        """
        Run method common to all evolutionary computation algs. It parallelizes the 
        genotype evaluation to obtain the fitness and performs the evolution step. 
        The precise method evolve has to be defined in the population class of the 
        precise algorithm that inherits from this class. 
        The method does not return any data. Instead, it saves all the required 
        information to resume the evolution periodically.
        ===============================================================
        - Args: None
        - Returns: None
        =============================================================== 
        """
        use_mpi = MPI.COMM_WORLD.Get_size() > 1 if MPI_AVAILABLE else False
        for k in range(self.init_generation, self.n_generations):
            fitness = []
            t0 = time.time()
            seed = (k) * self.num_evaluations 
           
            #* MULTIPROCESSING Parallelization
            if not use_mpi and self.n_processes > 1:
                with multiprocessing.Pool(processes=self.n_processes) as pool:
                    pool_args = zip(range(self.population_size), *[map(lambda x: copy.deepcopy(x), repeat(v))\
                                for v in iter([self.populations, self.world, self.eval_steps, self.num_evaluations,\
                                self.fitness_fn, seed, k])])
                    evaluation_res = pool.starmap(_run_worker, pool_args)
                    self.fitness = [v for _, v in sorted(evaluation_res, key=lambda x: x[0])]
            #* MPI Parallelization
            elif use_mpi:
                comm = MPI.COMM_WORLD
                rank = comm.Get_rank()
                size = comm.Get_size()
                comm.Barrier()
                indiv_per_core = (self.population_size // size) #!+ (rank == 0) * (self.population_size % size)
                my_individuals = np.arange(indiv_per_core*rank, indiv_per_core*(rank+1))
                my_fitness = [_run_worker(ii, self.populations, self.world, self.eval_steps, self.num_evaluations,\
                                self.fitness_fn, seed, k) for ii in my_individuals]
                comm.Barrier()
                eval_result = comm.gather(my_fitness, root=0)
                if rank == 0:
                    fitness = [vv for ff in fitness for vv in ff]
                    self.fitness = [f_val for _, f_val in sorted(fitness, key=lambda x: x[0])]
                comm.Barrier()
            else:
                eval_result = [_run_worker(i, self.populations, self.world, self.eval_steps, \
                                self.num_evaluations, self.fitness_fn, seed, k)\
                                for i in range(self.population_size)]
                self.fitness = [v for _, v in eval_result]
            #* No parallelization
            if not use_mpi or MPI.COMM_WORLD.Get_rank() == 0:
                #* Evolve Population
                mean_fitness, max_fitness, min_fitness = self.evolve()
                
                print('End of generation {} with mean fitness {} and max finess {} in {} seconds.'\
                    .format(k, round(mean_fitness, 3), round(max_fitness, 3), round(time.time() - t0, 2)), flush=True)
                any([self.evolution_history[stat_name].append(stat) for stat_name, stat in \
                            zip(['mean', 'max', 'min'], [mean_fitness, max_fitness, min_fitness])])
                if k % 5 == 0 and self.checkpoint_name is not None:
                    self.save_population(k)
            if use_mpi:
                #* Broadcast evolved populations to all nodes
                self.populations = MPI.COMM_WORLD.bcast(self.populations, root=0)

    def evolve(self):
        for pop in self.populations.values():
            pop.step(self.fitness)
        mean_fitness = np.mean(self.fitness)
        max_fitness = np.max(self.fitness)
        min_fitness = np.min(self.fitness)
        self.fitness = [0 for _ in range(self.population_size)]
        return mean_fitness, max_fitness, min_fitness

    def save_population(self, generation):
        """ Save the algorithm checkpoint. To be implemented in the particular algorithm. """
        raise NotImplementedError
    def load_population(self):
        """ Load the algorithm checkpoint. To be implemented in the particular algorithm. """
        raise NotImplementedError

    def evaluate(self, trials=50, timesteps=1000):
        """ Evaluates an individual of a population without any evolution. 
        Records the data for the specified amount of evaluation trials and time steps and 
        saves all the data records as a csv dataset (stored in spike_swarm_sim/logs/data).
        ============================================================
        - Args:
            trials [int] -> number of evaluation trials.
            timesteps [int] -> number of evaluation timesteps.
        - Returns: None
        ============================================================
        """
        world = self.world
        robots = [robot for robot in world.hierarchy.values() if robot.trainable]
        world.reset()
        interfaces = [GeneticInterface(bot.controller.neural_network) for bot in robots]
        for interface in interfaces:
            for pop in self.populations.values():
                genotype_segment = pop.population[1]
                interface.fromGenotype(pop.objects, genotype_segment, pop.min_vals, pop.max_vals)
        # fitness = np.zeros(len(robots))
        info = {n : deque() for n in self.fitness_fn.required_info}
        info['generation'] = 1
        # eval_hist = {'actions': [], 'states': []}
        sensor_names, actuator_names = list_sensors(robots[0]), list_actuators(robots[0])
        fieldnames = ['trial', 'timestep', 'entity', 'position_x', 'position_y', 'orientation'] + sensor_names + actuator_names
        data_logger = DataLogger(fieldnames)
        for trial in range(trials):
            world.reset()
            # eval_hist = {'actions': [], 'states': [], 'positions_x' : [], 'positions_y' : [], 'theta' : []}
            for timestep in range(timesteps):
                states, actions = world.step()
                for key, val in info.items():
                    if isinstance(val, deque):
                        val.append(get_info(key, robots, world))
                for robot, state, action in map(lambda x: (x[0], flatten_dict(x[1]), flatten_dict(x[2])), zip(world.robots.items(), states, actions)):
                    re_split = lambda x: re.split('_\d|_[a-z]$', x)[0]
                    st = np.hstack([state[s] for s in without_duplicates(map(re_split, sensor_names)) if s in state.keys()])
                    ac = np.hstack([action[a] for a in without_duplicates(map(re_split, actuator_names)) if a in action.keys()])
                    row_values = chain([trial, timestep], [robot[0]], np.hstack((robot[1].pos, robot[1].theta, st, ac)))
                    row_dict = {key: val for key, val in zip(fieldnames, row_values)}
                    data_logger.update(row_dict)
        data_logger.save(self.checkpoint_name, len(robots))
        import pdb; pdb.set_trace()

    def plot_learning_curve(self, smoothed=True):
        """ 
        Plots the fitness curve of the evolution. Paints the mean, max and min values.
        It can be smoothed by generational averaging (every 5 generations).
        ======================================
        - Args:
            smoothed [bool] -> flag denoting if the curve should be smoothed.
        - Returns: None
        ======================================
        """
        fitness_mean = np.array(self.evolution_history['mean'])
        fitness_max = np.array(self.evolution_history['max'])
        fitness_min = np.array(self.evolution_history['min'])
        if smoothed:
            fitness_mean = np.array([fitness_mean[i-5:i].mean() for i in range(5, len(fitness_mean))])
            fitness_max = np.array([fitness_max[i-5:i].mean() for i in range(5, len(fitness_max))])
            fitness_min = np.array([fitness_min[i-5:i].mean() for i in range(5, len(fitness_min))])
        plot.plot(fitness_mean)
        plot.fill_between(range(len(fitness_mean)), fitness_min, fitness_max, color='blue', alpha=.1)
        plot.xlabel('Generation')
        plot.ylabel('Fitness')
        # plot.ylim([0, 1])
        plot.show()