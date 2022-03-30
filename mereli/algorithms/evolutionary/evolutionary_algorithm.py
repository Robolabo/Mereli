import time
import copy
import re
import logging
from mereli.algorithms.evaluator import Evaluator

from mereli.algorithms.evolutionary.novelty_search import NoveltySearch
try:
    import multiprocessing
except:
    logging.warning('Running without multiprocessing.')
from collections import deque
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except:
    MPI_AVAILABLE = False
    logging.warning('MPI is not installed. Running without mpi4py.')
import numpy as np
import matplotlib.pyplot as plt
from mereli.algorithms.interfaces import InterfaceFactory
from mereli.utils import flatten_dict, DataLogger, without_duplicates
from mereli.sensors.utils import list_sensors
from mereli.actuators.utils import list_actuators
from mereli.globals import global_states          
from mereli.world import MultiWorldWrapper, SquareArena
from mereli.algorithms.evaluator import Evaluator, MPI_Evaluator

def get_info_old(name, robots, world,):
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
        'robot_positions' : np.stack([bot.position for bot in robots]),
        'robot_orientations' : np.array([bot.orientation for bot in robots]),
        'light_positions' : np.array([light.position for light in world.lights.values()]),
        'green_light_positions' : np.array([light.position for light in world.lights.values() if light.color == 'green']),
        'yellow_light_positions' : np.array([light.position for light in world.lights.values() if light.color == 'yellow']),
        'red_light_positions' : np.array([light.position for light in world.lights.values() if light.color == 'red']),
        'blue_light_positions' : np.array([light.position for light in world.lights.values() if light.color == 'blue'])
    }[name]

def get_info(names, world):
    obj_name = names.split(':')[0]
    obj_var = names.split(':')[1] if len(names.split(':')) > 1 else 'position'
    condition = names.split('@')[1] if '@' in names else None
    objects = world.entities(obj_name)
    if condition is not None:
        obj_var = obj_var.split('@')[0]
        objects = filter(lambda x: {
            't' : str(world.t) == condition.split('=')[1],
            'color' : x.color == condition.split('=')[1]
        }.get(condition.split('=')[0], True), objects.values())
        return np.array([getattr(v, obj_var) for v in objects if hasattr(v, obj_var)])
    return np.array([getattr(v, obj_var) for v in objects.values() if hasattr(v, obj_var)])


    

def _run_worker(genotype, worlds, eval_steps, \
        num_evaluations, fitness_fn, seed, generation, algorithm):
    """
    Worker function to evaluate an individual of the EA population and
    compute its fitness.
    =====================================================================
    - Args:
        env_id [int] -> if parallelized, the id of the genotype to eval.
        populations [dict] -> population dict storing all the subpopulations of the EA.
        world [World] -> world object to evaluate fitness.
        num_evaluations [int] -> number of eval repetitions or samples to average the fitness.
        fitness_fn [Fitness] -> fitness class or function to quantify evaluation performance.
        seed [int] -> random state to intialize the world equally for all individuals
                      in the population.
    - Returns:
        Tuple (env_id [int], fitness [float]) with the genotype id 
        and the resulting fitness.
    =====================================================================
    """
    
    t0 = time.time()
    if isinstance(worlds, MultiWorldWrapper):
        if MPI.COMM_WORLD.Get_size() > 1:
            rank = MPI.COMM_WORLD.Get_rank()
            # print('INFO: ', env_id, rank, flush=True)
            world = copy.deepcopy(worlds.all[rank])
        else:
            rank = multiprocessing.current_process()._identity[0]
            world = copy.deepcopy(worlds.all[(rank - 1) % worlds.n_cpu + 1])
    else:        
        world = worlds
    assert not world.physics_engine.connected 
    world.connect()
    # world.reset(seed=seed)
    robots = [robot for robot in world.robots.values()]
    interfaces = [InterfaceFactory().create(algorithm, bot.controller.neural_network) for bot in robots]
    for interface in interfaces:
        interface.fromGenotype(genotype)
    # print('Stuff:', time.time() - t0, flush=True)
    fitness = 0
    mean_survival_time = 0
    t0 = time.time()
    #* Evaluate gentoype several times and average
    for rep in range(num_evaluations):
        seed += 1
        
        world.reset(seed=seed)
        actions_history = deque()
        states_history = deque()
        info = {n : deque() for n in fitness_fn.required_info}
        info['generation'] = generation
        survival_time = 0
        while (not world.is_done and survival_time <= eval_steps):
            states, actions = world.step()
            for key, val in info.items():
                if isinstance(val, deque):
                    val.append(get_info(key, world))
            actions_history.append(actions)
            states_history.append(states)
            survival_time += 1
        mean_survival_time += survival_time
        fitness += fitness_fn(actions_history, states_history, info=info)
    
    mean_survival_time /= num_evaluations
    fitness /= num_evaluations
    world.disconnect()
    genotype.fitness = fitness
    genotype.novelty_variables = {'eval_time' : mean_survival_time}
    # print('Eval:', time.time() - t0, flush=True)
    return genotype

class EvolutionaryAlgorithm:
    """ Base class for evolutionary algorithms """
    def __init__(self, populations, world,
                 n_generations=100,
                 population_size=100,
                 eval_steps=500,
                 num_evaluations=3,
                 n_processes=1,
                 fitness_fn=None,
                 use_novelty_search=False,
                 checkpoint_name='chk',
                 resume=False):
        self.world = world
        evaluator_cls = MPI_Evaluator if self.use_mpi else Evaluator
        self.evaluator = evaluator_cls(num_evaluations=num_evaluations)
        self.populations = populations
        self.n_generations = n_generations
        self.population_size = population_size
        self.eval_steps = eval_steps
        self.num_evaluations = num_evaluations
        self.novelty_search = NoveltySearch() if use_novelty_search else None
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
            # use_mpi = MPI.COMM_WORLD.Get_size() > 1 if MPI_AVAILABLE else False
            #* Only one core is responsible of initialization
            if not self.use_mpi or MPI.COMM_WORLD.Get_rank() == 0:
                robots = [copy.deepcopy(robot) for robot in world.robots.values()]\
                            if not isinstance(world, MultiWorldWrapper) else\
                            [copy.deepcopy(robot) for robot in world.all[0].robots.values()]
                for pop in self.populations.values():
                    pop.initialize(InterfaceFactory().create(type(self).__name__, robots[0].controller.neural_network))
            if self.use_mpi:
                self.populations = MPI.COMM_WORLD.bcast(self.populations, root=0)
                MPI.COMM_WORLD.barrier()

    def create_world(self, world_config, ann_config=None):
        self.evaluator.create_world(world_config, ann_config=ann_config)


    def run(self):
        """ Run method common to all evolutionary computation algs. It parallelizes the 
        genotype evaluation to obtain the fitness and performs the evolution step. 
        The precise method evolve has to be defined in the population class of the 
        precise algorithm that inherits from this class. 
        The method does not return any data. Instead, it saves all the required 
        information to resume the evolution periodically.
        """
        # use_mpi = MPI.COMM_WORLD.Get_size() > 1 if MPI_AVAILABLE else False
        alg_name = type(self).__name__
        for k in range(self.init_generation, self.n_generations):
            t0 = time.time()
            seed = (k) * self.num_evaluations
            eval_genotypes = self.evaluator.batch_evaluate(self.populations['p1'].population, seed, alg_name)
            if not self.use_mpi or MPI.COMM_WORLD.Get_rank() == 0:
                #* Apply novelty search (if any)
                if self.novelty_search is not None:
                    for geno in eval_genotypes:
                        self.novelty_search.update(geno.novelty_metric['eval_time'])
                        ns_metric = self.novelty_search.novelty_metric(geno.novelty_metric['eval_time'])
                        geno.fitness = .3 * geno.fitness + .7 * ns_metric
                    self.populations['p1'].population = eval_genotypes #!
                #* Save evolution state 
                if k % 5 == 0 and self.checkpoint_name is not None:
                    if self.use_mpi:
                        print('SAVING CHECKPOINT', flush=True)
                    self.save_population(k)
                #* Evolve Population
                mean_fitness, max_fitness, min_fitness = self.evolve(k)
                #* Print Stuff
                print('End of generation {} with mean fitness {} and max finess {} in {} seconds.'\
                    .format(k, round(mean_fitness, 3), round(max_fitness, 3), round(time.time() - t0, 2)), flush=True)
                any([self.evolution_history[stat_name].append(stat) for stat_name, stat in \
                            zip(['mean', 'max', 'min'], [mean_fitness, max_fitness, min_fitness])])

            if self.use_mpi:
                #* Broadcast evolved populations to all nodes
                self.populations = MPI.COMM_WORLD.bcast(self.populations, root=0)
                MPI.COMM_WORLD.Barrier()

    def evolve(self, generation):
        for pop in self.populations.values():
            pop.step(self.fitness, generation)
        mean_fitness = np.mean(self.fitness)
        max_fitness = np.max(self.fitness)
        min_fitness = np.min(self.fitness)
        self.fitness = [0 for _ in range(self.population_size)]

        return mean_fitness, max_fitness, min_fitness

    @property
    def use_mpi(self):
        return MPI.COMM_WORLD.Get_size() > 1 if MPI_AVAILABLE else False

    def save_population(self, generation):
        """ Save the algorithm checkpoint. To be implemented in the particular algorithm. """
        raise NotImplementedError

    def load_population(self):
        """ Load the algorithm checkpoint. To be implemented in the particular algorithm. """
        raise NotImplementedError

    def evaluate(self, trials=30, timesteps=3000):
        """ Evaluates an individual of a population without any evolution. 
        Records the data for the specified amount of evaluation trials and time steps and 
        saves all the data records as a csv dataset (stored in mereli/logs/data).
        ============================================================
        - Args:
            trials [int] -> number of evaluation trials.
            timesteps [int] -> number of evaluation timesteps.
        - Returns: None
        ============================================================
        """
        # import pdb; pdb.set_trace()
        world = self.world
        robots = [*world.robots.values()] #[robot for robot in world.hierarchy.values() if robot.trainable]
        world.connect()
        # world.reset()
        interfaces = [InterfaceFactory().create(type(self).__name__, bot.controller.neural_network) for bot in robots]
        for interface in interfaces:
            for pop in self.populations.values():
                aux_pop = sorted(pop.population,key=lambda x: x.fitness)[::-1]
                # pop.species[0].compatibility(aux_pop[0])
                geno = aux_pop[0] # pop.best if pop.best is not None else pop.population[1] # pop.population[150]
                interface.fromGenotype(geno)
        info = {n : deque() for n in self.fitness_fn.required_info}
        info['generation'] = 1
        
        sensor_names, actuator_names = list_sensors(robots[0]), list_actuators(robots[0])
        #! Change list_sensors and actuators to add comm:msg
        # sensor_names = [sens for sens in sensor_names if 'IR_receiver' not in sens]
        # # actuator_names = [act for act in actuator_names if 'IR_transmitter' not in act]
        # actuator_names = [act for act in actuator_names if 'state' not in act]
        #--------------------------------------------------
        lights = self.world.lights
        cubes = self.world.entities('cube')
        fieldnames = ['trial', 'timestep', 'entity', 'position_x', 'position_y', 'orientation'] + sensor_names + actuator_names 
        fieldnames = fieldnames + [y for x in [['position_x_'+name, 'position_y_'+name] for name in {**lights, **cubes}] for y in x]
        data_logger = DataLogger(fieldnames)
        for trial in range(trials):
            
            eval_hist = {'actions': [], 'states': []} # For fitness function not recording
            info = {n : deque() for n in self.fitness_fn.required_info}
            world.reset()
            for timestep in range(timesteps):
                if world.is_done:
                    break
                states, actions = world.step()
                for key, val in info.items():
                    if isinstance(val, deque):
                        val.append(get_info(key, world))
                # for robot, state, action in map(lambda x: (x[0], flatten_dict(x[1]), flatten_dict(x[2])), zip(world.robots.items(), states, actions)):
                #     re_split = lambda x: re.split('_\d|_[a-z]$', x)[0]
                #     st = np.hstack([state[s] for s in without_duplicates(map(re_split, sensor_names)) if s in state.keys()])
                #     ac = np.hstack([action[a] for a in without_duplicates(map(re_split, actuator_names)) if a in action.keys()])
                    #! Provisionally commented
                    # row_values = chain([trial, timestep], [robot[0]], np.hstack((robot[1].position[:2], robot[1].orientation[-1], st, ac)))
                    # row_dict = {key: val for key, val in zip(fieldnames, row_values)}
                    # for name, obj in {**lights, **cubes}.items():
                    #     row_dict.update({'position_x_'+ name : obj.position[0], 'position_y_'+ name : obj.position[1]})
                    # data_logger.update(row_dict)
                eval_hist['states'].append(states)
                eval_hist['actions'].append(actions)
            self.fitness_fn(eval_hist['actions'], eval_hist['states'], info=info)
            print('End of evaluation trial ' + str(trial))
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
        fitness_max = np.array([0]+self.evolution_history['max'])
        fitness_min = np.array(self.evolution_history['min'])
        if smoothed:
            fitness_mean = np.array([fitness_mean[i-5:i].mean() for i in range(5, len(fitness_mean))])
            fitness_max = np.array([fitness_max[i-5:i].mean() for i in range(5, len(fitness_max))])
            fitness_min = np.array([fitness_min[i-5:i].mean() for i in range(5, len(fitness_min))])
        plt.plot(fitness_max)
        plt.ylim([0,1])
        # plt.fill_between(range(len(fitness_mean)), fitness_min, fitness_max, color='blue', alpha=.1)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.show()
    
    def plot_species_evolution(self, smoothed=True):
        species = self.populations['p1'].species
        for spc in species:
            fn_ts = np.array(spc.history['max_fitness'])
            if smoothed:
                fn_ts = np.array([fn_ts[i-5:i].mean() for i in range(5, len(fn_ts))])        
            plt.plot(np.arange(spc.creation_generation, spc.creation_generation + len(fn_ts)), fn_ts)
        plt.show()