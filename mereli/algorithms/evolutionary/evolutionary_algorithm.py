from heapq import merge
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from mereli.register import algorithm_registry
# try:
#     import multiprocessing
# except:
#     logging.warning('Running without multiprocessing.')
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except:
    MPI_AVAILABLE = False
    logging.warning('MPI is not installed. Running without mpi4py.')
from mereli.utils import save_pickle, load_pickle
from mereli.algorithms.evaluator import Evaluator
from mereli.algorithms.evolutionary.gene import GraphGenotype
from mereli.algorithms.evolutionary.novelty_search import NoveltySearch
from mereli.utils import DataLogger
from mereli.sensors.utils import list_sensors
from mereli.actuators.utils import list_actuators
from mereli.algorithms.evaluator import Evaluator, MPI_Evaluator


class EvolutionaryAlgorithm:
    """ Base class for evolutionary algorithms """
    def __init__(self, world, n_generations, population_size, targets,
                 num_evaluations=1,
                 fitness_fn=None,
                 novelty_search=None,
                 checkpoint_name='chk',
                 resume=False):
        self.world = world
        self.resume = resume
        self.targets = targets
        self.generation = 0
        self.n_generations = n_generations
        self.population_size = population_size
        self.checkpoint_name = checkpoint_name
        evaluator_cls = MPI_Evaluator if self.use_mpi else Evaluator
        self.evaluator = evaluator_cls(world, num_evaluations=num_evaluations, fitness_fn=fitness_fn)
        self.novelty_search = NoveltySearch(**novelty_search) if novelty_search is not None else None
        self.evolution_history = {stat : [] for stat in ['mean', 'max', 'min']}

    def initialize(self, gene_info, neural_net_config):
        assert self.world is not None
        if self.rank == 0:
            if self.resume:
                self.load()
            else:
                for g_id in range(self.population_size):
                    genotype = GraphGenotype(g_id)
                    genotype.configure(self.targets, gene_info, neural_net_config)
                    genotype.initialize()
                    self.population.append(genotype)
        if self.use_mpi:
            self.population = MPI.COMM_WORLD.bcast(self.population, root=0)
            MPI.COMM_WORLD.barrier()

    def create_world(self, world_config, ann_config=None):
        self.evaluator.create_world(world_config, ann_config=ann_config)

    def evaluate(self):
        self.population = self.evaluator.batch_evaluate(self.population, self.generation)

    def run_step(self):
        t0 = time.time()
        #* Evaluate all the genotypes
        self.evaluate()

        if self.rank == 0:
            #* Apply novelty search (if any)
            if self.novelty_search is not None:
                for geno in self.population:
                    self.novelty_search.update(geno.novelty_variables)
                    ns_metric = self.novelty_search.novelty_metric(geno.novelty_variables)
                    ns_weight = self.novelty_search.weight
                    geno.fitness = (1 - ns_weight) * geno.fitness + ns_weight * ns_metric
            #* Save evolution state 
            self.save()
            time_taken = time.time() - t0
            #* Print Stuff
            print(f"Generation {self.generation+1}: mean fitness={self.mean_fitness:3f},",
                    f"max finess={self.max_fitness:3f}, time elapsed={time_taken:2f} s.", 
                    flush=True)
            #* Evolve Population
            self.evolve()
            self.generation += 1
        self.broadcast_population()
    
    def broadcast_population(self):
        if self.use_mpi:
            #* Broadcast evolved population to all nodes
            self.population = MPI.COMM_WORLD.bcast(self.population, root=0)
            MPI.COMM_WORLD.Barrier()

    def run(self):
        """ Run method common to all evolutionary computation algs. It parallelizes the 
        genotype evaluation to obtain the fitness and performs the evolution step. 
        The precise method evolve has to be defined in the population class of the 
        precise algorithm that inherits from this class. 
        The method does not return any data. Instead, it saves all the required 
        information to resume the evolution periodically.
        """
        while self.generation <= self.n_generations:
            self.run_step()

    def evolve(self):
        raise NotImplementedError

    def save(self):
        """ Saves the checkpoint with the necessary information to resume the evolution. 
        """
        self.evolution_history['mean'].append(self.mean_fitness)
        self.evolution_history['max'].append(self.max_fitness)
        self.evolution_history['min'].append(self.min_fitness)
        if self.generation % 5 == 0 and self.checkpoint_name is not None:
            file_name = 'mereli/checkpoints/populations/' + self.checkpoint_name
            save_pickle(self.checkpoint_data, file_name)
            logging.info('Successfully saved evolution checkpoint.')
                
    def load(self):
        """ Loads a previously saved checkpoint to resume evolution.
        """
        checkpoint = load_pickle('mereli/checkpoints/populations/' + self.checkpoint_name)
        logging.info('Resuming NEAT evolution using checkpoint ' +  self.checkpoint_name)
        for key, value in checkpoint.items():
            setattr(self, key, value) 
        logging.info('Evolution checkpoint successfully restored.')
    
    @property
    def checkpoint_data(self):
        return {
            'generation' : self.generation,
            'population' : self.population,
            'novelty_search' : self.novelty_search,
            'evolution_hist' : self.evolution_history,
        }

    @property
    def pop_size(self):
        return len(self.population)

    @property
    def mean_fitness(self):
        return np.mean([geno.fitness for geno in self.population])

    @property
    def max_fitness(self):
        return np.max([geno.fitness for geno in self.population])

    @property
    def min_fitness(self):
        return np.min([geno.fitness for geno in self.population])

    @property
    def use_mpi(self):
        return MPI.COMM_WORLD.Get_size() > 1 if MPI_AVAILABLE else False

    @property
    def rank(self):
        return MPI.COMM_WORLD.Get_rank() if MPI_AVAILABLE else 0

    def validate(self, trials=30, timesteps=3000):
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
        robots = [robot for robot in self.world.robots.values()] 
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
        self.evaluator.use_seed = False
        if type(self).__name__ == 'MultiEA':
            best = [sorted(alg.population, key=lambda x: x.fitness, reverse=True)[0] for alg in self.algorithms]
        else: 
            best, second, third = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:3]
        get_weights = lambda x: np.array([x.parameters['weight'] for x in x.connections])
        __import__('pdb').set_trace()
        self.evaluator.evaluate(best, 0)
        import pdb; pdb.set_trace()
        # for trial in range(trials):
            
        #     eval_hist = {'actions': [], 'states': []} # For fitness function not recording
        #     info = {n : deque() for n in self.fitness_fn.required_info}
        #     world.reset()
        #     for timestep in range(timesteps):
        #         if world.is_done:
        #             break
        #         states, actions = world.step()
        #         for key, val in info.items():
        #             if isinstance(val, deque):
        #                 val.append(get_info(key, world))
        #         # for robot, state, action in map(lambda x: (x[0], flatten_dict(x[1]), flatten_dict(x[2])), zip(world.robots.items(), states, actions)):
        #         #     re_split = lambda x: re.split('_\d|_[a-z]$', x)[0]
        #         #     st = np.hstack([state[s] for s in without_duplicates(map(re_split, sensor_names)) if s in state.keys()])
        #         #     ac = np.hstack([action[a] for a in without_duplicates(map(re_split, actuator_names)) if a in action.keys()])
        #             #! Provisionally commented
        #             # row_values = chain([trial, timestep], [robot[0]], np.hstack((robot[1].position[:2], robot[1].orientation[-1], st, ac)))
        #             # row_dict = {key: val for key, val in zip(fieldnames, row_values)}
        #             # for name, obj in {**lights, **cubes}.items():
        #             #     row_dict.update({'position_x_'+ name : obj.position[0], 'position_y_'+ name : obj.position[1]})
        #             # data_logger.update(row_dict)
        #         eval_hist['states'].append(states)
        #         eval_hist['actions'].append(actions)
        #     self.fitness_fn(eval_hist['actions'], eval_hist['states'], info=info)
        #     print('End of evaluation trial ' + str(trial))
        # data_logger.save(self.checkpoint_name, len(robots))
        # import pdb; pdb.set_trace()

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
        species = self.species
        for spc in species:
            fn_ts = np.array(spc.history['max_fitness'])
            if smoothed:
                fn_ts = np.array([fn_ts[i-5:i].mean() for i in range(5, len(fn_ts))])        
            plt.plot(np.arange(spc.creation_generation, spc.creation_generation + len(fn_ts)), fn_ts)
        plt.show()

@algorithm_registry(name="multi_EA")
class MultiEA(EvolutionaryAlgorithm):
    def __init__(self, *args, **kwargs):
        super(MultiEA, self).__init__(*args, **kwargs)
        self.algorithms = []

    def merge_population(self):
        aux_pop = []
        for p in range(self.pop_size):
            aux_pop.append([alg.population[p] for alg in self.algorithms])
        return aux_pop

    def add_algorithm(self, alg):
        self.algorithms.append(alg)

    def evaluate(self):
        merged_pop = self.merge_population()
        merged_pop = self.evaluator.batch_evaluate(merged_pop, self.generation)
        if self.use_mpi and self.rank == 0:
            for p in range(self.pop_size):
                for alg, genotype in zip(self.algorithms, merged_pop[p]):
                    alg.population[p] = genotype
            # self.population = merged_pop

    def broadcast_population(self):
        if self.use_mpi:
            #* Broadcast evolved population to all nodes
            self.algorithms = MPI.COMM_WORLD.bcast(self.algorithms, root=0)
            MPI.COMM_WORLD.Barrier()

    def evolve(self):
        for alg in self.algorithms:
            alg.evolve()

    def save(self):
        """ Saves the checkpoint with the necessary information to resume the evolution. 
        """
        for k in range(len(self.algorithms)):
            self.algorithms[k].checkpoint_name = self.checkpoint_name + '_' + str(k+1)
            self.algorithms[k].save()
                
    def load(self):
        """ Loads a previously saved checkpoint to resume evolution.
        """
        import pdb; pdb.set_trace()
        for k in range(len(self.algorithms)):
            self.algorithms[k].checkpoint_name = self.checkpoint_name + '_' + str(k+1)
            self.algorithms[k].load()
        

    @property
    def pop_size(self):
        return len(self.algorithms[0].population)

    @property
    def mean_fitness(self):
        return np.mean([geno.fitness for geno in self.algorithms[0].population])

    @property
    def max_fitness(self):
        return np.max([geno.fitness for geno in self.algorithms[0].population])

    @property
    def min_fitness(self):
        return np.min([geno.fitness for geno in self.algorithms[0].population])
