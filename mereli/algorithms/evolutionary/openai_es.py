import logging
import copy
import numpy as np
from .population import OpenAI_ES_Population 
from .evolutionary_algorithm import EvolutionaryAlgorithm
from mereli.algorithms.interfaces import GeneticInterface
from mereli.register import algorithm_registry
from mereli.utils import save_pickle, load_pickle

@algorithm_registry(name='openai_es')
class OpenAI_ES(EvolutionaryAlgorithm):
    def __init__(self, *args, sigma=0.1, learning_rate=0.1,
                 **kwargs):
        super(OpenAI_ES, self).__init__(*args, **kwargs)
        self.mu = None
        self.sigma = sigma 
        self.learning_rate = learning_rate
        self.population = []

    def sample(self):
        sample = np.array([np.random.randn(len(self.mu)) for _ in range(self.pop_size)])
        return (self.mu + self.sigma * sample, sample)
    
    def evolve(self):
        # self.best = sorted(copy.deepcopy(self.population), key=lambda genotype: genotype.fitness, reverse=True)[0]
        fitness_vector = [geno.fitness for geno in self.population]
        fitness_order = np.argsort(fitness_vector.copy())[::-1]
        ord_samples = [self.z_samples[idx].copy() for idx in fitness_order]
        ord_fitness = np.array([fitness_vector[idx] for idx in fitness_order])

        utilities = np.array([((max(0, np.log(1 + 0.5 * len(self.population)) - np.log(i + 1)))\
                    / np.sum([max(0, np.log(1 + 0.5 * len(self.population)) - np.log(j + 1))\
                    for j in range(len(self.population))]))\
                    for i in range(len(self.population))])
        utilities -= 1 / len(self.population)

        #* --- Update distribution -- *#
        self.mu += (self.learning_rate / (self.sigma * len(ord_fitness))) * np.sum([ui * sample \
                for ui, sample in zip(utilities, ord_samples)], 0)
        self.mu = np.clip(self.mu, a_min=0, a_max=1)

        #* --- Sample New population -- *#
        self.samples, self.z_samples = self.sample()
        self.samples = [np.clip(v, a_min=0, a_max=1) for v in self.samples]
        self.update_population() 

    def initialize(self, *args, **kwargs):
        """ """
        super().initialize(*args, **kwargs)
        self.mu = {}
        for info in self.population[0].gene_info.keys():
            info_split = info.split(':')
            # self.mu[info] = np.zeros(len([*getattr(self.population[0], info_split[1])]))
            self.mu = 0.5 * np.ones(len([*getattr(self.population[0], info_split[1])]))
        if self.rank == 0:
            #! OJO PROV
            self.samples, self.z_samples = self.sample()
            self.samples = [np.clip(v, a_min=0, a_max=1) for v in self.samples]
            self.update_population() 

    def update_population(self):
        """TODO: Docstring for function.

        :arg1: TODO
        :returns: TODO

        """
        #! OJO PROV
        for sample, geno in zip(self.samples, self.population):
            for new_val, conn in zip(sample, geno.connections):
                conn.parameters['weight'] = new_val 

    @property
    def checkpoint_data(self):
        return {**super().checkpoint_data, **{'mu' : self.mu, 'sigma' : self.sigma, 
            'learning_rate' : self.learning_rate}}     
 

    # def save_population(self, generation):
    #     pop_checkpoint = {
    #         'populations' : {name : np.stack(pop.population) for name, pop in self.populations.items()},
    #         'generation' : generation,
    #         'mutation_prob' : {name : pop.mutation_prob for name, pop in self.populations.items()},
    #         'evolution_hist' : self.evolution_history,
    #         'mu' : {name : pop.mu for name, pop in self.populations.items()},
    #         'sigma' : {name : pop.sigma for name, pop in self.populations.items()},
    #         'learning_rate' :{name : pop.learning_rate for name, pop in self.populations.items()},
    #     }
    #     file_name = 'mereli/checkpoints/populations/' + self.checkpoint_name
    #     save_pickle(pop_checkpoint, file_name)
    #     logging.info('Successfully saved evolution checkpoint.')

    # def load_population(self):
    #     checkpoint = load_pickle('mereli/checkpoints/populations/' + self.checkpoint_name)
    #     logging.info('Resuming OpenaiES evolution using checkpoint ' +  self.checkpoint_name)
    #     key = tuple(self.populations.keys())[0]
    #     for key, pop in checkpoint['populations'].items():
    #         self.populations[key].mu = checkpoint['mu'][key]
    #         self.populations[key].sigma = checkpoint['sigma'][key]
    #         self.populations[key].learning_rate = checkpoint['learning_rate'][key]
    #         robots = [copy.deepcopy(robot) for robot in self.world.robots.values()]
    #         interface = GeneticInterface(robots[0].controller.neural_network)
    #         self.populations[key].segment_lengths = [interface.submit_query(query, primitive='LEN')\
    #                     for query in self.populations[key].objects]
    #         # import pdb; pdb.set_trace()
    #         self.populations[key].mu = (self.populations[key].mu - self.populations[key].min_vector) / (self.populations[key].max_vector - self.populations[key].min_vector)
    #         # import pdb; pdb.set_trace()
    #         # self.sigma = 1e-3
    #         self.populations[key].population, self.populations[key].z_samples = self.populations[key].sample()
    #         self.populations[key].population = [np.clip(v, a_min=0, a_max=1) for v in self.populations[key].population]
    #     self.init_generation = checkpoint['generation']
    #     self.evolution_history = checkpoint['evolution_hist']
