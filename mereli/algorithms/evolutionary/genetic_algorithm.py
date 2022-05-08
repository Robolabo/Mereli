import copy
from itertools import chain
from .evolutionary_algorithm import EvolutionaryAlgorithm
from mereli.register import algorithm_registry, evo_operators

@algorithm_registry(name='GA')
class GeneticAlgorithm(EvolutionaryAlgorithm):
    """ Class of the Canonical Genetic Algorithm. The evolution step is defined in the Population class.
    """
    def __init__(self, *args, 
            selection_operator='roulette',
            crossover_operator='multipoint', 
            mutation_operator='gaussian',
            mating_operator='random',
            mutation_prob=0.05, 
            crossover_prob=0.9,
            num_elite=5, **kwargs):
        super(GeneticAlgorithm, self).__init__(*args, **kwargs)
        self.selection_operator = evo_operators[selection_operator + '_selection']
        self.mutation_operator = evo_operators[mutation_operator + '_mutation']
        self.crossover_operator = evo_operators[crossover_operator + '_crossover']
        self.mating_operator = evo_operators[mating_operator + '_mating']
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.num_elite = num_elite
        self.population = []

    def evolve(self):
        #* --- Save elite based on highest fitness ---
        elites = sorted(copy.deepcopy(self.population), key=lambda genotype: genotype.fitness, reverse=True)[:self.num_elite]
        #* --- Apply Selection operator ---
        parents = self.selection_operator(self.population, len(self.population) - self.num_elite)
        #* --- Apply Mating operator ---
        parents = self.mating_operator(parents)
        #* --- Apply Crossover operator ---
        offspring = []
        for p1, p2 in zip(parents[::2], parents[1::2]):
            offspring.extend(self.crossover_operator(p1,p2, crossover_prob=self.crossover_prob))
        if len(parents) % 2 != 0:
            offspring.append(parents[-1])
        #* --- Apply mutation operator ---
        for genotype in offspring:
            #*Parameter Mutations
            for gene in chain(genotype.nodes, genotype.connections):
                gene.mutate()
        
        #* --- Update new population ---
        self.population = elites + offspring
        # Dynamic Mutation Prob.
        # self.mutation_prob = exp_schedule(self.mutation_prob, 0.01)


    # def save_population(self, generation):
    #     """ Saves the checkpoint with the necessary information to resume the 
    #     evolution.
    #     """
    #     pop_checkpoint = {
    #         'populations' : {name : np.stack(pop.population) for name, pop in self.populations.items()},
    #         'generation' : generation,
    #         'mutation_prob' : {name : pop.mutation_prob for name, pop in self.populations.items()},
    #         'evolution_hist' : self.evolution_history,
    #     }
    #     file_name = 'mereli/checkpoints/populations/' + self.checkpoint_name
    #     save_pickle(pop_checkpoint, file_name)
    #     logging.info('Successfully saved evolution checkpoint.')
        
    # def load_population(self):
    #     """ Loads a previously saved checkpoint to resume evolution.
    #     """
    #     checkpoint = load_pickle('mereli/checkpoints/populations/' + self.checkpoint_name)
    #     logging.info('Resuming GA evolution using checkpoint ' +  self.checkpoint_name)
    #     #! for robot in self.world.robots:
    #     #!    robot.controller.neural_net.graph = checkpoint['ann_graph']
    #     for name, pop in checkpoint['populations'].items():
    #         self.populations[name].population = [v for v in checkpoint['populations'][name]]
    #         self.populations[name].mutatation_prob = checkpoint['mutation_prob'][name]
    #         robots = [copy.deepcopy(robot) for robot in self.world.robots.values()]
    #         interface = GeneticInterface(robots[0].controller.neural_network)
    #         self.populations[name].segment_lengths = [interface.submit_query(query, primitive='LEN')\
    #                                         for query in self.populations[name].objects]
    #     self.init_generation = checkpoint['generation']
    #     self.evolution_history = checkpoint['evolution_hist']