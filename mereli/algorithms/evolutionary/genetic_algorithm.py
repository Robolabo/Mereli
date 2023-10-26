import copy
import numpy as np
from itertools import chain
from mereli.utils.decorators import time_elapsed
from .evolutionary_algorithm import EvolutionaryAlgorithm
from mereli.register import algorithm_registry, evo_operators

@algorithm_registry(name='GA')
class GeneticAlgorithm(EvolutionaryAlgorithm):
    """ Class of the Canonical Genetic Algorithm. The evolution step is defined in the Population class.
    """
    def __init__(self, *args, 
            selection='roulette',
            crossover='multipoint', 
            mutation='gaussian',
            mating='random',
            mutation_prob=0.05, 
            crossover_prob=1,
            num_elite=2, **kwargs):
        super(GeneticAlgorithm, self).__init__(*args, **kwargs)
        self.selection = evo_operators[selection+ '_selection']
        self.mutation = evo_operators[mutation+ '_mutation']
        self.crossover = evo_operators[crossover+ '_crossover']
        self.mating = evo_operators[mating+ '_mating']
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.num_elite = num_elite
        self.population = []

    @time_elapsed
    def evolve(self):
        #* --- Save elite based on highest fitness ---
        elites = sorted(copy.deepcopy(self.population), key=lambda genotype: genotype.fitness, reverse=True)[:self.num_elite]
        #* --- Apply Selection operator ---
        parents = self.selection(self.population,len(self.population) - self.num_elite)# int(0.3*self.pop_size))#len(self.population) - self.num_elite)
        #* --- Apply Mating operator ---
        # parents = self.mating(parents)
        #* --- Apply Crossover operator ---
        offspring = []
        i = 0
        # for p1, p2 in zip(parents[::2], parents[1::2]):
        while(len(offspring) < len(self.population)- self.num_elite):
            if i < len(parents)-1:
                p1 = parents[i]
                p2 = parents[i+1]
                i += 1
            else:
                p1 = parents[np.random.randint(len(parents))]
                p2 = parents[np.random.randint(len(parents))]
            offspring.extend(self.crossover(p1,p2, crossover_prob=self.crossover_prob))
        assert len(offspring) == len(self.population)- self.num_elite 
        # if len(parents) % 2 != 0:
        #     offspring.append(parents[-1])
        #* --- Apply mutation operator ---
        total_mutations = 0
        genos_mutated = 0  
        for genotype in offspring:
            geno_mutated = False
            #*Parameter Mutations
            for gene in chain(genotype.connections, genotype.nodes):
                gene_mutations  = gene.mutate()
                geno_mutated = geno_mutated or (gene_mutations > 0)
                total_mutations += gene_mutations
            genos_mutated += geno_mutated
        print(total_mutations, genos_mutated) 
        #* --- Update new population ---
        self.population = elites + offspring
        # Dynamic Mutation Prob.
        # self.mutation_prob = exp_schedule(self.mutation_prob, 0.01)
