import copy
import time
import logging
import numpy as np
from itertools import chain
from .evolutionary_algorithm import EvolutionaryAlgorithm
from mereli.register import algorithm_registry
from .species import Species
from mereli.algorithms.evolutionary.species import Species
from .operators.crossover import *
from .operators.mutation import *
from .operators.selection import *

class Innovation:
    def __init__(self):
        self.current = 0
        self.history = {}
    
    def increase(self):
        self.current += 1
    
    def assign(self, pre, post):
        if (pre, post) in self.history:
            return self.history[(pre, post)]
        else: 
            self.increase()
            self.history[(pre, post)] = self.current
            return self.current

@algorithm_registry(name='NEAT')
class NEAT(EvolutionaryAlgorithm):
    """ 
    """
    def __init__(self, *args, survival_rate=0.5, p_weight_mut=0.75, p_node_mut=0.08, 
            p_conn_mut=0.1, compatib_thresh=2, c1=1, c2=1, c3=2, species_elites=0, **kwargs):
        super(NEAT, self).__init__(*args, **kwargs)
        self.survival_rate = survival_rate
        self.p_weight_mut = p_weight_mut
        self.p_node_mut = p_node_mut
        self.p_conn_mut = p_conn_mut
        self.compatib_thresh = compatib_thresh
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.species_elites = species_elites
        self.species_count = 1
        # list of existing species. 1 species at first.
        self.species = []
        self.input_nodes = [] #* Cannot be altered by NEAT 
        self.population = []
        #* Global pointer of gene innovations
        self.innovation = Innovation()

    def evolve(self):
        t0 = time.time()
        offspring = []
        self.best = sorted(copy.deepcopy(self.population), key=lambda genotype: genotype.fitness, reverse=True)[0]
        #* Update species fitness statistics
        for spc in self.species:
            spc_genotypes = [genotype for genotype in self.population if genotype.species == spc.id]
            spc_fitness = np.array([genotype.fitness for genotype in spc_genotypes])
            spc.update_stats(spc_fitness)
        #* Compute the number of offspring for each species
        # species_offsprings = compute_spawn(self.species, self.pop_size, 2)
        total_fitness = np.sum([sp.adjusted_fitness for sp in self.species])
        species_offsprings = np.round(np.array([self.pop_size * sp.adjusted_fitness for sp in self.species]) / total_fitness).astype(int)
        species_offsprings[species_offsprings < 2] = 2
        while sum(species_offsprings) < self.pop_size:
            species_offsprings[np.random.randint(len(self.species))] += 1
        while sum(species_offsprings) > self.pop_size:
            ii = np.random.choice(np.where(species_offsprings > 2)[0])
            species_offsprings[ii] -= 1
        assert sum(species_offsprings) == self.pop_size
        #* Crossover in-between species individuals.
        for n_offspring, spc in zip(species_offsprings, self.species):
            #* Filter out genotypes from species.
            spc_genotypes = self.species_genotypes(spc.id)
            #* Apply species elitism
            if self.species_elites > 0 and n_offspring > self.species_elites:
                # Use dummy iterable range(self.species_elites) to set the loop length to the number of elites
                for _, elite_gnt in zip(range(self.species_elites), sorted(spc_genotypes, key=lambda x: x.fitness, reverse=True)):
                    n_offspring -= 1
                    offspring.append(copy.deepcopy(elite_gnt))
            #* Truncate bests
            # n_sel = max(2, round(self.survival_rate * len(spc_genotypes)))
            parents = tournament_selection(spc_genotypes, n_offspring)
            # parents = truncation_selection(spc_genotypes, n_sel)
            # parents_mating = np.random.choice(n_sel, size=n_offspring, replacement=False) #* Random Mating (OJO REPLACEMENT)
            # parents = [parents[idx] for idx in parents_mating] # shuffle parents
            #* NEAT Crossover
            offspring.extend(neat_crossover(parents))
        #* NEAT Mutation
        offspring, self.innovation = neat_mutation(offspring, self.input_nodes, self.innovation, 
                p_node_mut=self.p_node_mut, p_conn_mut=self.p_conn_mut)
        #* Update population
        self.population = offspring
        if len(self.population) != self.pop_size:
            logging.error('Population Size altered.')
        #* Speciation
        self.update_species()
        logging.info('Num. species is {}'.format(len(self.species)))
        print('NEAT: ', time.time() - t0)
        #* Adaptive species thresh.
        # num_tar_species = 15
        # if len(self.species) != num_tar_species:
        #     self.compatib_thresh += 0.1 * (-1, 1)[len(self.species) > num_tar_species]
        #     self.compatib_thresh = np.clip(self.compatib_thresh, a_min=0.5, a_max=5)
        #     for sp in self.species:
        #         sp.compatib_thresh = self.compatib_thresh
        # print('NEAT ', time.time() - t0)
        # import pdb; pdb.set_trace()        

    def update_species(self):

        #* Assign Species. Use representatives from the previous generation.
        #* If a new species is created the current representative is the genotype 
        #* that created it.
        for spc in self.species:
            if spc.is_extinct:
                logging.info('Extint Species {} due to stagnation.'.format(species.id))
                self.species.pop(i)
            spc.num_genotypes = 0

        for genotype in self.population:
            if len(self.species) != 0:
                compatible, distances = zip(*[spc.compatibility(genotype) for spc in self.species])
            else:
                compatible, distances = [False], None
            
            if not any(compatible): #* create new species
                self.species_count += 1
                new_species = Species(self.species_count, self.generation, compatib_thresh=self.compatib_thresh,
                                    c1=self.c1, c2=self.c2, c3=self.c3)
                new_species.num_genotypes += 1
                new_species.representative = copy.deepcopy(genotype)
                self.species.append(new_species)
                genotype.species = new_species.id
            else:
                compatible_species = np.arange(len(self.species))[list(compatible)]
                compatible_distances = np.array(distances)[list(compatible)]
                species_idx, _ = sorted(zip(compatible_species, compatible_distances), key=lambda x: x[1])[0]
                self.species[species_idx].num_genotypes += 1
                genotype.species = self.species[species_idx].id
        #* Check extintion and update representatives.
        for i, species in enumerate(self.species):
            if species.num_genotypes == 0 or species.is_extinct:
                logging.info('Extint Species {}'.format(species.id))
                self.species.pop(i)
            else:
                assert species.representative is not None
                spc_genotypes = self.species_genotypes(species.id)
                try:
                    compatible, distances = zip(*[species.compatibility(gnt) for gnt in spc_genotypes])
                except:
                    import pdb; pdb.set_trace()
                species.representative = copy.deepcopy(spc_genotypes[np.argmin(distances)])

    def species_genotypes(self, species_id):
        return [genotype for genotype in self.population if genotype.species == species_id]

    def initialize(self, *args, **kwargs):
        """ """
        self.species = []#Species(self.species_count, 0, compatib_thresh=self.compatib_thresh, 
                            #c1=self.c1, c2=self.c2, c3=self.c3)]
        super().initialize(*args, **kwargs)
        self.input_nodes = [*chain(*[[name + '_' + str(num) for num in range(stim['n'])]
                for name, stim in [*args[1].values()][0]['stimuli'].items()])]
        if self.rank == 0:
            #* Only initialize weights randomly, the structure is always the same.
            for genotype in self.population:
                #* Assign innovation numbers
                for conn in genotype.connections:
                    conn.innovation = self.innovation.assign(conn.pre, conn.post)
            #* Initial Speciation
            self.update_species()

    @property
    def checkpoint_data(self):
        return {**super().checkpoint_data, **{'innovation' : self.innovation,'species' : self.species, 'species_count' : self.species_count}}     
 
