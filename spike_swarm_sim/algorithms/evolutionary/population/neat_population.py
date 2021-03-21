import logging
import copy
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns
from scipy.linalg import expm
from .population import Population
from spike_swarm_sim.utils import eigendecomposition, normalize
from spike_swarm_sim.algorithms.evolutionary.species import Species
from ..operators.crossover import *
from ..operators.mutation import *
from ..operators.selection import *


class NEAT_Population(Population):
    """  
    """ 
    def __init__(self, *args, p_weight_mut=0.75, p_node_mut=0.08, p_conn_mut=0.1,
                    compatib_thresh=2, c1=1, c2=1, c3=2, **kwargs):
        super(NEAT_Population, self).__init__(*args, **kwargs)
        self.p_weight_mut = p_weight_mut
        self.p_node_mut = p_node_mut
        self.p_conn_mut = p_conn_mut
        self.compatib_thresh = compatib_thresh
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.species_count = 1
        # list of existing species. 1 species at first.
        self.species = []
        self.input_nodes = [] #* Cannot be altered by NEAT 
        self.population = []
        #* Global pointer of gene innovations
        self.current_innovation = 0
        #* Dict mapping (pre, post) tuple connections to innovation numbers.
        #* It is used for assigning same innovations to mutations already occured in 
        #* the evolution.
        self.innovation_history = {}
        
    def step(self, fitness_vector, generation):
        """
        ==================================================================================
        - Args:
            fitness_vector [np.ndarray or list]: array of computed fitness values.
        - Returns: None
        ==================================================================================
        """
        offspring = []
        self.best = copy.deepcopy(self.population[np.argmax(fitness_vector)])
        raw_fitness = fitness_vector.copy()
        #* Adjust fitness scores according to the fitness sharing as defined in the NEAT paper.
        fitness_vector = [f / [sp for sp in self.species if sp.id == genotype['species']][0].num_genotypes \
                            for f, genotype in zip(fitness_vector, self.population)]
        #* Update species fitness statistics
        for spc in self.species:
            spc_fitness = [ft for ft, gt in zip(fitness_vector, self.population) if gt['species'] == spc.id]
            spc.update_stats(np.array(spc_fitness) * spc.num_genotypes)

        #* DISABLED: Elites, 5 in total from most fit species with more than 5 genotypes.
        num_elites = np.zeros(len(self.species), dtype=int)
        # for i, spc in sorted(enumerate(self.species), key=lambda x: x[1].fitness_sum['adjusted'])[::-1]:
        #     if spc.num_genotypes > 5 and sum(num_elites) < 5:
        #         num_elites[i] += 1
        #         fittest = [gt for _, gt in sorted(zip(fitness_vector, self.population),
        #                     key=lambda x: x[0])[::-1] if gt['species'] == spc.id][0]
        #         offspring.append(fittest)

        #* Compute the number of offspring for each species
        species_offsprings = []
        for spc_elites, spc in zip(num_elites, self.species):
            spc_fitness, spc_genotypes = zip(*filter(lambda x: x[1]['species'] == spc.id, zip(fitness_vector, self.population)))
            num_offspring = max(2, int(np.round((self.pop_size - sum(num_elites)) * sum(spc_fitness) / sum(fitness_vector))))
            prev_spc_size = len(spc_genotypes) - spc_elites
            num_offspring = int(0.5 * prev_spc_size + 0.5 * num_offspring)\
                    if np.abs(num_offspring - prev_spc_size) > 0 else prev_spc_size
            species_offsprings.append(num_offspring)
        while(sum(species_offsprings) != self.pop_size - sum(num_elites)):
            species_offsprings[np.random.randint(len(self.species))] += (1, -1)[sum(species_offsprings) > self.pop_size]
        
        #* Crossover in-between species individuals.
        for n_offspring, spc in zip(species_offsprings, self.species):
            #! OJO DEEPCOPY????
            spc_fitness, spc_genotypes = zip(*filter(lambda x: x[1]['species'] == spc.id, zip(fitness_vector, self.population)))
            if len(spc_genotypes) == 1: # If only one genotype in species, no crossover.
                offspring.append(spc_genotypes[0])
                continue
            #* Truncate bests
            n_sel = max(2, int(0.4 * len(spc_genotypes))) #! Truncate only 40% best. Note that implem is diff from GA!
            parents, fitness_parents = truncation_selection(spc_genotypes, np.array(spc_fitness), n_sel)
            #* Random Mating (OJO REPLACEMENT)
            parents_mating = np.random.choice(n_sel, size=n_offspring)
            try:
                parents = [parents[idx] for idx in parents_mating] # shuffle parents
                fitness_parents = [fitness_parents[idx] for idx in parents_mating]
            except:
                import pdb; pdb.set_trace()
            #* NEAT Crossover
            offspring.extend(neat_crossover(parents, fitness_parents))
        #* Mutation
        offspring, self.current_innovation, self.innovation_history = neat_mutation(
                        offspring, self.input_nodes, self.current_innovation,
                        self.innovation_history, self.objects, p_weight_mut=self.p_weight_mut,
                        p_node_mut=self.p_node_mut, p_conn_mut=self.p_conn_mut)

        #* Assign Species. Use representatives from the previous generation.
        #* If a new species is created the current representative is the genotype 
        #* that created it.
        for species in self.species:
            species.num_genotypes = 0
        for genotype in offspring:
            compatible, distances = zip(*[species.compatibility(genotype) for species in self.species])
            if not any(compatible): #* create new species
                self.species_count += 1
                self.species.append(Species(self.species_count, generation, compatib_thresh=self.compatib_thresh,
                                    c1=self.c1, c2=self.c2, c3=self.c3))
                self.species[-1].num_genotypes += 1
                self.species[-1].representative = copy.deepcopy(genotype)
                genotype['species'] = self.species[-1].id
            else:
                # species_idx = np.random.choice(np.arange(len(self.species))[list(compatible)]) # Random
                compatible_species = np.arange(len(self.species))[list(compatible)]
                compatible_distances = np.array(distances)[list(compatible)]
                species_idx, _  = sorted(zip(compatible_species, compatible_distances), key=lambda x : x[1])[0]
                self.species[species_idx].num_genotypes += 1
                genotype['species'] = self.species[species_idx].id

        #* Check extintion
        for i, species in enumerate(self.species):
            if species.num_genotypes == 0:
                logging.info('Extint Species {}'.format(species.id))
                self.species.pop(i)
            else:
                species.representative = copy.deepcopy(offspring[np.random.choice(\
                    [n for n, g in enumerate(offspring) if g['species'] == species.id])])
        logging.info('Num. species is {}'.format(len(self.species)))
        #! Update species fitness statistics!!!
        #* Update popultation
        self.population = offspring
        fitness_vector = raw_fitness #!
        if len(self.population) != self.pop_size:
            logging.error('Population Size altered.')
            import pdb; pdb.set_trace()
        
    @property
    def min_vector(self):
        raise NotImplementedError

    @property
    def max_vector(self):
        raise NotImplementedError

    def initialize(self, interface):
        """ Initializes the parameters and population of SNES.
        =====================================================================
        - Args:
            interface [GeneticInterface] : Phenotype to genotype interface of 
                Evolutionary algs.
        - Returns: None
        =====================================================================
        """
        self.species = [Species(self.species_count, 0, compatib_thresh=self.compatib_thresh, 
                            c1=self.c1, c2=self.c2, c3=self.c3)]
        self.input_nodes = [*interface.neural_net.graph['inputs'].keys()]
        #* Only initialize weights randomly, the structure is always the same.
        for n in range(self.pop_size):
            interface.initGenotype(self.objects, self.min_vals, self.max_vals)
            #* Initialize genotype (ANN architectural traits)
            self.population.append({
                'species' : self.species[0].id,
                'nodes' : copy.deepcopy(interface.neural_net.graph['neurons']),
                'connections' : copy.deepcopy(interface.neural_net.graph['synapses'])
            })
            #* Initialize genotype (ANN parameters and weights traits)
            for query, max_val, min_val in zip(self.objects, self.min_vals, self.max_vals):
                gnt_segment = interface.toGenotype([query], [min_val], [max_val])
                gene_type = {'synapses' : 'connections', 'neurons' : 'nodes'}.get(query.split(':')[0], 'connections')
                variable = {'weights' : 'weight'}.get(query.split(':')[1], query.split(':')[1])
                for gene, value in zip(self.population[-1][gene_type].values(), gnt_segment):
                    gene[variable] = value
            #* Assign innovation numbers
            for i, conn in enumerate(self.population[-1]['connections'].values()):
                if n == 0:
                    conn['innovation'] = self.current_innovation
                    self.innovation_history[(conn['pre'], conn['post'])] = self.current_innovation
                    self.current_innovation += 1
                else:
                    conn['innovation'] = self.innovation_history[(conn['pre'], conn['post'])]
        #* Assign species representative. There is only 1 species.
        self.species[0].representative = copy.deepcopy(self.population[np.random.randint(self.pop_size)])
        self.species[0].num_genotypes = self.pop_size
