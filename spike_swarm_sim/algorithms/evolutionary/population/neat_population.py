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
    def __init__(self, *args, **kwargs):
        super(NEAT_Population, self).__init__(*args, **kwargs)
        self.node_mut_prob = None
        self.conn_mut_prob = None
        self.weight_mut_prob = None
        self.species_count = 1
        self.species = [Species(self.species_count)] # list of existing species. 1 species at first.
        self.input_nodes = [] #* Cannot be altered by NEAT 
        self.population = []
        #* Global pointer of gene innovations
        self.current_innovation = 0
        #* Dict mapping (pre, post) tuple connections to innovation numbers.
        #* It is used for assigning same innovations to mutations already occured in 
        #* the evolution.
        self.innovation_history = {}
      
    def step(self, fitness_vector):
        """ 
        ==================================================================================
        - Args:
            fitness_vector [np.ndarray or list]: array of computed fitness values.
        - Returns: None
        ==================================================================================
        """
        #TODO self.segment_lengths = [interface.submit_query(query, primitive='LEN') for query in self.objects]
        #TODO adjusted_fitness = 
        offspring = []
        #* Crossover in-between species individuals.
        for spc in self.species:
            #! OJO DEEPCOPY????
            spc_fitness, spc_genotypes = zip(*filter(lambda x: x[1]['species'] == spc.id, zip(fitness_vector, self.population)))
            #* Truncate bests
            n_sel = max(2, int(0.4 * len(spc_genotypes))) #! Truncate only 40% best. Note that implem is diff from GA!
            parents, fitness_parents = truncation_selection(spc_genotypes, np.array(spc_fitness), n_sel)

            #* Decide the number of species offspring
            num_offspring = max(2, int(self.pop_size * sum(spc_fitness) / sum(fitness_vector)))
            num_offspring = int(0.6 * len(spc_genotypes) + 0.4 * num_offspring)\
                    if np.abs(num_offspring - len(spc_genotypes)) > 0 else len(spc_genotypes)
            #* Random Mating (OJO REPLACEMENT)
            parents_mating = np.random.choice(n_sel, size=num_offspring)
            parents = [parents[idx] for idx in parents_mating] # shuffle parents
            fitness_parents = [fitness_parents[idx] for idx in parents_mating]
            #* NEAT Crossover
            offspring.extend(neat_crossover(parents, fitness_parents))
        
        #* Mutation
        offspring, self.current_innovation, self.innovation_history = neat_mutation(
                        offspring, self.input_nodes, self.current_innovation,
                        self.innovation_history, p_weight_mut=0.75, p_node_mut=0.08, p_conn_mut=0.1)

        #* Assign Species. Use representatives from the previous generation.
        #* If a new species is created the current representative is the genotype 
        #* that created it.
        for species in self.species:
            species.num_genotypes = 0
        for genotype in offspring:
            compatible, distances = zip(*[species.compatibility(genotype) for species in self.species])
            if not any(compatible): #* create new species
                self.species_count += 1
                self.species.append(Species(self.species_count))
                self.species[-1].num_genotypes += 1
                self.species[-1].representative = copy.deepcopy(genotype)
                genotype['species'] = self.species[-1].id
            else:
                species_idx = np.random.choice(np.arange(len(self.species))[list(compatible)])
                self.species[species_idx].num_genotypes += 1
                genotype['species'] = self.species[species_idx].id

        #! check extintion
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
        for pp in self.population:
            for x, xx in pp['connections'].items():
                if xx['pre'] not in self.input_nodes + list(pp['nodes']):
                    import pdb; pdb.set_trace()
                if xx['post'] not in list(pp['nodes']):
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
        self.input_nodes = [*interface.neural_net.graph['inputs'].keys()]
        #* Only initialize weights randomly, the structure is always the same.
        for n in range(self.pop_size):
            interface.initGenotype(self.objects, self.min_vals, self.max_vals)
            self.population.append({
                'species' : self.species[0].id,
                'nodes' : copy.deepcopy(interface.neural_net.graph['neurons']),
                'connections' : copy.deepcopy(interface.neural_net.graph['synapses'])
            })
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
        #* Segment lengths have to be updated in every generation.
        self.segment_lengths = [interface.submit_query(query, primitive='LEN') for query in self.objects]

        #* --- NEAT Hyperparams --- #
        self.p_weight_mut = 0.8 #! NO del todo, mirar paper
        self.p_node_mut = 0.03
        self.p_conn_mut = 0.05
