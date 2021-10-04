import logging
import copy
import numpy as np
from functools import reduce

from .population import Population
from mereli.algorithms.evolutionary.species import Species
from ..operators.crossover import *
from ..operators.mutation import *
from ..operators.selection import *
from ..gene import ConnectionGene, GraphGenotype


#! OJO (prov) to test NEAT: extracted from https://github.com/CodeReclaimers/neat-python/blob/c2b79c88667a1798bfe33c00dd8e251ef8be41fa/neat/reproduction.py#L84
def compute_spawn(species, pop_size, min_species_size):
    """Compute the proper number of offspring per species (proportional to fitness)."""
    adjusted_fitness = [spc.mean_fitness['raw'] / spc.num_genotypes for spc in species]
    af_sum = sum(adjusted_fitness)
    previous_sizes = [spc.num_genotypes for spc in species]
    spawn_amounts = []
    for af, ps in zip(adjusted_fitness, previous_sizes):
        if af_sum > 0:
            s = max(min_species_size, af / af_sum * pop_size)
        else:
            s = min_species_size
        d = (s - ps) * 0.5
        c = int(round(d))
        spawn = ps
        if abs(c) > 0:
            spawn += c
        elif d > 0:
            spawn += 1
        elif d < 0:
            spawn -= 1
        spawn_amounts.append(spawn)

    # Normalize the spawn amounts so that the next generation is roughly
    # the population size requested by the user.
    total_spawn = sum(spawn_amounts)
    norm = pop_size / total_spawn
    spawn_amounts = [max(min_species_size, int(round(n * norm))) for n in spawn_amounts]
    
    while(sum(spawn_amounts) != pop_size):
        spawn_amounts[np.random.choice(len(species))] += (1, -1)[sum(spawn_amounts) > pop_size]
    return spawn_amounts

class NEAT_Population(Population):
    """  
    """ 
    def __init__(self, *args, p_weight_mut=0.75, p_node_mut=0.08, p_conn_mut=0.1,
                compatib_thresh=2, c1=1, c2=1, c3=2, species_elites=0, **kwargs):
        super(NEAT_Population, self).__init__(*args, **kwargs)
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
        self.current_innovation = 0
        #* Dict mapping (pre, post) tuple connections to innovation numbers.
        #* It is used for assigning same innovations to mutations already occured in 
        #* the evolution.
        self.innovation_history = {}
        
    def step(self, fitness_vector, generation):
        """
        
        :param list fitness_vector: vector collecting the achieved fitness score of every individual.
        :param int generation: current generation of the evolution process.
        """
        offspring = []
        for genotype, fitness in zip(self.population, fitness_vector):
            genotype.fitness = fitness
        self.best = sorted(copy.deepcopy(self.population), key=lambda genotype: genotype.fitness, reverse=True)[0]

        #* Update species fitness statistics
        for spc in self.species:
            spc_genotypes = [genotype for genotype in self.population if genotype.species == spc.id]
            spc_fitness = np.array([genotype.fitness for genotype in spc_genotypes])
            spc.update_stats(spc_fitness)

        #* Compute the number of offspring for each species
        species_offsprings = compute_spawn(self.species, self.pop_size, 2)
        #* Crossover in-between species individuals.
        for n_offspring, spc in zip(species_offsprings, self.species):
            #* Filter out genotypes from species.
            spc_genotypes = [genotype for genotype in self.population if genotype.species == spc.id]
            #* Apply species elitism
            if self.species_elites > 0 and n_offspring > self.species_elites:
                # Use dummy iterable range(self.species_elites) to set the loop length to the number of elites
                for _, elite_gnt in zip(range(self.species_elites), sorted(spc_genotypes, key=lambda x: x.fitness, reverse=True)):
                    n_offspring -= 1
                    offspring.append(copy.deepcopy(elite_gnt))
            #* Truncate bests
            n_sel = max(1, round(0.3 * len(spc_genotypes)))
            parents = truncation_selection(spc_genotypes, n_sel)
            #* Random Mating (OJO REPLACEMENT)
            parents_mating = np.random.choice(n_sel, size=2 * n_offspring)

            parents = [parents[idx] for idx in parents_mating] # shuffle parents
            #* NEAT Crossover
            offspring.extend(neat_crossover(parents))
        #* NEAT Mutation
        offspring, self.current_innovation, self.innovation_history = neat_mutation(
                        offspring, self.input_nodes, copy.deepcopy(self.current_innovation),
                        copy.deepcopy(self.innovation_history), self.objects, p_weight_mut=self.p_weight_mut,
                        p_node_mut=self.p_node_mut, p_conn_mut=self.p_conn_mut)
        #* Update popultation
        self.population = offspring
        if len(self.population) != self.pop_size:
            logging.error('Population Size altered.')
        #* Speciation
        self.update_species(generation)
        logging.info('Num. species is {}'.format(len(self.species)))
        #* Adaptive species thresh.
        # num_tar_species = 15
        # if len(self.species) != num_tar_species:
        #     self.compatib_thresh += 0.1 * (-1, 1)[len(self.species) > num_tar_species]
        #     self.compatib_thresh = np.clip(self.compatib_thresh, a_min=0.5, a_max=5)
        #     for sp in self.species:
        #         sp.compatib_thresh = self.compatib_thresh
        # print('NEAT ', time.time() - t0)
        # import pdb; pdb.set_trace()        
       
    
    def update_species(self, generation):
        #* Assign Species. Use representatives from the previous generation.
        #* If a new species is created the current representative is the genotype 
        #* that created it.
        for spc in self.species:
            if spc.representative is not None:
                #TODO 
                compatible, distances = zip(*[spc.compatibility(gnt) for gnt in self.population])
                spc.representative = copy.deepcopy(self.population[np.argmin(distances)])
            spc.num_genotypes = 0

        for genotype in self.population:
            compatible, distances = zip(*[spc.compatibility(genotype) for spc in self.species])
            if not any(compatible): #* create new species
                self.species_count += 1
                new_species = Species(self.species_count, generation, compatib_thresh=self.compatib_thresh,
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

        #* check extintion
        for i, species in enumerate(self.species):
            if species.num_genotypes == 0:
                logging.info('Extint Species {}'.format(species.id))
                self.species.pop(i)
            # else:
            #     species.representative = copy.deepcopy(self.population[np.random.choice(\
            #         [n for n, g in enumerate(self.population) if g['species'] == species.id])])

    @property
    def min_vector(self):
        raise NotImplementedError

    @property
    def max_vector(self):
        raise NotImplementedError

    def initialize(self, interface):
        """ Initializes the parameters and population of NEAT.

        - Args:
            interface [GeneticInterface] : Phenotype to genotype interface of 
                Evolutionary algs.
        - Returns: None
        """
        self.species = [Species(self.species_count, 0, compatib_thresh=self.compatib_thresh, 
                            c1=self.c1, c2=self.c2, c3=self.c3)]
        self.input_nodes = [*interface.neural_net.graph['inputs'].keys()]
        #* Only initialize weights randomly, the structure is always the same.
        for n in range(self.pop_size):
            interface.initGenotype(self.objects, self.min_vals, self.max_vals)

            #* Create new genotype
            genotype = GraphGenotype()
            # import pdb; pdb.set_trace()
            # genotype.evolvable_structs = [x.split(':')[] for x in self.objects]
            for name, node_vals in interface.neural_net.graph['neurons'].items():
                genotype.add_node_from_dict(name, **node_vals)
            for name, conn_vals in interface.neural_net.graph['synapses'].items():
                genotype.add_conn_from_dict(name, **conn_vals)
            
            #* Initialize genotype (ANN parameters and weights traits)
            for query, min_val, max_val in zip(self.objects, self.min_vals, self.max_vals):
                gnt_segment = interface.toGenotype([query], [min_val], [max_val])
                variable = query.split(':')[1]
                if variable == 'weights': 
                    variable = 'weight'
                gene_type = {'synapses' : 'connections', 'neurons' : 'nodes'}.get(query.split(':')[0], 'connections')
                for gene, value in zip(getattr(genotype, gene_type), gnt_segment):
                    assert hasattr(gene, variable)
                    setattr(gene, variable, value)
            #* Assign innovation numbers
            for conn in genotype.connections:
                if n == 0:
                    conn.innovation = self.current_innovation
                    self.innovation_history[(conn.pre, conn.post)] = self.current_innovation
                    self.current_innovation += 1
                else:
                    conn.innovation = self.innovation_history[(conn.pre, conn.post)]
            #* Add genotype to the population
            self.population.append(genotype)
        #* Initial Speciation
        self.update_species(0)
        # self.species[0].representative = copy.deepcopy(self.population[np.random.randint(self.pop_size)])
        # self.species[0].num_genotypes = self.pop_size
