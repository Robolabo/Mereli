import copy
import time
import logging
import numpy as np

# from .population import NEAT_Population  
from .evolutionary_algorithm import EvolutionaryAlgorithm
from mereli.algorithms.interfaces import NEATInterface
from mereli.register import algorithm_registry
from mereli.globals import global_states
from mereli.utils import save_pickle, load_pickle
from .species import Species
from mereli.algorithms.evolutionary.species import Species
from .operators.crossover import *
from .operators.mutation import *
from .operators.selection import *
from .gene import GraphGenotype

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
        # populations = {name : NEAT_Population(kwargs['population_size'],\
        #         pop['min_vals'], pop['max_vals'], pop['objects'], **pop['params'])\
        #         for name, pop in populations.items()}
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
        # for genotype, fitness in zip(self.population, fitness_vector):
        #     genotype.fitness = fitness
        self.best = sorted(copy.deepcopy(self.population), key=lambda genotype: genotype.fitness, reverse=True)[0]
        #* Update species fitness statistics
        for spc in self.species:
            spc_genotypes = [genotype for genotype in self.population if genotype.species == spc.id]
            spc_fitness = np.array([genotype.fitness for genotype in spc_genotypes])
            try:
                spc.update_stats(spc_fitness)
            except:
                print(spc_fitness, spc_genotypes, spc.num_genotypes, spc.id, self.species)
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
                p_weight_mut=self.p_weight_mut, p_node_mut=self.p_node_mut, p_conn_mut=self.p_conn_mut)
        #* Update popultation
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
            if species.num_genotypes == 0:
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

    def initialize(self, interface):
        """ Initializes the parameters and population of NEAT.

        - Args:
            interface [GeneticInterface] : Phenotype to genotype interface of 
                Evolutionary algs.
        - Returns: None
        """
        self.species = []#Species(self.species_count, 0, compatib_thresh=self.compatib_thresh, 
                            #c1=self.c1, c2=self.c2, c3=self.c3)]
        self.input_nodes = [*interface.neural_net.graph['inputs'].keys()]
        #* Only initialize weights randomly, the structure is always the same.
        for n in range(self.pop_size):
            interface.initGenotype(self.objects, self.min_vals, self.max_vals)
            #* Create new genotype
            genotype = GraphGenotype()
            genotype.evolvable_structs = {obj : {'min' : min_v, 'max' : max_v}\
                    for obj, min_v, max_v in zip(self.objects, self.min_vals, self.max_vals)}
            # genotype.evolvable_structs = [x.split(':')[] for x in self.objects]
            for name, node_vals in interface.neural_net.graph['neurons'].items():
                genotype.add_node_from_dict(name, **node_vals)
            for name, conn_vals in interface.neural_net.graph['synapses'].items():
                genotype.add_conn_from_dict(name, **conn_vals)

            #* Initialize genotype (ANN parameters and weights traits)
            for query, min_val, max_val in zip(self.objects, self.min_vals, self.max_vals):
                gnt_segment = interface.toGenotype([query], [min_val], [max_val])
                variable = query.split(':')[1]
                variable = variable.replace('weights', 'weight')
                gene_type = {'synapses' : 'connections', 'neurons' : 'nodes'}.get(query.split(':')[0], 'connections')
                for gene, value in zip(getattr(genotype, gene_type), gnt_segment):
                    gene.add_parameter(variable, value)
            #* Assign innovation numbers
            for conn in genotype.connections:
                conn.innovation = self.innovation.assign(conn.pre, conn.post)
            #* Add genotype to the population
            self.population.append(genotype)
        #* Initial Speciation
        self.update_species(0)
        # self.species[0].representative = copy.deepcopy(self.population[np.random.randint(self.pop_size)])
        # self.species[0].num_genotypes = self.pop_size

    def save_population(self, generation):
        """ Saves the checkpoint with the necessary information to resume the evolution. 
        """
        pop_checkpoint = {
            'populations' : {name : {
                'best' : pop.best,
                'genotypes' : pop.population, # List of dicts
                'innovation' : pop.innovation,
                'input_nodes' : pop.input_nodes,
                'species_count' : pop.species_count,
                'species' : [{
                    'id' : spc.id,
                    'creation_generation' : spc.creation_generation,
                    'history' : copy.deepcopy(spc.history),
                    'representative' : spc.representative,
                    'thresh' : spc.compatib_thresh,
                    'c1' : spc.c1, 'c2' : spc.c2, 'c3' : spc.c3}
                for spc in pop.species],
                'species_hist' : None
            } for name, pop in self.populations.items()},
            'generation' : generation,
            'p_weight_mut' : {name : pop.p_weight_mut for name, pop in self.populations.items()},
            'p_node_mut' : {name : pop.p_node_mut for name, pop in self.populations.items()},
            'p_conn_mut' : {name : pop.p_conn_mut for name, pop in self.populations.items()},
            'evolution_hist' : self.evolution_history,
        }
        file_name = 'mereli/checkpoints/populations/' + self.checkpoint_name
        save_pickle(pop_checkpoint, file_name)
        logging.info('Successfully saved evolution checkpoint.')
                
    def load_population(self):
        """ Loads a previously saved checkpoint to resume evolution.
        """
        checkpoint = load_pickle('mereli/checkpoints/populations/' + self.checkpoint_name)
        logging.info('Resuming NEAT evolution using checkpoint ' +  self.checkpoint_name)
        key = tuple(self.populations.keys())[0]
        for key, pop in checkpoint['populations'].items():
            self.populations[key].p_weight_mut = checkpoint['p_weight_mut'][key]
            self.populations[key].p_node_mut = checkpoint['p_node_mut'][key]
            self.populations[key].p_conn_mut = checkpoint['p_conn_mut'][key]

            self.populations[key].population = pop['genotypes']
            
            self.populations[key].best = pop.get('best', None)
            self.populations[key].innovation = pop['innovation']
            self.populations[key].input_nodes = pop['input_nodes']
            self.populations[key].species_count = pop['species_count']
            self.populations[key].species = []
            for spc_chk in pop['species']:
                spc = Species(spc_chk['id'], spc_chk['creation_generation'],
                        compatib_thresh=spc_chk['thresh'], c1=spc_chk['c1'], c2=spc_chk['c2'], c3=spc_chk['c3'])
                spc.history = copy.deepcopy(spc_chk['history'])
                self.populations[key].species.append(spc)
                spc.representative = spc_chk['representative']
                spc.num_genotypes = np.sum([genotype.species == spc.id  for genotype in pop['genotypes']])
            robots = [copy.deepcopy(robot) for robot in self.world.robots.values()]
            interface = NEATInterface(robots[0].controller.neural_network)
            # #!
            # self.populations[key].segment_lengths = [interface.submit_query(query, primitive='LEN')\
            #             for query in self.populations[key].objects]         
        self.generation = checkpoint['generation']
        self.evolution_history = checkpoint['evolution_hist']
