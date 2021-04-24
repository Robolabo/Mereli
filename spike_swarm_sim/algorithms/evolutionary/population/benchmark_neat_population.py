import os
import logging
import copy
import numpy as np
from itertools import count
import neat

from . import Population, NEAT_Population

def parse_config(neat_config, pop_size, **kwargs):
    neat_config.pop_size = pop_size
    neat_config.genome_config.conn_add_prob = kwargs['p_conn_mut']
    # neat_config.genome_config.conn_delete_prob = 
    neat_config.genome_config.node_delete_prob = kwargs['p_node_mut']
    # neat_config.genome_config.node_delete_prob = 
    neat_config.genome_config.weight_mutate_rate = kwargs['p_weight_mut']
    neat_config.genome_config.compatibility_disjoint_coefficient = kwargs['c1']
    neat_config.genome_config.compatibility_weight_coefficient = kwargs['c3']
    neat_config.species_set_config.compatibility_threshold = kwargs['compatib_thresh']

    return neat_config


def adapt_genotype(gA, gB):
    import pdb; pdb.set_trace()


class BenchmarkNeatPopulation(neat.Population, Population):
    def __init__(self, *args, p_weight_mut=0.75, p_node_mut=0.08, p_conn_mut=0.1,
                    compatib_thresh=2, c1=1, c2=1, c3=2, **kwargs):
        
        Population.__init__(self, *args, **kwargs)
        config_path = 'spike_swarm_sim/config/others/benchmarkNEAT_ctrnn_config'
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
        # import pdb; pdb.set_trace()
        self.config = parse_config(config, self.pop_size, p_weight_mut=p_weight_mut, p_node_mut=p_node_mut, 
                    p_conn_mut=p_conn_mut, compatib_thresh=compatib_thresh, c1=c1, c2=c2, c3=c3)
        # neat.Population.__init__(self, config)

    def step(self, fitness_vector, generation):
        #! self.reporters.start_generation(self.generation)

        # Evaluate all genomes using the user-provided function.
        #! fitness_function(list(self.population.items()), self.config)

        # Gather and report statistics.
        best = None
        for g in self.population.values():
            if g.fitness is None:
                raise RuntimeError("Fitness not assigned to genome {}".format(g.key))

            if best is None or g.fitness > best.fitness:
                best = g
        self.reporters.post_evaluate(self.config, self.population, self.species, best)

        # Track the best genome ever seen.
        if self.best_genome is None or best.fitness > self.best_genome.fitness:
            self.best_genome = best


        # Create the next generation from the current generation.
        self.population = self.reproduction.reproduce(self.config, self.species,
                                                        self.config.pop_size, self.generation)

        # Check for complete extinction.
        if not self.species.species:
            self.reporters.complete_extinction()

            # If requested by the user, create a completely new population,
            # otherwise raise an exception.
            if self.config.reset_on_extinction:
                self.population = self.reproduction.create_new(self.config.genome_type,
                                                                self.config.genome_config,
                                                                self.config.pop_size)
            else:
                raise neat.CompleteExtinctionException()

        # Divide the new population into species.
        self.species.speciate(self.config, self.population, self.generation)

        #! Report stuff
        #! self.reporters.end_generation(self.config, self.population, self.species)



    def initialize(self, interface):
        """ Initializes the parameters and population of SNES.
        =====================================================================
        - Args:
            interface [GeneticInterface] : Phenotype to genotype interface of 
                Evolutionary algs.
        - Returns: None
        =====================================================================
        """
        
        self.config.genome_config.num_inputs = interface.neural_net.num_inputs
        self.config.genome_config.num_outputs = interface.neural_net.num_motor
        self.config.genome_config.num_hidden = 0
        self.config.genome_config.input_keys = [-i - 1 for i in range(self.config.genome_config.num_inputs)]
        self.config.genome_config.output_keys = [i for i in range(self.config.genome_config.num_outputs)]
        self.config.genome_config.node_indexer = count(interface.neural_net.num_motor)
        neat.Population.__init__(self, self.config)
        # self.population = self.reproduction.create_new(self.config.genome_type,
        #                                                    self.config.genome_config,
        #                                                    self.config.pop_size)
        # self.species = self.config.species_set_type(self.config.species_set_config, self.reporters)
        # self.generation = 0
        # self.species.speciate(self.config, self.population, self.generation)


        for genotype in self.population.values():
            interface.initGenotype(self.objects, self.min_vals, self.max_vals)
            import pdb; pdb.set_trace()