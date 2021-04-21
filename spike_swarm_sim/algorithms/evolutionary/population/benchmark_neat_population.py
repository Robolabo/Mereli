import logging
import copy
import numpy as np
from scipy.linalg import expm
import neat

from .population import Population, NEAT_Population




class BenchmarkNeatPopulation(neat.Population, NEAT_Population):
    def __init__(self, *args, **kwargs):
        BenchmarkNeatPopulation.__init__(*args, **kwargs)
        NEAT_Population.__init__(*args, **kwargs)

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
