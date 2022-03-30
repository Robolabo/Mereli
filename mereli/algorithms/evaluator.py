import logging
import numpy as np
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except:
    MPI_AVAILABLE = False
    logging.warning('MPI is not installed. Running without mpi4py.')
from mereli.register import worlds, physics_engines
from mereli.algorithms.interfaces import InterfaceFactory
from mereli.register import fitness_functions

#! seed?
class Evaluator:
    def __init__(self, num_evaluations=1, fitness_fn=None):
        self.world = None
        self.fitness_fn = fitness_fn
        self.num_evaluations = num_evaluations

    def create_world(self, world_config, ann_config=None):
        physics_engine = physics_engines[world_config.get('engine', 'pybullet')](
                        dt=world_config.get('physics_dt', 0.02), 
                        T_control=world_config.get('T_control', 0.1))
        world_cls = worlds[world_config.get('name', 'square_arena')]
        arena_params = world_config.get('arena_params', {})
        self.world = world_cls(physics_engine, **arena_params)
        self.world.build_from_dict(world_config, ann_topology=ann_config)
        if self.fitness_fn is not None:
            self.fitness_fn = fitness_functions[self.fitness_fn]

    def batch_evaluate(self, genotypes, seed, algorithm):
        return [self.evaluate(geno, seed, algorithm) for geno in genotypes]

    def evaluate(self, genotype, seed, algorithm):
        self.world.connect()
        interfaces = [InterfaceFactory().create(algorithm, bot.controller.neural_network) for bot in self.robots]
        for interface in interfaces:
            interface.fromGenotype(genotype)
        for trial in range(self.num_evaluations):
            seed += 1
            survival_time = 0
            self.world.reset(seed=seed)
            while (not self.world.is_done):
                states, actions = self.world.step()
                rewards = np.array([robot.reward for robot in self.world.robots.values()])
                survival_time += 1
            mean_survival_time += survival_time
            if self.fitness_fn is not None:
                self.fitness_fn()
        mean_survival_time /= self.num_evaluations
        self.world.disconnect()
        if self.fitness_fn is not None:
            genotype.fitness = self.fitness_fn.fitness
            genotype.novelty_variables = {'eval_time' : mean_survival_time}
        return genotype

    @property
    def robots(self):
        return [robot for robot in self.world.robots.values()]

class MPI_Evaluator(Evaluator):
    def __init__(self, *args, **kwargs):
        super(MPI_Evaluator, self).__init__(*args, **kwargs)
        self.rank = MPI.COMM_WORLD.Get_rank()


    # def evaluate(self, )