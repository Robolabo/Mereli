import logging
import numpy as np
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except:
    MPI_AVAILABLE = False
    logging.warning('MPI is not installed. Running without mpi4py.')
from mereli.register import worlds, physics_engines

#! seed?
class Evaluator:
    def __init__(self, num_evaluations=1):
        self.world = None
        self.num_evaluations = num_evaluations

    def create_world(self, world_config, ann_config=None):
        physics_engine = physics_engines[world_config.get('engine', 'pybullet')](
                        dt=world_config.get('physics_dt', 0.02), 
                        T_control=world_config.get('T_control', 0.1))
        world_cls = worlds[world_config.get('name', 'square_arena')]
        arena_params = world_config.get('arena_params', {})
        self.world = world_cls(physics_engine, **arena_params)
        self.world.build_from_dict(world_config, ann_topology=ann_config)

    def evaluate(self, genotype, seed, algorithm):
        self.world.connect()
        robots = [robot for robot in self.world.robots.values()]
        interfaces = [InterfaceFactory().create(algorithm, bot.controller.neural_network) for bot in robots]
        for interface in interfaces:
            genotype_segment = pop.population[env_id]
            interface.fromGenotype(pop.objects, genotype_segment, pop.min_vals, pop.max_vals)
        for trial in range(self.num_evaluations):
            survival_time = 0
            self.world.reset(seed=seed)
            while (not self.world.is_done):
                states, actions = self.world.step()
                rewards = np.array([robot.reward for robot in self.world.robots.values()])
                survival_time += 1
            mean_survival_time += survival_time
        mean_survival_time /= self.num_evaluations
        self.world.disconnect()

    @property
    def robots(self):

class MPI_Evaluator(Evaluator):
    def __init__(self, *args, **kwargs):
        super(MPI_Evaluator, self).__init__(*args, **kwargs)
        self.rank = MPI.COMM_WORLD.Get_rank()


    def 