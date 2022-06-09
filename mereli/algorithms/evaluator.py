import logging
import copy
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

class Evaluator:
    def __init__(self, world, num_evaluations=1, fitness_fn=None, use_seed=True):
        self._world = copy.deepcopy(world)
        self.fitness_fn = fitness_fn
        if self.fitness_fn is not None:
            self.fitness_fn = fitness_functions[self.fitness_fn](self._world)
        self.num_evaluations = num_evaluations
        self.use_seed = use_seed


    def create_world(self, world_config, ann_config=None):
        physics_engine = physics_engines[world_config.get('engine', 'pybullet')](
                        dt=world_config.get('physics_dt', 0.02), 
                        T_control=world_config.get('T_control', 0.1))
        world_cls = worlds[world_config.get('name', 'square_arena')]
        arena_params = world_config.get('arena_params', {})
        self.world = world_cls(physics_engine, **arena_params)
        self.world.build_from_dict(world_config, ann_topology=ann_config)
        if self.fitness_fn is not None:
            self.fitness_fn = fitness_functions[self.fitness_fn](self.world)

    def batch_evaluate(self, genotypes, generation):
        return [self.evaluate(geno, generation) for geno in genotypes]

    def build_phenotype(self, genotype):
        #* Build phenotype
        for ent_name, entity in self.world.hierarchy.items():
            phenotype = genotype.as_phenotype()
            for target, pheno in phenotype.items():
                target_route, target_asset = target.split('@')
                target_route = target_route.split(':')
                if entity.group == target_route[0]:
                    res = entity
                    for tar_point in target_route[1:]:
                        res = getattr(res, tar_point) if not isinstance(res, dict) else res[tar_point]
                    setattr(res, target_asset, pheno)
            # entity.controller.neural_network = genotype.as_phenotype()

    def evaluate(self, genotype, generation):
        assert self.world is not None
        self.world.connect()
        if isinstance(genotype, list):
            for geno_i in genotype:
                self.build_phenotype(geno_i)
        else:
            self.build_phenotype(genotype)
        # Genotype is evaluated N_E independent trials  
        mean_survival_time = 0
        seed = generation * self.num_evaluations
        fitness = 0
        for trial in range(self.num_evaluations):
            seed += 1
            survival_time = 0
            
            # Reset the world for a new simulation/episode
            self.world.reset(seed=seed if self.use_seed else None)
            if self.fitness_fn is not None:
                self.fitness_fn.reset()
            while (not self.world.is_done):
                states, actions = self.world.step()
                # rewards = np.array([robot.reward for robot in self.world.robots.values()])
                if self.fitness_fn is not None:
                    self.fitness_fn()
                survival_time += 1
            mean_survival_time += survival_time
            if self.fitness_fn is not None:
                fitness += self.fitness_fn.fitness
        mean_survival_time /= self.num_evaluations
        #TODO Mejorar.
        if isinstance(genotype, list):
            for geno_i in genotype:
                geno_i.novelty_variables = {
                    'eval_time' : mean_survival_time, 
                    'fitness' : fitness / self.num_evaluations,
                    'positions' : np.hstack([robot.position[:2] for robot in self.world.robots.values()])
                }
                geno_i.fitness = fitness / self.num_evaluations
        else:
            genotype.novelty_variables = {
                    'eval_time' : mean_survival_time, 
                    'fitness' : fitness / self.num_evaluations,
                    'positions' : np.hstack([robot.position[:2] for robot in self.world.robots.values()])
                }
            genotype.fitness = fitness / self.num_evaluations
        self.world.disconnect()
        return genotype

    @property
    def robots(self):
        return [robot for robot in self.world.robots.values()]

    @property
    def world(self):
        return self._world

    @world.setter
    def world(self, new_world):
        self._world = new_world


class MPI_Evaluator(Evaluator):
    def __init__(self, *args, **kwargs):
        super(MPI_Evaluator, self).__init__(*args, **kwargs)
        
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.size = MPI.COMM_WORLD.Get_size()
        # self._worlds = [copy.deepcopy(self._world) for _ in range(self.size)]
    
    def batch_evaluate(self, genotypes, generation):
        genotypes = MPI.COMM_WORLD.bcast(genotypes, root=0)
        MPI.COMM_WORLD.Barrier()
        pop_size = len(genotypes)
        num_genotypes = pop_size // self.size + (self.rank == 0) * (pop_size % self.size)
        rnk_genotype_ids = np.arange(num_genotypes * self.rank, num_genotypes * (self.rank + 1))
        rnk_genotypes = [genotypes[g_id] for g_id in rnk_genotype_ids]

        rnk_genotypes = [self.evaluate(geno, generation) for geno in rnk_genotypes]
        MPI.COMM_WORLD.barrier()
        all_genotypes = MPI.COMM_WORLD.gather(rnk_genotypes, root=0) #! OJO COMPROBAR
        if self.rank == 0:
            all_genotypes = [geno for geno_batch in all_genotypes for geno in geno_batch]
        return all_genotypes

    # @property
    # def world(self):
    #     return self._worlds[self.rank]
        