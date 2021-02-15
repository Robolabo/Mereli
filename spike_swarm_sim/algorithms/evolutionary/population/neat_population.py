import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns
from scipy.linalg import expm
from .population import Population
from spike_swarm_sim.utils import eigendecomposition, normalize

class NEAT_Population(Population):
    """  
    """
    def __init__(self, *args, **kwargs):
        super(NEAT_Population, self).__init__(*args, **kwargs)
        self.node_mut_prob = None
        self.conn_mut_prob = None
        self.weight_mut_prob = None
        self.species = None
        self.population = []
      

    def step(self, fitness_vector):
        """ SNES Evolution step applied at the end of each generation to update the population.
        ==================================================================================
        - Args:
            fitness_vector [np.ndarray or list]: array of computed fitness values.
        - Returns: None
        ==================================================================================
        """
        fitness_order = np.argsort(fitness_vector.copy())[::-1]
        ord_samples = [self.z_samples[idx].copy() for idx in fitness_order]
        ord_fitness = np.array([fitness_vector[idx] for idx in fitness_order])



    def initialize(self, interface):
        """ Initializes the parameters and population of SNES.
        =====================================================================
        - Args:
            interface [GeneticInterface] : Phenotype to genotype interface of 
                Evolutionary algs.
        - Returns: None
        =====================================================================
        """
        #! OJO: esto puede no ser útil.
        self.segment_lengths = [interface.submit_query(query, primitive='LEN') for query in self.objects]
        genotype_length = interface.toGenotype(self.objects, self.min_vector, self.max_vector).shape[0]
        np.random.seed()

        for _ in range(self.pop_size):
            interface.initGenotype(self.objects, self.min_vals, self.max_vals)
            self.population.append(interface.toGenotype(self.objects, self.min_vals, self.max_vals))




        #* Use larger sigma at first for better initialization
        self.mu = 0.5 * np.ones(genotype_length)
        self.mu = np.clip(self.mu, a_min=0., a_max=1.)
        self.sigma = np.ones(genotype_length) #0.2 * np.ones(genotype_length)
        d = self.mu.shape[0]
        n_expected = int(4 + np.floor(3 * np.log(d)))
        self.eta_mu = 1.
        self.eta_s = (3 + np.log(d)) / (5 * np.sqrt(d)) + 0.2 #!
        
        #* sample initial pop
        self.population, self.z_samples = self.sample()
        self.population = [np.clip(v, a_min=0., a_max=1.) for v in self.population]
        self.sigma = 0.5 * np.ones(genotype_length) # 0.1 * np.ones(genotype_length)
