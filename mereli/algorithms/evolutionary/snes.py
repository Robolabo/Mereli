import copy
import logging
import numpy as np
from .population import SNES_Population  
from .evolutionary_algorithm import EvolutionaryAlgorithm
from mereli.algorithms.interfaces import GeneticInterface
from mereli.register import algorithm_registry
from mereli.globals import global_states
from mereli.utils import save_pickle, load_pickle

@algorithm_registry(name='SNES')
class SNES(EvolutionaryAlgorithm):
    """ Class of the Separable Natural Evolution Strategy (SNES).  
    The evolution step is defined in the SNES_Population class.
    """
    def __init__(self, *args, **kwargs):
        super(SNES, self).__init__(*args, **kwargs)
        self.eta_mu = None
        self.eta_s = None 
        self.mu = None
        self.sigma = None
        self.z_samples = None
    

    def evolve(self):
        pass
    
    @property
    def checkpoint_data(self):
        return {**super().checkpoint_data, 
            **{'mu' : self.species, 'sigma' : self.sigma, 'eta_mu' : self.eta_mu, 'eta_s' : self.eta_s}}     
    
    # def load(self):
    #     super().load()
    #     if global_states.EVAL:
    #         self.sigma = 1e-3 * np.ones_like(self.sigma)
