
import logging
import copy
import numpy as np
from functools import reduce

from . import NEAT_Population
from mereli.algorithms.evolutionary.species import Species
from ..operators.crossover import *
from ..operators.mutation import *
from ..operators.selection import *
from ..gene import ConnectionGene, GraphGenotype




class CPPN_NEAT_Population(NEAT_Population):
    """  
    """ 
    def __init__(self, *args, **kwargs):
        super(CPPN_NEAT_Population, self).__init__(*args, **kwargs)
