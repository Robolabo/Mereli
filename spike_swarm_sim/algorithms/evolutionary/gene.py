import numpy as np
from collections import deque


class BaseGene:

    def __init__(self):
        pass
    

class NodeGene(BaseGene):
    def __init__(self, *args, **kwargs):
        super(NodeGene, self).__init__(*args, **kwargs)
        self.ensemble
        self.is_output
        self.idx

    def value(self):
        return {''}
        
class ConnectionGene(BaseGene):
    def __init__(self, *args, **kwargs):
        super(NodeGene, self).__init__(*args, **kwargs)
        self.pre
        self.post
        self.weight
        self.delay = None
        self.group = None
        self.learning_rule = None
        self.innovation = None
        self.idx = None

    def value(self):
        return {''}



# class BaseGenotype:
#     def __init__(self):


class FixedLenGenotype:
    def __init__(self):
        self._fitness = None
        self.genes = deque([])

    def add_gene(self, gene):
        self.genes.append(gene)

    def __len__(self):
        return len(self.genes)

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, fitness_value):
        self._fitness = fitness_value

class GraphGenotype:
    def __init__(self):
        self._fitness = None
        self.node_genes = deque([])
        self.connection_genes = deque([])

    def add(self, gene):
        pass
