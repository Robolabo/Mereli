
import numpy as np

class Species:
    def __init__(self):
        self.id = None
        self.centroid = None
        self.mean_fitness = None
        self.max_fitness = None
        self.last_improvement = 0

    def update(self):
        pass

    @property
    def is_extinct(self):
        return self.last_improvement >= 15