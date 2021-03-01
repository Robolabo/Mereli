import logging
import numpy as np

class Species:
    def __init__(self, id, compatib_thresh=3, c1=1, c2=1, c3=2.):
        self.id = id
        self.compatib_thresh = compatib_thresh
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.num_genotypes = 0
        self.representative = {}
        self.mean_fitness = None
        self.max_fitness = None
        self.last_improvement = 0

    def compatibility(self, genotype):
        """ Computes the compatibility distance of the genotype to the species 
        as defined in the NEAT paper. It returns both the distance and whether the 
        genotype is compatible to the species or not.
        ============================================================================
        - Args:
            genotype [dict] :
        - Returns:
            is_compatible [bool] : whether the genotype is compatible or not.
            distance [float] : compatibility distance of the genotype to the species 
                    representative.
        ============================================================================
        """
        if self.representative is None or len(self.representative) == 0:
            logging.error(Exception('Cannot compute compatibility of genotype '\
                'to species if there is no species representative.'))
        repr_innovations = set([g['innovation'] for g in self.representative['connections'].values()])
        genotype_innovations = set([g['innovation'] for g in genotype['connections'].values()])
        # Do not care about disjoint and excess. For the moment we use same 
        # weights.
        diff_genes = genotype_innovations - repr_innovations
        weights_repr = np.array([g['weight'] for g in self.representative['connections'].values()])
        weights_genotype = np.array([g['weight'] for g in genotype['connections'].values()])
        W_dist = np.abs(weights_repr.mean() - weights_genotype.mean()) #!CHECK
        dist = 2 * self.c1 * (len(diff_genes) / max(len(weights_repr), len(weights_genotype))) \
                + self.c3 * W_dist
        return dist < self.compatib_thresh, dist

    def update(self):
        pass

    #! OJO
    @property
    def is_extinct(self):
        return self.last_improvement >= 15
