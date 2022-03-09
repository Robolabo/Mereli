import logging
from itertools import chain
import numpy as np

class Species:
    def __init__(self, id, generation, compatib_thresh=3, c1=1, c2=1, c3=2.):
        self.id = id
        self.compatib_thresh = compatib_thresh
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.num_genotypes = 0
        self.representative = None
        self.mean_fitness = {'raw' : 0, 'adjusted' : 0}
        self.max_fitness = {'raw' : 0, 'adjusted' : 0}
        self.min_fitness = {'raw' : 0, 'adjusted' : 0}
        self.fitness_sum = {'raw' : 0, 'adjusted' : 0}
        self.creation_generation = generation
        self.history = {key : [] for key in ['num_genotypes', 'mean_fitness',
                                        'max_fitness', 'min_fitness', 'sum_fitness']}
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
        if self.representative is None:
            return (False, 1000.)

        repr_innovations = set(g.innovation for g in self.representative.connections)
        genotype_innovations = set(g.innovation for g in genotype.connections)
        repr_nodes = set(g.name for g in self.representative.nodes)
        geno_nodes = set(g.name for g in genotype.nodes)

        # Do not care about disjoint and excess. For the moment we use same weights.
        diff_genes = genotype_innovations - repr_innovations
        common_genes = genotype_innovations.intersection(repr_innovations)
        # import pdb; pdb.set_trace()
        param_distance = 0
        conn_params = [*genotype.connections][0].parameters
        node_params = [*genotype.nodes][0].parameters
        #* Connection parameter's distance 
        if len(conn_params):
            for param in conn_params:
                param_repr = np.array([g.parameters[param] for g in self.representative.connections
                                        if g.innovation in common_genes])
                param_genotype = np.array([g.parameters[param] for g in genotype.connections 
                                        if g.innovation in common_genes])
                assert len(param_repr) == len(param_genotype)
                param_distance += np.linalg.norm(param_repr - param_genotype) / np.sqrt(len(param_genotype))
            param_distance /= len(conn_params)
        #* Node parameter's distance
        if len(node_params):
            for param in node_params:
                param_repr = np.array([self.representative.get_node(node).parameters[param] 
                            for node in geno_nodes.intersection(repr_nodes)])
                param_genotype = np.array([genotype.get_node(node).parameters[param]
                            for node in geno_nodes.intersection(repr_nodes)])
                assert len(param_repr) == len(param_genotype)
                param_distance += np.linalg.norm(param_repr - param_genotype) / np.sqrt(len(param_genotype))
            param_distance /= len(node_params) 
        
        #* Topological distance
        arch_conn_distance = len(diff_genes) / 10 #max(len(repr_innovations), len(genotype_innovations))
        arch_node_distance = len(geno_nodes - repr_nodes) / max(len(geno_nodes), len(repr_nodes))
        total_dist = self.c1 * (arch_conn_distance + arch_node_distance) + self.c3 * param_distance
        return total_dist < self.compatib_thresh, total_dist

    def update_stats(self, fitness_scores):
        self.history['num_genotypes'].append(self.num_genotypes)
        self.history['mean_fitness'].append(np.mean(fitness_scores))
        self.history['max_fitness'].append(max(fitness_scores))
        self.history['min_fitness'].append(min(fitness_scores))
        self.history['sum_fitness'].append(sum(fitness_scores))

        adj_fitness_scores = fitness_scores.copy() / self.num_genotypes
        self.mean_fitness.update({
            'raw' : np.mean(fitness_scores),
            'adjusted': np.mean(adj_fitness_scores)
        })
        self.max_fitness.update({
            'raw' : max(fitness_scores),
            'adjusted': max(adj_fitness_scores)
        })
        self.min_fitness.update({
            'raw' : min(fitness_scores),
            'adjusted': min(adj_fitness_scores)
        })
        self.fitness_sum.update({
            'raw' : sum(fitness_scores),
            'adjusted': sum(adj_fitness_scores)
        })

    @property
    def adjusted_fitness(self):
        return self.mean_fitness['raw'] / self.num_genotypes

    @property
    def is_extinct(self):
        return self.last_improvement >= 15
        
