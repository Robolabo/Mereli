import copy
import random 
import numpy as np
from spike_swarm_sim.register import evo_operator_registry

def neat_crossover(parents, fitness_values, crossover_prob=1., disable_prob=0.75):
    #! First version, to be optimized
    offspring = []
    if len(parents) % 2:
        offspring.append(parents.pop(0))
    for f1, f2, parent1, parent2 in zip(fitness_values[::2], fitness_values[1::2], parents[::2], parents[1::2]):
        # child_1 = {'species' : None, 'nodes': copy.deepcopy((parent1, parent2)[f2 >= f1]['nodes']), 'connections' : {}}
        child_1 = {'species' : None, 'nodes': {}, 'connections' : {}}
        child_2 = copy.deepcopy(child_1)
        innov_ids_1 = set([gene['innovation'] for gene in parent1['connections'].values()])
        innov_ids_2 = set([gene['innovation'] for gene in parent2['connections'].values()])
        common_genes = innov_ids_1.intersection(innov_ids_2)
        random_mask = np.random.randint(2, size=len(common_genes))
        #* Common connection genes
        for rnd_val, gene_innovation in zip(random_mask, common_genes):
            parent1_gene = {name : conn for name, conn in parent1['connections'].items()\
                            if conn['innovation'] == gene_innovation}
            parent2_gene = {name : conn for name, conn in parent2['connections'].items()\
                            if conn['innovation'] == gene_innovation}
            child1_genes = copy.deepcopy((parent1_gene, parent2_gene)[rnd_val])
            child2_genes = copy.deepcopy((parent1_gene, parent2_gene)[1 - rnd_val])
            assert len(child1_genes) == 1 and len(child2_genes) == 1
            if not all([[*parent1_gene.values()][0]['enabled'],\
                        [*parent2_gene.values()][0]['enabled']]):
                [*child1_genes.values()][0]['enabled'] = np.random.random() > disable_prob
                [*child2_genes.values()][0]['enabled'] = np.random.random() > disable_prob
            child_1['connections'].update(child1_genes)
            child_2['connections'].update(child2_genes)
        
        #* Disjoint and excess connection genes
        for gene_innovation in (innov_ids_1, innov_ids_2)[f2 >= f1] - common_genes:
            winner_gene = {name : conn.copy() for name, conn in (parent1, parent2)[f2 >= f1]['connections'].items()\
                            if conn['innovation'] == gene_innovation}
            assert len(child1_genes) == 1 and len(child2_genes) == 1
            child_1['connections'].update(winner_gene)
            child_2['connections'].update(winner_gene)

        #* Crossover Nodes
        for node_name in (parent1, parent2)[f2 >= f1]['nodes']:
            if node_name in parent1['nodes'] and node_name in parent2['nodes']:
                rnd_gene = np.random.random() > 0.5
                child_1['nodes'][node_name] = (parent1, parent2)[rnd_gene]['nodes'][node_name].copy()
                child_2['nodes'][node_name] = (parent1, parent2)[not rnd_gene]['nodes'][node_name].copy()
            else:
                child_1['nodes'][node_name] = (parent1, parent2)[f2 >= f1]['nodes'][node_name].copy()
                child_2['nodes'][node_name] = (parent1, parent2)[f2 >= f1]['nodes'][node_name].copy()

        #* Formalize recombination
        do_crossover = np.random.random() < crossover_prob
        offspring.append((parent1, child_1)[do_crossover])
        offspring.append((parent2, child_2)[do_crossover])
    return offspring

@evo_operator_registry(name='uniform_crossover')
def uniform_crossover(parents, random_pairs=False, crossover_prob=1.):
    """ Uniform crossover of GA. Genotypes are pairwise grouped and 
    recombined, resulting in two children per recombination. A recombination 
    is applied with a fixed probability crossover_prob. In uniform crossover 
    each gene of the children is selected from one of the parents with the 
    same probability.
    ========================================================================
    - Args:
        parents [list of np.ndarray]: list of genotypes to be recombined.
        random_pairs [bool]: whether to shuffle the parents or mate them 
                preserving the order. Note that the mating operation can be
                accomplished using the mating operators. 
        crossover_prob [float]: probability of performing crossover between
                two parents.
    - Returns:
        offspring [list of np.ndarray]: list of offspring genotypes 
                resulting from recombination.
    ========================================================================
    """
    offspring = []
    if random_pairs:
        np.random.shuffle(parents)
    if len(parents) % 2:
        offspring.append(parents.pop(0))
    for parent1, parent2  in zip(parents[::2], parents[1::2]):
        crossover_mask = np.random.randint(2, size=parent1.shape[0])
        new1 = parent1 * crossover_mask + parent2 * (1 - crossover_mask)
        new2 = parent2 * crossover_mask + parent1 * (1 - crossover_mask)
        do_crossover = np.random.random() < crossover_prob
        offspring.append((parent1, np.hstack(new1))[do_crossover])
        offspring.append((parent2, np.hstack(new2))[do_crossover])
    return offspring

@evo_operator_registry(name='onepoint_crossover')
def onepoint_crossover(parents, random_pairs=False, crossover_prob=1.):
    """ One-point crossover of GA. Genotypes are pairwise grouped and 
    recombined, resulting in two children per recombination. A recombination 
    is applied with a fixed probability crossover_prob. In one-point 
    crossover, a point or index is selected for each pair of parents, and 
    the children are created as the combination of the partition elements 
    of the parents' genotypes.
    See the following example:
        Parent 1: AB|C  -> AB|F
                        
        Parent 2: DE|F  -> DE|C
    ========================================================================
    - Args:
        parents [list of np.ndarray]: list of genotypes to be recombined.
        random_pairs [bool]: whether to shuffle the parents or mate them 
                preserving the order. Note that the mating operation can be
                accomplished using the mating operators. 
        crossover_prob [float]: probability of performing crossover between
                two parents.
    - Returns:
        offspring [list of np.ndarray]: list of offspring genotypes 
                resulting from recombination.
    ========================================================================
    """
    offspring = []
    if random_pairs:
        np.random.shuffle(parents)
    if len(parents) % 2:
        offspring.append(parents.pop(0))
    for parent1, parent2 in zip(parents[::2], parents[1::2]):
        cut_idx = np.random.randint(parent1.shape[0])
        new1 = np.hstack((parent1[:cut_idx], parent2[cut_idx:]))
        new2 = np.hstack((parent2[:cut_idx], parent1[cut_idx:]))
        do_crossover = np.random.random() < crossover_prob
        offspring.append((parent1, np.hstack(new1))[do_crossover])
        offspring.append((parent2, np.hstack(new2))[do_crossover])
    return offspring

@evo_operator_registry(name='blxalpha_crossover')
def blxalpha_crossover(parents, random_pairs=False, crossover_prob=1., alpha=.3):
    offspring = []
    if random_pairs:
        np.random.shuffle(parents)
    if len(parents) % 2:
        offspring.append(parents.pop(0))
    for parent1, parent2 in zip(parents[::2], parents[1::2]):
        genes_min = np.min((parent1, parent2), axis=0) - alpha * np.abs(parent1-parent2)
        genes_max = np.max((parent1, parent2), axis=0) + alpha * np.abs(parent1-parent2)
        new1 = np.random.random(size=parent1.shape) * (genes_max - genes_min) + genes_min
        new2 = np.random.random(size=parent2.shape) * (genes_max - genes_min) + genes_min
        do_crossover = np.random.random() < crossover_prob
        offspring.append((parent1, np.hstack(new1))[do_crossover])
        offspring.append((parent2, np.hstack(new2))[do_crossover])
    return offspring


@evo_operator_registry(name='combination_crossover')
def combination_crossover(parents, random_pairs=False, crossover_prob=1., alpha=.3):
    offspring = []
    if random_pairs:
        np.random.shuffle(parents)
    if len(parents) % 2:
        offspring.append(parents.pop(0))
    for parent1, parent2 in zip(parents[::2], parents[1::2]):
        genes_min = np.min((parent1, parent2), axis=0) - alpha * np.abs(parent1-parent2)
        genes_max = np.max((parent1, parent2), axis=0) + alpha * np.abs(parent1-parent2)
        new1 = np.random.random(size=parent1.shape) * (genes_max - genes_min) + genes_min
        new2 = np.random.random(size=parent2.shape) * (genes_max - genes_min) + genes_min
        do_crossover = np.random.random() < crossover_prob
        offspring.append((parent1, np.hstack(new1))[do_crossover])
        offspring.append((parent2, np.hstack(new2))[do_crossover])
    return offspring

@evo_operator_registry(name='multipoint_crossover')
def multipoint_crossover(parents, ncuts=3, crossover_prob=1.):
    offspring = []
    if len(parents) % 2: 
        offspring.append(parents.pop(np.random.choice(len(parents))))
    for parent1, parent2  in zip(parents[::2], parents[1::2]):
        cut_idxs = np.sort(np.random.choice(range(1, parent1.shape[0] - 1), size=ncuts, replace=False))
        cut_idxs = np.hstack(([0], cut_idxs, [None]))
        new1 = []; new2 = []
        for ii, cut_idx in enumerate(cut_idxs[:-1]):
            if ii % 2 == 0:
                new1.append(parent1[cut_idx:cut_idxs[ii + 1]])
                new2.append(parent2[cut_idx:cut_idxs[ii + 1]])
            else:
                new1.append(parent2[cut_idx:cut_idxs[ii + 1]])
                new2.append(parent1[cut_idx:cut_idxs[ii + 1]])
        do_crossover = np.random.random() < crossover_prob
        offspring.append((parent1, np.hstack(new1))[do_crossover])
        offspring.append((parent2, np.hstack(new2))[do_crossover])
    return offspring


@evo_operator_registry(name='simulated_binary_crossover')
def simulated_binary_crossover(parents, eta=.5, crossover_prob=1.):#! Mirar valores eta
    offspring = []
    if len(parents) % 2:
        offspring.append(parents.pop(np.random.choice(len(offspring))))
    for parent1, parent2 in zip(parents[::2], parents[1::2]):
        mu = np.random.random()
        beta = ((2*mu, .5/(1-mu))[mu >= .5]) ** (1/(eta+1))
        new1 = .5 * ((1+beta) * parent1 + (1-beta) * parent2)
        new2 = .5 * ((1-beta) * parent1 + (1+beta) * parent2)
        do_crossover = np.random.random() < crossover_prob
        offspring.append((parent1, np.hstack(new1))[do_crossover])
        offspring.append((parent2, np.hstack(new2))[do_crossover])
    return offspring

@evo_operator_registry(name='pcx_crossover')
def pcx_crossover(parents, crossover_prob=1.):
    #TODO
    raise NotImplementedError 


def combined_crossover(parents, eta=1., crossover_prob=1.):
    #TODO
    raise NotImplementedError 


# def blxalpha_beta_crossover(parents, random_pairs=False, crossover_prob=1., alpha=.3):
#     offspring = []
#     if random_pairs: np.random.shuffle(parents)
#     if len(parents) % 2: offspring.append(parents.pop(0))
#     for parent1, parent2 in zip(parents[::2], parents[1::2]):
#         genes_min = np.min((parent1, parent2),axis=0)-alpha*np.abs(parent1-parent2)
#         genes_max = np.max((parent1, parent2),axis=0)+alpha*np.abs(parent1-parent2)
#         new1 = np.random.random(size=parent1.shape)*(genes_max - genes_min) + genes_min
#         new2 = np.random.random(size=parent2.shape)*(genes_max - genes_min) + genes_min
#         do_crossover = np.random.random() < crossover_prob
#         offspring.append((parent1, np.hstack(new1))[do_crossover])
#         offspring.append((parent2, np.hstack(new2))[do_crossover])
#     return offspring
