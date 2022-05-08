import copy
import numpy as np
from mereli.register import evo_operator_registry
from ..gene import GraphGenotype

@evo_operator_registry(name='uniform_crossover')
def uniform_crossover(genotypeA, genotypeB, crossover_prob=1.):
    """ Uniform crossover of GA. Genotypes are pairwise grouped and 
    recombined, resulting in two children per recombination. A recombination 
    is applied with a fixed probability crossover_prob. In uniform crossover 
    each gene of the children is selected from one of the parents with the 
    same probability.
    """
    childA, childB = GraphGenotype(genotypeA.g_id), GraphGenotype(genotypeB.g_id)
    for connA, connB in zip(genotypeA.connections, genotypeB.connections):
        if np.random.random() < 0.5:
            childA.add_connection(connA.copy())
            childB.add_connection(connB.copy())
        else:
            childA.add_connection(connB.copy())
            childB.add_connection(connA.copy())
    for nodeA, nodeB in zip(genotypeA.nodes, genotypeB.nodes):
        if np.random.random() < 0.5:
            childA.add_node(nodeA.copy())
            childB.add_node(nodeB.copy())
        else:
            childA.add_node(nodeB.copy())
            childB.add_node(nodeA.copy())
    if np.random.random() < crossover_prob:
        return [childA, childB]
    else:
        return [genotypeA, genotypeB]

@evo_operator_registry(name='onepoint_crossover')
def onepoint_crossover(genotypeA, genotypeB, crossover_prob=1.):
    """ One-point crossover of GA. Genotypes are pairwise grouped and 
    recombined, resulting in two children per recombination. A recombination 
    is applied with a fixed probability crossover_prob. In one-point 
    crossover, a point or index is selected for each pair of parents, and 
    the children are created as the combination of the partition elements 
    of the parents' genotypes.
    See the following example:
        Parent 1: AB|C  -> AB|F
                        
        Parent 2: DE|F  -> DE|C
    """
    childA, childB = GraphGenotype(genotypeA.g_id), GraphGenotype(genotypeB.g_id)
    cut_conn_idx = np.random.randint(len(genotypeA.connections))
    cut_nodes_idx = np.random.randint(len(genotypeA.nodes))
    for i, (connA, connB) in enumerate(zip(genotypeA.connections, genotypeB.connections)):
        childA.add_connection(connA if i <= cut_conn_idx else connB)
        childB.add_connection(connB if i <= cut_conn_idx else connA)
    for i, (nodeA, nodeB) in enumerate(zip(genotypeA.nodes, genotypeB.nodes)):
        childA.add_node(nodeA if i <= cut_nodes_idx else nodeB)
        childB.add_node(nodeB if i <= cut_nodes_idx else nodeA)  
    if np.random.random() < crossover_prob:
        return [childA, childB]
    else:
        return [genotypeA, genotypeB]

@evo_operator_registry(name='blxalpha_crossover')
def blxalpha_crossover(genotypeA, genotypeB, crossover_prob=1., alpha=.3):
    childA, childB = GraphGenotype(genotypeA.g_id), GraphGenotype(genotypeB.g_id)
    for connA, connB in zip(genotypeA.connections, genotypeB.connections):
        childA.add_connection(connA.copy())
        childB.add_connection(connB.copy())
        for param in connA.parameters:
            paramA = connA.parameters[param]
            paramB = connB.parameters[param]
            g_min = min(paramA, paramB) - alpha * np.abs(paramA - paramB)
            g_max = max(paramA, paramB) + alpha * np.abs(paramA - paramB)
            childA.get_connection(connA.name).parameters[param] = np.random.random() * (g_max - g_min) + g_min
            childB.get_connection(connB.name).parameters[param] = np.random.random() * (g_max - g_min) + g_min
    for nodeA, nodeB in zip(genotypeA.nodes, genotypeB.nodes):
        childA.add_node(nodeA.copy())
        childB.add_node(nodeB.copy())
        for param in nodeA.parameters:
            paramA = nodeA.parameters[param]
            paramB = nodeB.parameters[param]
            g_min = min(paramA, paramB) - alpha * np.abs(paramA - paramB)
            g_max = max(paramA, paramB) + alpha * np.abs(paramA - paramB)
            childA.get_node(nodeA.name).parameters[param] = np.random.random() * (g_max - g_min) + g_min
            childB.get_node(nodeB.name).parameters[param] = np.random.random() * (g_max - g_min) + g_min
    if np.random.random() < crossover_prob:
        return [childA, childB]
    else:
        return [genotypeA, genotypeB]

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
    #! TODO
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


def neat_crossover(parents, crossover_prob=0.8, disable_prob=0.75):
    offspring = []
    if len(parents) % 2 != 0:
        offspring.append(parents.pop(0))
    for parent1, parent2 in zip(parents[::2], parents[1::2]):
        children = [GraphGenotype(len(offspring)), GraphGenotype(len(offspring)+1)]
        for child in children:
            child.gene_info = parent1.gene_info
            child.neural_net_config = parent1.neural_net_config
        innovations_1 = set([gene.innovation for gene in parent1.connections])
        innovations_2 = set([gene.innovation for gene in parent2.connections])
        common_genes = innovations_1.intersection(innovations_2)
        #* Common connection genes
        for gene_innovation in common_genes:
            parent1_gene = [conn for conn in parent1.connections\
                            if conn.innovation == gene_innovation][0]
            parent2_gene = [conn for conn in parent2.connections\
                            if conn.innovation == gene_innovation][0]
            dice = np.random.random() < 0.5
            children[0].add_connection(parent1_gene.copy() if dice else parent2_gene.copy())
            children[1].add_connection(parent1_gene.copy() if not dice else parent2_gene.copy())
        #* Disjoint and excess connection genes
        fittest_parent = parent1 if parent1.fitness >= parent2.fitness else parent2
        fittest_innovations = innovations_1 if parent1.fitness >= parent2.fitness else innovations_2
        for gene_innovation in fittest_innovations - common_genes:
            winner_gene = [conn for conn in fittest_parent.connections if conn.innovation == gene_innovation][0]
            children[0].add_connection(winner_gene.copy())
            children[1].add_connection(winner_gene.copy())

        #* Crossover Nodes
        for node in fittest_parent.nodes:
            if parent1.contains_node(node.name) and parent2.contains_node(node.name):
                dice = np.random.random() > 0.5
                selected_parent1 = parent1 if dice else parent2
                selected_parent2 = parent1 if not dice else parent2
                children[0].add_node(selected_parent1.get_node(node.name).copy())
                children[1].add_node(selected_parent2.get_node(node.name).copy())
            else:
                children[0].add_node(fittest_parent.get_node(node.name).copy())
                children[1].add_node(fittest_parent.get_node(node.name).copy())
        #* Formalize recombination
        do_crossover = np.random.random() < crossover_prob
        offspring.append((parent1.copy(), children[0])[do_crossover])
        offspring.append((parent2.copy(), children[1])[do_crossover])
    return offspring