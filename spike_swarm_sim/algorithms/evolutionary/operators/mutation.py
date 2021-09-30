from itertools import product
import numpy as np
from spike_swarm_sim.algorithms.evolutionary.gene import ConnectionGene
from spike_swarm_sim.register import evo_operator_registry
from spike_swarm_sim.utils import ShapeMismatchException
from ..gene import ConnectionGene, NodeGene


def add_node(genotype, current_innovation, innovation_history, node_variables, **kwargs):
    """ Add a new node in between an existing connection. The existing connection 
    is disabled and two new synapses are included.
    """
    
    #* Randomly select an enabled connection
    sel_conn = np.random.choice([*genotype.enabled_connections])
    sel_conn.enabled = False # Disable connection 

    #* Create and add new node gene in-between pre and post nodes of sel_conn
    node_name = f'Node_{sel_conn.innovation}'
    new_node = NodeGene(node_name)
    new_node.idx = genotype.num_nodes
    # Initialize randomly node parameters
    new_node.set_params(**{param : np.random.random() for param in node_variables})
    genotype.add_node(new_node)

    #* Add new connections
    for n, (pre, post) in enumerate(zip([node_name, sel_conn.pre], [sel_conn.post, node_name])):
        weight = (sel_conn.weight, np.clip(np.random.normal(loc=.5, scale=.1), 0, 1))[n]
        new_conn = ConnectionGene(f'{pre}-{post}', pre=pre, post=post, weight=weight)
        new_conn.innovation = innovation_history.get((pre, post), current_innovation) #!check
        new_conn.idx = genotype.num_connections
        if sel_conn.learning_rule is not None:
            new_conn.learning_rule = (sel_conn.learning_rule, {v : np.clip(np.random.normal(loc=.5, scale=.05), 0, 1)\
                                        for v in ['A', 'B', 'C', 'D']})[n]
        genotype.add_connection(new_conn)
        #* Update innovation dict
        if (pre, post) not in innovation_history:
            innovation_history[(pre, post)]= current_innovation
            current_innovation += 1
    return genotype, current_innovation, innovation_history

def add_connection(genotype, input_nodes, current_innovation, innovation_history, **kwargs):
    """ Add a new gene connection to the genotype. The pre and post 
    synaptic nodes are selected randomly (validating that the connection does not 
    exist).
    """
    node_names = [node.name for node in genotype.nodes]
    pos_conns = set([*product(input_nodes, node_names)] + [*product(node_names, repeat=2)])
    existing_conns = set([(conn.pre, conn.post) for conn in genotype.connections])
    allowed_conns = list(pos_conns - existing_conns)
    if len(allowed_conns) == 0:
        return genotype, current_innovation, innovation_history
    new_conn = allowed_conns[np.random.choice(range(len(allowed_conns)))]
    #* Create new connection gene
    conn_name = '-'.join(new_conn)
    new_connection = ConnectionGene(conn_name)
    new_connection.pre = new_conn[0]
    new_connection.post = new_conn[1]
    new_connection.weight = np.clip(0.1 * np.random.randn() + 0.5, a_min=0, a_max=1) # Random weight in [0,1] (denormalized later).
    new_connection.learning_rule = {v : np.random.random() for v in ['A', 'B', 'C', 'D']}
    new_connection.innovation = innovation_history.get((new_conn[0], new_conn[1]), current_innovation)
    new_connection.idx = len([*genotype.connections])

    if (new_conn[0], new_conn[1]) not in innovation_history:
        innovation_history.update({(new_conn[0], new_conn[1]): current_innovation})
        current_innovation += 1
    return genotype, current_innovation, innovation_history


def delete_node(genotype, input_nodes, **kwargs):
    #TODO: to be implemented (not ready yet)
    node = np.random.choice([*genotype['nodes']])
    if node in input_nodes or genotype['nodes'][node]['is_motor']:
        return
    for conn_name, conn in genotype['connections'].items():
        if node in [conn['pre'], conn['post']]:
            del genotype['connections'][conn_name]
    del genotype['nodes'][node]

    import pdb; pdb.set_trace()

def delete_connection(genotype, input_nodes, **kwargs):
    #TODO: to be implemented (not ready yet)
    conn = np.random.choice([*genotype['connections']])
    pre_node = genotype['connections'][conn]['pre']
    post_node = genotype['connections'][conn]['post']
    condition = any(conn2['pre'] == pre_node and conn2['post'] == post_node\
        for conn2_name, conn2 in genotype['connections'].items() if conn2_name != conn)
    if condition:
        return
    del genotype['connections'][conn]


def neat_mutation(population, input_nodes, current_innovation, innovation_history,
            mutable_variables, p_weight_mut=0.75, p_node_mut=0.03, p_conn_mut=0.5):
    for genotype in filter(lambda x: np.random.random() < p_weight_mut, population):
        #* Parameter Mutations
        for param in mutable_variables:
            gene_type = {'synapses' : 'connections', 'neurons' : 'nodes'}.get(param.split(':')[0], 'connections')
            variable = {'weights' : 'weight'}.get(param.split(':')[1], param.split(':')[1])
            for gene in getattr(genotype, gene_type): 
                if np.random.random() < 0.02:
                    if 'learning_rule' in param:
                        assert type(gene).__name__ == 'ConnectionGene'
                        setattr(gene, 'learning_rule', {v : np.random.random() for v in ['A', 'B', 'C', 'D']})
                    else:
                        setattr(gene, variable, np.random.random())
                else:
                    if 'learning_rule' in param:
                        assert type(gene).__name__ == 'ConnectionGene'
                        new_value = {}
                        for v in ['A', 'B', 'C', 'D']:
                            current_val = getattr(gene, 'learning_rule')[v]
                            new_value[v] = np.clip(current_val + np.random.randn() * .05, a_min=0, a_max=1)
                        setattr(gene, 'learning_rule', new_value)
                    else:
                        current_value = getattr(gene, variable)
                        setattr(gene, variable, np.clip(current_value + np.random.randn() * 0.05, a_min=0, a_max=1))

    #* Connnections mutations
    for genotype in filter(lambda x: np.random.random() < p_conn_mut, population):
        genotype, current_innovation, innovation_history = add_connection(genotype, input_nodes, current_innovation, innovation_history)
    #* Node mutations
    for genotype in filter(lambda x: np.random.random() < p_node_mut, population):
        genotype, current_innovation, innovation_history = add_node(genotype, current_innovation,
                    innovation_history, [var.split(':')[1] for var in mutable_variables if var.split(':')[0] == 'neurons'])
    return population, current_innovation, innovation_history

@evo_operator_registry(name='gaussian_mutation')
def gaussian_mutation(population, mutation_prob=0.05, sigma=0.1, min_vals=0, max_vals=1):
    """ Gaussian mutation operator for GA. Each genotype gene is mutated with a 
    probability mutation_prob. Mutation is accomplished by sampling a new gene value 
    from a gaussian dist. centered at the gene and with a std. dev. sigma.
    ================================================================================
    - Args:
        mutation_prob [float]: probability of mutating a gene.
        sigma [float]: std. dev. of the gaussian mutation.
    - Returns
        new_pop [list of np.ndarray]: list of mutated genotypes.
    ================================================================================
    """
    for genotype in population:
        genotype_vals = genotype.values
        mutation_mask = np.random.random(size=len(genotype)) < mutation_prob
        mutated_vals = genotype_vals + mutation_mask * np.random.randn(len(genotype)) * sigma
        mutated_vals = np.clip(mutated_vals, a_min=min_vals, a_max=max_vals)
        for mutated_val, gene in zip(mutated_vals, genotype):
            gene.value = mutated_val
    return population

@evo_operator_registry(name='uniform_mutation')
def uniform_mutation(population, **kwargs):

    """ TODO revise. 
    Uniform mutation operator for GA. Each genotype gene is mutated with a 
    probability mutation_prob. Mutation is accomplished by uniformly resampling the
    gene within the interval [min_vals[g], max_vals[g]], where min_vals and max_vals 
    are the bounds of each gene and g is the gene index.
    ================================================================================
    - Args:
        mutation_prob [float]: probability of mutating a gene.
        min_vals [float or np.ndarray]: minimum values of uniform mutation. It can be 
                either a numpy array of same length as the genotype or a float. If float, 
                it is assumed that all genes have the same min value. 
        max_vals [float or np.ndarray]: maximum values of uniform mutation. It can be 
                either a numpy array of same length as the genotype or a float. If float, 
                it is assumed that all genes have the same max value. 
    - Returns
        new_pop [list of np.ndarray]: list of mutated genotypes.
    ================================================================================
    """
    min_vals = {
        'int' : np.repeat(kwargs['min_vals'], len(population[0])),
        'float' : np.repeat(kwargs['min_vals'], len(population[0])),
        'list' : np.array(kwargs['min_vals']),
        'ndarray' : kwargs['min_vals']
    }[type(kwargs['min_vals']).__name__]
    max_vals = {
        'int' : np.repeat(kwargs['max_vals'], len(population[0])),
        'float' : np.repeat(kwargs['max_vals'], len(population[0])),
        'list' : np.array(kwargs['max_vals']),
        'ndarray' : kwargs['max_vals']
    }[type(kwargs['max_vals']).__name__]
    if len(min_vals) != len(population[0]):
        raise ShapeMismatchException('Dimension of minimum values of uniform mutation '\
            'did not match with the genotype length.')
    if len(max_vals) != len(population[0]):
        raise ShapeMismatchException('Dimension of maximum values of uniform mutation '\
            'did not match with the genotype length.')
    new_pop = []
    for indiv in population:
        mutation_mask = np.random.random(size=indiv.shape) < kwargs['mutation_prob']
        mutated = (1. - mutation_mask) * indiv + mutation_mask * np.random.uniform(low=min_vals, high=max_vals)
        new_pop.append(mutated)
    return new_pop

@evo_operator_registry(name='bitflip_mutation')
def bitFlip_mutation(population, **kwargs):
    """ Bit-Flip mutation operator for binary coded GA. Each genotype gene is mutated 
    with a probability mutation_prob. Mutation is accomplished by flipping the gene 
    bit (0 -> 1 or 1 -> 0).
    ================================================================================
    - Args:
        mutation_prob [float]: probability of mutating a gene.
        min_vals [float or np.ndarray]: minimum values of uniform mutation. It can be 
                either a numpy array of same length as the genotype or a float. If float, 
                it is assumed that all genes have the same min value. 
        max_vals [float or np.ndarray]: maximum values of uniform mutation. It can be 
                either a numpy array of same length as the genotype or a float. If float, 
                it is assumed that all genes have the same max value. 
    - Returns
        new_pop [list of np.ndarray]: list of mutated genotypes.
    ================================================================================
    """
    new_pop = []
    for indiv in population:
        mutation_mask = np.random.random(size=indiv.shape) < kwargs['mutation_prob']
        indiv[mutation_mask] = 1 - indiv[mutation_mask]
        new_pop.append(indiv.copy())
    return new_pop

@evo_operator_registry(name='categorical_mutation')
def categorical_mutation(population, **kwargs):
    new_pop = []
    for indiv in population:
        mutation_mask = np.random.random(size=indiv.shape) < kwargs['mutation_prob']
        indiv[mutation_mask] = np.random.randint(kwargs['min_vals'][mutation_mask], kwargs['max_vals'][mutation_mask])
        new_pop.append(indiv)
    return new_pop
