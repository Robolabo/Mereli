import numpy as np
from spike_swarm_sim.register import evo_operator_registry
from spike_swarm_sim.utils import ShapeMismatchException

@evo_operator_registry(name='gaussian_mutation')
def gaussian_mutation(population, **kwargs):
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
    new_pop = []
    for indiv in population:
        mutation_mask = np.random.random(size=indiv.shape) < kwargs['mutation_prob']
        mutated = indiv + mutation_mask * np.random.randn(indiv.shape[0]) * kwargs['sigma']
        new_pop.append(mutated)
    return new_pop

@evo_operator_registry(name='uniform_mutation')
def uniform_mutation(population, **kwargs):
    """ Uniform mutation operator for GA. Each genotype gene is mutated with a 
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