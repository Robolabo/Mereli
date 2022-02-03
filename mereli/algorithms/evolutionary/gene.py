import numpy as np
from collections import deque
import copy

class BaseGene:

    def __init__(self):
        self._innovation = None
        self._value = None
        self._encoded_struct = None
        self._max_dec_val = None
        self._min_dec_val = None
    
    def copy(self):
        return copy.deepcopy(self)

    @property 
    def innovation(self):
        return self._innovation

    @innovation.setter 
    def innovation(self, new_innovation):
        self._innovation = new_innovation

    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, gene_val):
        self._value = gene_val
    
    @property
    def encoded_struct(self):
        return self._encoded_struct
    
    @encoded_struct.setter
    def encoded_struct(self, new_encoded_struct):
        self._encoded_struct = new_encoded_struct

    @property
    def max_dec_val(self):
        return self._max_dec_val
    
    @max_dec_val.setter
    def max_dec_val(self, new_max):
        self._max_dec_val = new_max

    @property
    def min_dec_val(self):
        return self._min_dec_val
    
    @min_dec_val.setter
    def min_dec_val(self, new_min):
        self._min_dec_val = new_min




# class BaseGenotype:
#     def __init__(self):


class FixedLenGenotype:
    def __init__(self, encoding='real'):
        self.encoding = encoding
        self._fitness = None
        self.genes = deque([])

    def add_gene(self, gene_value, encoded_struct=None, min_dec_val=0, max_dec_val=1):
        if self.encoding == 'binary' and isinstance(gene_value, float):
            # gene_value = np.base_repr(gene_value)
            raise NotImplementedError
        gene = BaseGene()
        gene.value = gene_value
        gene.encoded_struct = encoded_struct
        gene.min_dec_val = min_dec_val
        gene.max_dec_val = max_dec_val
        gene.innovation = len(self)
        self.genes.append(gene)

    def __iter__(self):
        for gene in self.genes:
            yield gene

    @property
    def values(self):
        return np.array([g.value for g in self])

    # def __next__(self):

    def __len__(self):
        return len(self.genes)

    def genes_of_struct(self, encoded_structure):
        return filter(lambda gene: gene.encoded_struct == encoded_structure, self)

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, fitness_value):
        self._fitness = fitness_value



class NodeGene(BaseGene):
    def __init__(self, name, *args,  **kwargs):
        super(NodeGene, self).__init__(*args, **kwargs)
        self.name = name
        self.ensemble = name
        self.idx = None
        self.activation = 'sigmoid'
        self.is_output = False
        self.enabled = True
        self.parameters = {}

    def add_parameter(self, name, value):
        self.parameters[name] = value

    # def set_params(self, tau=None, bias=None, gain=None, activation=None):
    #     if tau is not None:
    #         self.tau = tau
    #     if bias is not None:
    #         self.bias = bias
    #     if gain is not None:
    #         self.gain = gain
    #     if activation is not None:
    #         self.activation = activation
    # def value(self):
    #     return {''}


    
class ConnectionGene(BaseGene):
    def __init__(self, name, *args, 
            pre=None, post=None, 
            **kwargs):
        super(ConnectionGene, self).__init__(*args, **kwargs)
        self.name = name
        self.group = name
        self.pre = pre
        self.post = post
        self.innovation = None
        self.idx = None
        self.enabled = True
        self.learning_rule = None
        self.parameters = {}

    def add_parameter(self, name, value):
        self.parameters[name] = value

    @property
    def weight(self):
        return self.parameters.get('weight')
    
    @property
    def has_learning_rule(self):
        return self.learning_rule is not None

    def value(self):
        return {''}

class GraphGenotype:
    def __init__(self):
        self._fitness = None
        self._evolvable_structs = None  
        self._node_genes = deque([])
        self._connection_genes = deque([])
        self._species = None

    
    def add_node(self, gene):
        self._node_genes.append(gene)
    
    def add_connection(self, gene):
        self._connection_genes.append(gene)

    def add_node_from_dict(self, name, **kwargs):
        new_gene = NodeGene(name)
        new_gene.is_output = kwargs.get('is_motor', False)
        for param, val in kwargs.items():
            if hasattr(new_gene, param):
                setattr(new_gene, param, val)
        self._node_genes.append(new_gene)

    def add_conn_from_dict(self, name, pre, post, weight, **kwargs):
        new_gene = ConnectionGene(name)
        new_gene.pre = pre
        new_gene.post = post
        if 'learning_rule' in kwargs and kwargs['learning_rule']['name'] is not None:
            new_gene.learning_rule = kwargs['learning_rule']['name']
        
        for param, val in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, val)
        self._connection_genes.append(new_gene)
    
    # def delete_connection(self, name):
    #     idx = [conn. for conn in self.connections]
    #     import pdb; pdb.set_trace()
    #     self._connection_genes.pop()

    def copy(self):
        return copy.deepcopy(self)

    def get_node(self, name):
        candidates = [gene for gene in self.nodes if gene.name == name]
        assert len(candidates) > 0
        return candidates[0]

    def get_connection(self, name):
        candidates = [gene for gene in self.connections if gene.name == name]
        assert len(candidates) > 0
        return candidates[0]

    def contains_connection(self, name):
        return name in [gene.name for gene in self.connections]
    
    def contains_node(self, name):
        return name in [gene.name for gene in self.nodes]

    @property
    def nodes(self):
        for node in self._node_genes:
            yield node

    @property   
    def connections(self):
        for conn in self._connection_genes:
            yield conn


    @property
    def enabled_connections(self):
        for conn in self.connections:
            if conn.enabled:
                yield conn

    @property
    def num_nodes(self):
        return len(self._node_genes)

    @property
    def num_connections(self):
        return len(self._connection_genes)

    @property
    def evolvable_structs(self):
        return self._evolvable_structs

    @evolvable_structs.setter
    def evolvable_structs(self, evolvable_structs):
        self._evolvable_structs = evolvable_structs

    @property
    def species(self):
        return self._species
    
    @species.setter
    def species(self, new_species):
        self._species = new_species