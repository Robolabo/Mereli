import numpy as np
from collections import deque
import copy
from mereli.neural_networks import NeuralNetwork
from mereli.register import normalizations, distributions, mutations

class BaseGene:
    def __init__(self):
        self._innovation = None
        self._value = None
        self._encoded_struct = None
        self._mutation = {}
        self._initialization = {}
        self._normalization = {}
        self.parameters = {}

    def configure_mutation(self, gene_info):
        mutation_type = gene_info['mutation']['type']
        params = {key : value for key, value in gene_info['mutation'].items() if key != 'type'}
        return mutations.get(mutation_type)(**params)

    def configure_initialization(self, gene_info):
        dist_type = gene_info['initialization']['type']
        params = {key : value for key, value in gene_info['initialization'].items() if key != 'type'}
        return distributions.get(dist_type)(**params)

    def configure_normalization(self, gene_info):
        if gene_info['normalization'] is None:
            return None
        norm_type = gene_info['normalization']['type']
        params = {key : value for key, value in gene_info['normalization'].items() if key != 'type'}
        return normalizations.get(norm_type)(**params)

    def initialize(self):
        for name in self.parameters:
            if name in self._initialization:
                value = self._initialization[name]()
                if self._normalization.get(name) is not None:
                    value = self._normalization[name].apply(value)
                self.parameters[name] = value
        
    def mutate(self):
        for name, param in self.parameters.items():
            if name in self._mutation:
                self.parameters[name] = self._mutation[name](param)


    def copy(self):
        return copy.deepcopy(self)

    @property
    def as_dict(self):
        pass

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


class NodeGene(BaseGene):
    def __init__(self, name, *args,  **kwargs):
        super(NodeGene, self).__init__(*args, **kwargs)
        self.name = name
        self.ensemble = name
        self.idx = None
        self.is_output = False
        self.enabled = True
        self.topology = None

    def configure(self, gene_info):
        for name, info in gene_info.items():
            topology, struct, param_name = name.split(':')[:3]
            if struct == 'nodes' and topology == self.topology:
                self._mutation[param_name] = self.configure_mutation(info)
                self._initialization[param_name] = self.configure_initialization(info)
                self._normalization[param_name] = self.configure_normalization(info)
                if param_name not in self.parameters:
                    self.parameters[param_name] = None # Init later

    def add_parameter(self, name, value):
        self.parameters[name] = value

    def as_dict(self):
        denorm_parameters = {param : self._normalization[param].revert(value) if param in self._normalization else value\
                for param, value in self.parameters.items()}
        return {**{'ensemble' : self.ensemble}, **denorm_parameters}

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
        self.topology = None


    def configure(self, gene_info):
        for name, info in gene_info.items():
            topology, struct, param_name = name.split(':')[:3]
            if struct == 'connections' and topology == self.topology:
                self._mutation[param_name] = self.configure_mutation(info)
                self._initialization[param_name] = self.configure_initialization(info)
                self._normalization[param_name] = self.configure_normalization(info)
                self.parameters[param_name] = None # Init later
            # elif struct == 'connections':
            #     import pdb; pdb.set_trace()
        
    def add_parameter(self, name, value):
        self.parameters[name] = value

    @property
    def weight(self):
        return self.parameters.get('weight')
    
    @property
    def has_learning_rule(self):
        return self.learning_rule is not None


    def as_dict(self):
        denorm_parameters = {param : self._normalization[param].revert(value) if param in self._normalization else value\
                for param, value in self.parameters.items()}
        return {**{'pre' : self.pre, 'post' : self.post, 'group' : self.group, 
                'enabled' : self.enabled}, **denorm_parameters}

class GraphGenotype:
    def __init__(self, g_id):
        self.g_id = g_id
        self.gene_info = {}
        self.neural_net_config = {}
        self._fitness = None
        self.eval_time = None
        self._evolvable_structs = None
        self._node_genes = deque([])
        self._connection_genes = deque([])
        self._species = None

    def initialize(self):
        for node in self.nodes:
            node.initialize()
        for conn in self.connections:
            conn.initialize()

    def configure(self, targets, gene_info, ann_config):
        self.gene_info = gene_info
        self.neural_net_config = ann_config
        self.targets = targets
        self._phenotype = {}
        for target in targets:
            # Create phenotype provisionally to build genotype.
            topology_name = target['topology']
            topology = self.neural_net_config[topology_name]
            self._phenotype[topology_name] = NeuralNetwork(topology['dt'], time_scale=topology['time_scale'],\
                neuron_model=topology['neuron_model'], synapse_model=topology['synapse_model'])
            self._phenotype[topology_name].build_from_dict(topology)
            __import__('pdb').set_trace()
            # Add genes to genotype
            for name, node in self._phenotype[topology_name].graph['neurons'].items():
                gene_name = name
                self.add_node_from_dict(gene_name, **node)
                self.get_node(gene_name).topology = topology_name
                self.get_node(gene_name).configure(gene_info)
            for name, conn in self._phenotype[topology_name].graph['synapses'].items():
                gene_name = name
                self.add_conn_from_dict(gene_name, **conn)
                self.get_connection(gene_name).topology = topology_name
                self.get_connection(gene_name).configure(gene_info)
                self.get_connection(gene_name).group = conn['group']
                

    def as_phenotype(self):
        phenotypes = {}
        for target in self.targets:
            topology_name = target['topology']
            topology = self.neural_net_config[topology_name]
            phenotype = NeuralNetwork(topology['dt'], time_scale=topology['time_scale'],\
                    neuron_model=topology['neuron_model'], synapse_model=topology['synapse_model'])
            phenotype.build_from_dict(topology)
            phenotype.reset_graph()
            for node in self.nodes:
                if node.topology == topology_name:
                    phenotype.add_neuron(node.name, **node.as_dict())
                    if node.is_output:
                        phenotype.set_motor(node.ensemble)
            for conn in self.connections:
                if conn.topology == topology_name:
                    phenotype.add_synapse(conn.name, **conn.as_dict())
            phenotype.build()
            phenotypes[target['object']] = phenotype
        return phenotypes

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
            elif param != 'is_motor':
                new_gene.add_parameter(param, val)
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
        if not len(candidates) > 0:
            import pdb; pdb.set_trace()
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
