import copy
import numpy as np
from .interpreter import language_dict
from mereli.neural_networks.mlp import MLP
from mereli.neural_networks.neuron_models import Activation

class GeneticInterface:
    """ Interface used by all the evolutionary algorithms to manage phenotype to 
    genotype transactions (bijectively), genotype initialization and so on.

    - Params:
        neural_net [NeuralNetwork] : ANN of a robot.
    """
    def __init__(self, neural_net):
        self.neural_net = neural_net 

    def submit_query(self, query, primitive='GET', **kwargs):
        """
        Submits a query to read or write the ANN. The operation is of the syntax 
        "PRIMITIVE query data" (data if primitive is write). 
        - Args:
            query [str] : query subject to the primitive. The query addresses some 
                    variable of the ANN and it has the following syntax:
                        "ANN_part:variable:name"
                    Some examples are:
                        "synapses:weights:all" -> weights of all the synapses.
                        "synapses:weights:S1" -> weights of synapse with name S1 (must be)
                                                 defined
                        "neurons:tau:all" -> time constants of all neuron models.
                        "neurons:gain:sensory" -> gains of sensory neurons.
                        "decoding:weights:all" -> weights of all decoders (in SNN).
            primitive [str]: action or primitive of the query. Possible primitives 
                    are GET, SET, LEN, INIT.
            data [np.ndarray]: genotype array only supplied when the primitive is SET.
            min_vals [np.ndarray]: minimum bound of the search space. In GET it is None.
            max_vals [np.ndarray]: max. bound of the search space. In GET it is None.
        """
        query_hierarchy = [primitive] + query.split(':')
        query_status = language_dict
        for query_elem in query_hierarchy[:-1]:
            assert query_elem in query_status
            query_status = query_status[query_elem]
        func = getattr(getattr(self.neural_net, query_hierarchy[1]), query_status)\
               if query_hierarchy[1] not in ['decoding', 'encoding']\
               else getattr(self.neural_net, query_status)
        return func(query_hierarchy[-1], self.neural_net.graph, **kwargs)


    def toGenotype(self, queries, min_vals, max_vals):
        """ Converts a phenotype or structured ANN into a vector genotype. 
        It performs a series of queries (depending on the population segments) 
        and gathers the results as the final genotype.
        """
        genotype = np.hstack([self.submit_query(query, primitive='GET',\
                   min_val=min_val, max_val=max_val)\
                   for query, max_val, min_val in zip(queries, max_vals, min_vals)])
        return genotype


    def fromGenotype(self, queries, genotype, min_vals, max_vals):
        """ Converts a genotype into a phenotype or, in this case, structured ANN. 
        It iterates across population opt. vars. with the corresponding queries and 
        submits a SET operation towards the ANN.
        """
        for query, max_val, min_val in zip(queries, max_vals, min_vals):
            genes = genotype.genes_of_struct(query)
            genes_values = np.array([gene.value for gene in genes])
            self.neural_net.graph = self.submit_query(query, primitive='SET',\
                    data=genes_values, min_val=min_val, max_val=max_val)
        self.neural_net.build() #* Compile changes.
    
    def initGenotype(self, queries, min_vals, max_vals):
        """ Method for initializing the values of the genotype.
        """
        for query, max_val, min_val in zip(queries, max_vals, min_vals):
            self.neural_net.graph = self.submit_query(query, primitive='INIT', min_val=min_val, max_val=max_val)
        self.neural_net.build()

class NEATInterface(GeneticInterface):
    def __init__(self, neural_net):
        super(NEATInterface, self).__init__(neural_net)

    def fromGenotype(self, queries, genotype, min_vals, max_vals):
        """ Converts a genotype into a phenotype or, in this case, structured ANN.
        """
        effective_genotype = copy.deepcopy(genotype)  
        #* Clean previous architecture
        self.neural_net.reset_graph()
        #* Add neurons
        for node in effective_genotype.nodes:
            if node.name in self.neural_net.graph['neurons']:
                for key, param in self.neural_net.graph['neurons'][node.name].items():
                    if hasattr(node, key) and key != 'idx':
                        self.neural_net.graph['neurons'][node.name][key] = getattr(node, key)
                continue
            # neuron_params = {param : getattr(node, param) for param in ['bias', 'gain', 'activation', 'tau'] \
            #                 if hasattr(self.neural_net.neurons, param) and hasattr(node, param)} #! Modify for SNN
            self.neural_net.add_neuron(node.name, node.ensemble, **node.parameters)#! OJO default activation
            if node.is_output:
                self.neural_net.set_motor(node.ensemble)
        for conn in effective_genotype.enabled_connections:
            self.neural_net.add_synapse(conn.name, conn.pre, conn.post, weight=conn.weight)
            if conn.has_learning_rule:
                self.neural_net.add_learning_rule(conn.name, conn.learning_rule, conn.parameters['lr_weight'])
        #* Update parameters (Decoders and encoders not supported yet).
        for query, max_val, min_val in zip(queries, max_vals, min_vals):
            gene_type = {'synapses' : 'connections', 'neurons' : 'nodes'}.get(query.split(':')[0], 'connections')
            variable = query.split(':')[1]
            if variable == 'weights': 
                variable = 'weight'
            if 'learning_rule' in query:
                genotype_segment = np.array([gene.parameters['lr_weight'] for gene in filter(lambda x: x.has_learning_rule and x.enabled, effective_genotype.connections)]).flatten()
            else:
                if gene_type == 'connections':
                    genotype_segment = np.array([gene.parameters[variable] for gene in getattr(effective_genotype, gene_type) if gene.enabled]).flatten()
                else:
                    genotype_segment = np.array([gene.parameters[variable] for gene in getattr(effective_genotype, gene_type)]).flatten()
            self.neural_net.graph = self.submit_query(query, primitive='SET',\
                        data=genotype_segment, min_val=min_val, max_val=max_val)
        self.neural_net.build() #* Compile changes.
        # conn_g = [n.name for n in genotype.connections if n.enabled]
        # conn_ann = [n for n, v in self.neural_net.graph['synapses'].items()]
        # w_g = np.array([n.parameters['weight'] for n in genotype.connections if n.enabled])
        # w_ann = np.array([v['weight'] for n, v in self.neural_net.graph['synapses'].items()])
        # import pdb; pdb.set_trace()

    def initGenotype(self, queries, min_vals, max_vals):
        """ Method for initializing the values of the genotype.
        """
        for query, max_val, min_val in zip(queries, max_vals, min_vals):
            self.neural_net.graph = self.submit_query(query, primitive='INIT', min_val=min_val, max_val=max_val)
        self.neural_net.build()

class CPPN_NEAT_Interface(NEATInterface):
    def __init__(self, neural_net):
        self.n_inputs = 3
        dev_mlp = MLP()
        dev_mlp.add_stimuli('I', self.n_inputs, sensor='I')
        dev_mlp.add_ensemble('O', 1, bias=0.0, activation=Activation.LINEAR)
        dev_mlp.set_motor('O')
        dev_mlp.add_synapse('I-O', 'I', 'O', weight=1., conn_prob=1.)
        dev_mlp.build()
        self.final_ANN = neural_net
        super(CPPN_NEAT_Interface, self).__init__(dev_mlp)

    def dev_decode(self, genotype):
        pass 

    def fromGenotype(self, queries, genotype, min_vals, max_vals):
        #! REVISAR
        super().fromGenotype(queries, genotype, min_vals, max_vals)
        n_neurons = 100 
        W_shape = (n_neurons, n_neurons + self.final_ANN.num_inputs)
        xx, yy = np.meshgrid(np.linspace(-1, 1, W_shape[0]), np.linspace(-1, 1, W_shape[1]) )
        self.neural_net.neurons.activation = [Activation.GAUSSIAN] * self.neural_net.num_neurons#!
        zz = np.reshape([self.neural_net.step(np.r_[x, y, np.linalg.norm(np.r_[x, y])])\
                    for x, y in zip(xx.flatten(), yy.flatten())], W_shape)
        self.final_ANN.build_from_adjmat(zz)
        import pdb; pdb.set_trace()


    def initGenotype(self, queries, min_vals, max_vals):
        for query, max_val, min_val in zip(queries, max_vals, min_vals):
            kwargs = {'max_val' : max_val, 'min_val' : min_val} if 'activation' not in query else {}
            self.neural_net.graph = self.submit_query(query, primitive='INIT', **kwargs)
        self.neural_net.build()

class InterfaceFactory:
    def create(self, algorithm, neural_net):
        return {
            'GeneticAlgorithm' : GeneticInterface,
            'SNES' : GeneticInterface,
            'xNES' : GeneticInterface,
            'CMA_ES' : GeneticInterface,
            'OpenAI_ES' : GeneticInterface,
            'NEAT' : NEATInterface,
            'CPPN_NEAT' : CPPN_NEAT_Interface
        }[algorithm](neural_net)
