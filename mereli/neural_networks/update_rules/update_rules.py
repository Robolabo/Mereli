from collections import deque
import numpy as np
from mereli.algorithms.interfaces import GET, SET, LEN, INIT
from mereli.register import learning_rule_registry, learning_rules



class LearningRuleWrapper:
    def __init__(self):
        self._rules = {}

    def add_rule(self, synapse_name, rule_name):
        lr_instance = learning_rules[rule_name]()
        if rule_name not in self._rules: 
            self._rules[rule_name] = lr_instance

    def build(self, ann_graph):
        """  """    
        ref_mask = np.full((len(ann_graph['neurons']), len(ann_graph['inputs']) + len(ann_graph['neurons'])), False) 
        for rule in self._rules.values():
            rule.mask = ref_mask.copy()
            rule.weights = ref_mask.astype(float)
        n_inputs = len(ann_graph['inputs'])
        for syn in ann_graph['synapses'].values():
            if syn['enabled']:
                if syn['pre'] in ann_graph['inputs']:
                    pre_idx = ann_graph['inputs'][syn['pre']]['idx']
                else:
                    pre_idx = ann_graph['neurons'][syn['pre']]['idx'] + n_inputs
                post_idx = ann_graph['neurons'][syn['post']]['idx']
                if 'learning_rule' in syn:
                    self._rules[syn['learning_rule']['name']].mask[post_idx, pre_idx] = True                
                    self._rules[syn['learning_rule']['name']].weights[post_idx, pre_idx] = syn['learning_rule']['lr_weight']
        # Run specific build methods of each rule
        for rule in self._rules.values():
            rule.build()

    def step(self, synapses, activities, stimuli, reward=None):
        Weight_Delta = np.zeros_like(synapses.weights)
        for rule in self._rules.values():
            Weight_Delta += rule.step(synapses.weights, activities, stimuli, reward=reward)
        synapses.weights += Weight_Delta
        synapses.weights = np.clip(synapses.weights, a_min=-10, a_max=10)
        return synapses
    
    @property
    def weights(self):
        return np.sum([rule.weights * rule.mask for rule in self._rules], 0)
    

    def reset(self):
        pass

    @GET("learning_rule:lr_weights")
    def get_weights(self, conn_name, ann_graph, min_val=-1, max_val=1.):
        #* Return scaled in [0,1]
        if conn_name == 'all':
            
            weights = np.array([syn['learning_rule']['lr_weight'] for syn in \
                filter(lambda x: x['trainable'] and 'learning_rule' in x, ann_graph['synapses'].values())  ])
            return (weights - min_val) / (max_val - min_val)
        #* Special queries of synapses
        conn_name = {
            'sensory' : [key for key, syn in ann_graph['synapses'].items()\
                            if syn['pre'] in ann_graph['inputs']],
            'hidden' : [key for key, syn in ann_graph['synapses'].items()\
                        if syn['pre'] in ann_graph['neurons']\
                        and not ann_graph['neurons'][syn['pre']]['is_motor']],
            'motor' : [key for key, syn in ann_graph['synapses'].items()\
                        if syn['pre'] in ann_graph['neurons']\
                        and ann_graph['neurons'][syn['pre']]['is_motor']]
        }.get(conn_name, [conn_name])
        weights = np.array([ann_graph['synapses'][name]['learning_rule']['lr_weight']\
                for name in conn_name if ann_graph['synapses'][name]['trainable']])
        return (weights - min_val) / (max_val - min_val)

    @SET("learning_rule:lr_weights")
    def set_weights(self, conn_name, ann_graph, data, min_val=-1, max_val=1.,):
        """
        """
        #* rescale genotype segment to weight range
        data = min_val + data * (max_val - min_val)
        if conn_name == 'all':
            for w, syn in zip(data, filter(lambda x: x['trainable'] and 'learning_rule' in x, ann_graph['synapses'].values())):
                syn['learning_rule']['lr_weight'] = w
            return ann_graph
        else:
            #* Special queries of synapses
            conn_name = {
                'sensory' : [key for key, syn in ann_graph['synapses'].items()\
                                if syn['pre'] in ann_graph['inputs']],
                'hidden' : [key for key, syn in ann_graph['synapses'].items()\
                            if syn['pre'] in ann_graph['neurons']\
                            and not ann_graph['neurons'][syn['pre']]['is_motor']],
                'motor' : [key for key, syn in ann_graph['synapses'].items()\
                            if syn['pre'] in ann_graph['neurons']\
                            and ann_graph['neurons'][syn['pre']]['is_motor']]
            }.get(conn_name, [conn_name])
            for w, syn_name in zip(data, conn_name):
                if ann_graph['synapses'][syn_name]['trainable'] and 'learning_rule' in ann_graph['synapses'][syn_name]:
                    ann_graph['synapses'][syn_name]['learning_rule']['lr_weight'] = w
            return ann_graph

    @INIT('learning_rule:lr_weights')
    def init_weights(self, conn_name, ann_graph, min_val=0., max_val=1.):
        """
        """
        weights_len = self.len_weights(conn_name, ann_graph)
        random_weights = np.random.randn(weights_len)
        random_weights = np.clip(random_weights, a_min=0, a_max=1)
        return self.set_weights(conn_name, ann_graph, random_weights,\
                            min_val=min_val, max_val=max_val)

    @LEN('learning_rule:lr_weights')
    def len_weights(self, conn_name, ann_graph):
        """
        """
        return self.get_weights(conn_name, ann_graph).shape[0]




class BaseLearningRule:
    def __init__(self):
        # boolean mask, with the same shape as weight adjacency matrix, that states if the connection 
        # has the learning rule attached.
        self.mask = None
        self.weights = None

    def step(self, weights, activities, stimuli, reward=None):
        pass
    
    def build(self):
        pass

    def reset(self):
        pass

        
@learning_rule_registry(name='simple_hebb')
class SimpleHebbian(BaseLearningRule):
    def __init__(self):
        super(SimpleHebbian, self).__init__()
        self.modulated = False #!
        self.learning_rate = 0.1
        self.rule_weights = None

    def step(self, weights, activities, stimuli, reward=None):
        # return self.learning_rate * self.mask * self.weights * np.outer(activities, np.r_[stimuli, activities])
        W_tar = self.weights * np.outer(activities, np.r_[stimuli, activities])
        # return self.learning_rate * self.mask * (-weights + W_tar)
        return self.learning_rate * self.mask * W_tar

    def build(self):
        # import pdb; pdb.set_trace()
        pass

    
@learning_rule_registry(name='modulated_simple_hebb')
class ModulatedSimpleHebbian(SimpleHebbian):
    def __init__(self):
        super(ModulatedSimpleHebbian, self).__init__()
        self.modulated = True #!

    def step(self, *args, reward=None):
        assert reward is not None
        return reward * super().step(*args)




@learning_rule_registry(name='generalized_hebb')
class ModulatedSimpleHebbian(BaseLearningRule):
    def __init__(self):
        super(ModulatedSimpleHebbian, self).__init__()
        self.modulated = True #!

    def step(self, *args, reward=None):
        assert reward is not None
        return reward * super().step(*args)



def append_and_pop(queue, new_elem):
    queue.append(new_elem)
    queue.popleft()
    return queue

@learning_rule_registry(name='generalized_hebbian')
class GeneralizedHebbian:
    def __init__(self):
        self.modulated = False #!
        self.learning_rate = 5e-3
        self.A = 1.0
        self.B = 0.0
        self.C = 0.0
        self.D = 0.0
        self.gamma = 0.9
        self.timesteps_update = 50 # Gather XX rewards before updating for computing value func.
        self.reward_queue = deque([])
        self.activities_queue = deque([])
        self.inputs_queue = deque([])
        self.t = 0

    def __step(self, inputs, activities, reward=None):
        act_inpt_cat = np.r_[inputs, activities]
        weight_update = self.learning_rate * (
                        self.A * np.outer(activities, act_inpt_cat)\
                        + self.B * np.outer(activities, np.ones_like(act_inpt_cat))\
                        + self.C * np.outer(np.ones_like(activities), act_inpt_cat)\
                        + self.D)
        return weight_update * reward if reward is not None else weight_update
    
    def step(self, inputs, activities, reward=None):
        #!
        return self.__step(inputs, activities, reward=reward)
        #!
        if not self.modulated:
            return self.__step(inputs, activities, reward=1.)
        if self.t < self.timesteps_update:
            self.activities_queue.append(activities)
            self.inputs_queue.append(inputs)
            if reward is not None:
                self.reward_queue.append(reward)
            self.t += 1
            return 0.0
        weight_update = 0.0
        if self.timesteps_update > 1:
            self.activities_queue = append_and_pop(self.activities_queue, activities)
            self.inputs_queue = append_and_pop(self.inputs_queue, inputs)
            if reward is not None:
                self.reward_queue = append_and_pop(self.reward_queue, reward)
            value_fn = np.sum([rew * self.gamma ** k for k, rew in enumerate(self.reward_queue)])
            weight_update = self.__step(self.inputs_queue[0], self.activities_queue[0], reward=value_fn)
        else:
            import pdb; pdb.set_trace()
            weight_update = self.__step(inputs, activities, reward=reward)
        self.t += 1
        return weight_update

    def build(self, ann_graph):
        mask = np.full((len(ann_graph['neurons']), len(ann_graph['inputs']) + len(ann_graph['neurons'])), False)
        trainable_mask = mask.copy()
        A = mask.copy().astype(float)
        B = A.copy()
        C = A.copy()
        D = A.copy()
        for name, node in ann_graph['neurons'].items():
            in_connections = [syn for syn in ann_graph['synapses'].values() if syn['post'] == name]
            for syn in in_connections:
                if syn['enabled']:
                    pre_idx = ann_graph['inputs'][syn['pre']]['idx'] if syn['pre'] in ann_graph['inputs']\
                                else ann_graph['neurons'][syn['pre']]['idx'] + len(ann_graph['inputs'])
                    mask[node['idx'], pre_idx] = True
                    trainable_mask[node['idx'], pre_idx] = True
                    A[node['idx'], pre_idx] = syn.get('learning_rule', {}).get('A', 0.)
                    B[node['idx'], pre_idx] = syn.get('learning_rule', {}).get('B', 0.)
                    C[node['idx'], pre_idx] = syn.get('learning_rule', {}).get('C', 0.)
                    D[node['idx'], pre_idx] = syn.get('learning_rule', {}).get('D', 0.)
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        # self.learning_rate = np.random.randn(*self.A.shape) * 1e-3    


    def reset(self):
        self.t = 0
        self.reward_queue = deque([])
        self.activities_queue = deque([])
        self.inputs_queue = deque([])

    # #! OJO refactorizar queries!!!!
    # @GET("learning_rule:params")
    # def get_params(self, conn_name, ann_graph, min_val=0., max_val=1., only_trainable=True):
    #     #* Return scaled in [0,1]
    #     if conn_name == 'all':
    #         params = np.hstack([np.array([syn['learning_rule'][param] for syn in ann_graph['synapses'].values()\
    #                 if syn['trainable']]) for param in ['A', 'B', 'C', 'D']])
    #         return (params - min_val) / (max_val - min_val)
    #     #* Special queries of synapses
    #     conn_name = {
    #         'sensory' : [key for key, syn in ann_graph['synapses'].items()\
    #                         if syn['pre'] in ann_graph['inputs']],
    #         'hidden' : [key for key, syn in ann_graph['synapses'].items()\
    #                     if syn['pre'] in ann_graph['neurons']\
    #                     and not ann_graph['neurons'][syn['pre']]['is_motor']],
    #         'motor' : [key for key, syn in ann_graph['synapses'].items()\
    #                     if syn['pre'] in ann_graph['neurons']\
    #                     and ann_graph['neurons'][syn['pre']]['is_motor']]
    #     }.get(conn_name, [conn_name])
    #     weights = np.hstack([np.array([ann_graph['synapses'][name]['learning_rule'][param] for name in conn_name\
    #                 if not only_trainable or ann_graph['synapses'][name]['trainable']]) for param in ['A', 'B', 'C', 'D']])
    #     return (weights - min_val) / (max_val - min_val)
    

    # @SET("learning_rule:params")
    # def set_params(self, conn_name, ann_graph, data, min_val=0., max_val=1.,):
    #     """
    #     """
    #     #* rescale genotype segment to weight range
    #     data = min_val + data * (max_val - min_val)
    #     if conn_name == 'all':
    #         for i, param in enumerate(['A', 'B', 'C', 'D']):
    #             param_data = data[i * len(data) // 4 : (i + 1) * len(data) // 4]
    #             for val, syn in zip(param_data, filter(lambda x: x['trainable'], ann_graph['synapses'].values())):
    #                 syn['learning_rule'][param] = val
    #         return ann_graph
    #     else:
    #         raise NotImplementedError #!!!
   
    # @INIT("learning_rule:params")
    # def init_params(self, conn_name, ann_graph, min_val=0., max_val=1., only_trainable=True):
    #     """
    #     """
    #     params_len = self.len_params(conn_name, ann_graph)
    #     random_params = 0.5 + np.random.randn(params_len) * 0.2
    #     random_params = np.clip(random_params, a_min=0, a_max=1)
    #     return self.set_params(conn_name, ann_graph, random_params, min_val=min_val, max_val=max_val)

    # @LEN("learning_rule:params")
    # def len_params(self, conn_name, ann_graph, only_trainable=True):
    #     """
    #     """
    #     return self.get_params(conn_name, ann_graph, only_trainable=True).shape[0]




class BufferedHebb(GeneralizedHebbian):
    def __init__(self, *args, **kwargs):
        super(BufferedHebb, self).__init__(*args, **kwargs)
        self.buffer_len = 30
        self.gamma = 0.9
        self.buffer = {'in' : deque([]), 'activ': deque([]), 'R': deque([])}
        self.t = 0

    def step(self, inputs, activities, reward=None):
        weight_update = 0.0
        if self.t >= self.buffer_len:
            cumm_reward = np.sum([rew*self.gamma**k for k, rew in enumerate(self.buffer['R'])])
            weight_update = super().step(self.buffer['in'][0].copy(), self.buffer['activ'][0].copy(), reward=cumm_reward)
        self.buffer['in'].append(inputs)
        self.buffer['activ'].append(activities)
        self.buffer['R'].append(reward)
        if self.t >= self.buffer_len:
            self.buffer['in'].popleft()
            self.buffer['activ'].popleft()
            self.buffer['R'].popleft()
        self.t += 1
        return weight_update

    def reset(self):
        self.t = 0
        self.buffer = {'in' : deque([]), 'activ': deque([]), 'R': deque([])}

# class HebbA2C(GeneralizedABCDHebbian):
#     def __init__(self, *args, **kwargs):
#         super(HebbA2C, self).__init__(*args, **kwargs)
#         self.critic_lr = 1e-3
#         self.gamma = 0.9
#         self.prev_state = None
#         self.prev_action = None
#         self.critic_weights = None

    
#     def step(self, outputs, state, actions, reward=None):
#         #* Compute value estimation
#         st_ac_vec = np.r_[state, actions]
#         estim_value = self.critic_weights.dot(st_ac_vec)
#         self.super().step(reward=estim_value)

#         # #* Update weights of critic
#         # td_err = reward + self.gamma * estim_value
#         # self.critic_weights += self.critic_lr * 

#     def reset(self):
#         pass
