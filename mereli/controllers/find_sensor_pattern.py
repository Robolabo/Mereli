import string
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mereli.communication import LandmarkMolecule
from mereli.controllers import RobotController, BasicObstacleAvoider
from mereli.register import controller_registry
from mereli.utils import compute_angle, softmax, sigmoid
from mereli.globals import global_states 
from swarm_language.discrimination import PatternDetector  # shared with build_lexicon_B


class Lexicon:
    def __init__(self, dim=3, h=4, w=4):
        self.dim = dim
        self.h = h
        self.w = w
        self.short_term_memory = 100
        self.forget_rate = 1e-3 
        self.learn_rate = 1e-2
        # self.particle = None 

        self.words = np.empty((0, self.dim)) 
        self.visible_words = np.array([]) 
        self.word_traces = np.empty((0)).astype(float) 
        self.word_creation_probs = np.empty((0,0)).astype(float) 
        self.word_tree = np.empty((0,0)) 
        self.patt_word_dict = {} # Maps pattern indices to word indices

        # For plotting and debugging only
        self.nx_tree = None
        if True: #global_states.DEBUG:
            self.nx_tree = nx.DiGraph()

    def step(self):
        if len(self.word_traces) == 0:
            return
        self.propagate_tree()
        self.word_traces = self.word_traces - (1 / self.short_term_memory) * self.word_traces
        self.word_traces[self.visible_words] = 1.0
        aux_vector = np.zeros(len(self.word_traces))
        aux_vector[self.visible_words] = 1.0
        self.word_creation_probs += self.learn_rate * np.outer(aux_vector, aux_vector)
        delta_probs = self.forget_rate * (-self.word_creation_probs)

        self.word_creation_probs = self.word_creation_probs + delta_probs 
        # random_mat = np.random.random(self.word_tree.shape
        # creation_idxs = np.where(random_mat < self.word_creation_probs)
        creation_idxs = np.where(self.word_creation_probs > 0.3)

        # DEBUG
        # for i, w1 in enumerate(self.words):
        #     for j, w2 in enumerate(self.words):
        #         if i != j and  self.is_same_word(w1, w2):
        #             __import__('pdb').set_trace()
        # END DEBUG
        if len(creation_idxs[0]) > 0: 
            for w1, w2 in zip(*creation_idxs):
                if w1 != w2:
                    # print(f'Concept {w1}-{w2} created')
                    self.create_relation(w1,w2)
                    
                    

    def propagate_tree(self):
        if self.word_tree is None:
            return
        # Binary vector with ones where words are currently present/visible
        input_indices = self.visible_words.astype(int).copy()
        vector = np.zeros(self.word_tree.shape[0])
        vector[input_indices] = 1
        
        # Only sensory pattern words as input words to the tree 
        words_vector = np.zeros((self.word_tree.shape[0], self.dim)).astype(float)
        words_vector[[*self.patt_word_dict.values()]] = self.words[[*self.patt_word_dict.values()]].copy()

        # for pidx, widx in self.patt_word_dict.items():
        #     words_vector[widx] = self.words[widx]
        prev_vector = np.zeros(len(self.words)) 
        prev_wvector = words_vector.copy() 
        i = 1
        while any(vector != prev_vector):
            prev_vector = vector.copy()
            prev_wvector = words_vector.copy() 
            # input to the graph always active 
            vector[input_indices] = 1
            words_vector[[*self.patt_word_dict.values()]] = self.words[[*self.patt_word_dict.values()]].copy()
            vector = self.word_tree.dot(vector)
            # if i > 2:
            #     __import__('pdb').set_trace()
            words_vector = self.word_tree.dot(words_vector)
            vector = np.where(vector == 2, 1, 0)
            
            # Update non-sensory pattern words with computed word_vectors 
            if any(vector == 1):
                for j in np.where(vector == 1)[0]:
                    if prev_vector[j] == 0:
                        if j not in self.visible_words:
                            self.visible_words = np.append(self.visible_words, j)
                            new_word = words_vector[j]
                            self.set_word(j, new_word)
                        elif j not in input_indices:
                            new_word = words_vector[j]
                            self.set_word(j, new_word)

            vector[input_indices] = 1
            i += 1
    

    def backpropagate_tree(self, word_idx):
        vector = np.zeros(self.word_tree.shape[0])
        prev_vector = np.zeros(self.word_tree.shape[0])
        vector[word_idx] = 1
        while any(vector != prev_vector):
            prev_vector = vector.copy()
            vector[word_idx] = 1
            vector = self.word_tree.T.dot(vector)
        vector[word_idx] = 1
        return vector#np.where(vector == 1)[0]
   
    
    def add_word(self, new_word, patt_idx=None):
        """ Adds a new specific word to the lexicon """
        self.words = np.vstack((self.words, new_word))
        self.word_traces = np.pad(self.word_traces, (0,1))
        self.word_tree = np.pad(self.word_tree, (0,1))
        self.word_creation_probs = np.pad(self.word_creation_probs, (0,1))
        if patt_idx is not None:
            self.patt_word_dict[patt_idx] = self.words.shape[0] - 1
        if self.nx_tree is not None:
            self.nx_tree.add_node(len(self.words), layer=0, lab=string.ascii_uppercase[len(self.patt_word_dict) - 1])

    def create_word(self, patt_idx=None): 
        """ Creates a random word and adds it to the lexicon """
        new_word = np.random.uniform(-2, 2, size=self.dim)
        done = False
        while not done:
            if not self.word_exists(new_word, eps=0.1):
                self.add_word(new_word, patt_idx=patt_idx)
                done = True
            else:
                new_word = np.random.uniform(-2, 2, size=self.dim)

    def update_word(self, index, new_word, is_pattern=True):
        """ Updates existing word as a combination between the learnt and the new.  """
        word = self.get_word(index, is_pattern=is_pattern)
        lr = 0.1#0.05
        self.set_word(index, word + lr * (new_word - word), is_pattern=is_pattern)

        self.visible_words = np.r_[*self.patt_word_dict.values()]
        self.propagate_tree()
        self.reset_visible_words()
        # END DEBUG
        """ ERROR IS HERE """

        # if self.n_words > 1:
        #     word = self.get_word(index, is_pattern=True)
        #     other_words = self.words[np.arange(self.n_words) != idx] 
        #     clst_word = np.argmin([self.word_distance(word, other) for other in other_words])
        #     if self.word_distance(word, other_words[clst_word]) < 1:
        #         self.set_lmark(idx, word - 0.1 * (other_words[clst_word] - word))
     
    def diffuse_words(self):
        """ Modify words so that their positions are not too close """
        if len(self.words) < 2: 
            return
        for i, word in enumerate(self.words):
            other_words = self.words[np.arange(self.n_words) != i] 
            clst_word = np.argmin([self.word_distance(word, other) for other in other_words])
            if self.word_distance(word, other_words[clst_word]) < 0.4 and i in self.patt_word_dict.values():
                self.set_word(i, word - 0.1 * (other_words[clst_word] - word))
    
    def create_relation(self, w1, w2):
        auxvar = np.zeros(self.word_tree.shape[0])
        auxvar[[w1,w2]] = 1
        # Check if there is a word already coming from w1 and w2 (check duplicates) 
        if any(np.abs(self.word_tree - auxvar).sum(1) == 0):
            return
        # New category p1 + p2
        # State is no longer random, but rather s_p1 + s_p2
        new_word = self.words[w1] + self.words[w2]
        if self.word_exists(new_word):
            return
        col1 = self.word_tree[w1]
        col2 = self.word_tree[w2]
        # common previous concept
        if col1[w2] == 1 or col2[w1] == 1 or np.logical_and(col1 == 1,col1 ==col2).any():
            return
        origins_w1 = self.backpropagate_tree(w1)
        origins_w2 = self.backpropagate_tree(w2)
        
        if any(origins_w1 + origins_w2 == 2):
            return    

        # All conditions met => create new word
        self.add_word(new_word.copy(), patt_idx=None)

        # Update compositionality matrix
        self.word_tree[-1, [w1, w2]] = 1
        if self.nx_tree is not None:
            lay = max(self.nx_tree.nodes[w1+1]['layer'], self.nx_tree.nodes[w2+1]['layer']) + 1
            labels = nx.get_node_attributes(self.nx_tree, 'lab')
            lab =  ''.join(labels[w1+1].split('+')) + '+'  + ''.join(labels[w2+1].split('+'))
            self.nx_tree.add_node(len(self.word_tree), layer=lay, lab=lab, word=new_word.round(3).tolist())
            self.nx_tree.add_edge(w1+1, len(self.word_tree))
            self.nx_tree.add_edge(w2+1, len(self.word_tree))
            
    
    def closest_word(self, word):
        distances = np.array([self.word_distance(word, ww) for ww in self.words])
        sorted_words = np.argsort(distances)
        clst_word = self.words[sorted_words[0]]
        word_dist = distances[sorted_words[0]]
        return clst_word, word_dist 
    
    def is_same_word(self, w1, w2, eps=1e-2):
        return self.word_distance(w1, w2) < eps

    def word_exists(self, word, eps=1e-2):
        return any([self.word_distance(word, ww) < eps for ww in self.words])

    def reset_visible_words(self):
        # must be called at every time instant
        self.visible_words = np.array([]).astype(int)

    def add_visible_word(self, idx, is_pattern=True):
        # must be called at every time instant
        ii = idx if not is_pattern else self.patt_word_dict[idx]
        self.visible_words = np.append(self.visible_words, ii)
     
    def attach_particle(self, particle):
        self.particle = particle

    def set_word(self, idx, new_word, is_pattern=False):
        ii = idx if not is_pattern else self.patt_word_dict[idx]
        self.words[ii] = new_word.copy()
        # self.particle.landmarks[ii].state = new_word.copy()

    def get_word(self, idx, is_pattern=True):
        ii = idx if not is_pattern else self.patt_word_dict[idx]
        return self.words[ii].copy()
        # return self.particle.landmarks[ii].state.copy()

    def random_word(self):
        idx = np.random.choice(self.n_words)
        return self.words[idx].copy(), idx
    
    def word_distance(self, w1, w2):
        return np.linalg.norm(w1 - w2)

    @property
    def n_words(self):
        return self.word_traces.shape[0] 
    
    # @property
    # def words(self):
    #     if self.n_words == 0:
    #         return np.array([])
    #     return np.stack([self.get_word(i, is_pattern=False) for i in range(self.n_words)])

    @property
    def word_patt_dict(self):
        return {v: k for k, v in self.patt_word_dict.items()}

            
    def plot_concept_tree(self):
        import networkx as nx
        import matplotlib.pyplot as plt
        from networkx.drawing.nx_pydot import graphviz_layout
        G = nx.from_numpy_array(self.pattern_relation_mat.T, create_using=nx.DiGraph())
        attrs = {i : {'type' :( 'pattern', 'concept')[i < len(self.pattern_mat)] } for i in range(len(self.pattern_relation_mat))}   
        nx.set_node_attributes(G, attrs)
        # pos = nx.nx_agraph.graphviz_layout(G, prog="twopi")
        pos = nx.spring_layout(G, seed=3068) 
        nx.draw(G, pos=pos, with_labels=True)

        plt.show()




@controller_registry(name='build_lexicon') 
class BuildLexiconController(RobotController):
    def __init__(self, *args, **kwargs):
        super(BuildLexiconController, self).__init__(*args, **kwargs)
        self.obstacle_avoider = BasicObstacleAvoider() 
        self.lexicon = Lexicon()
        self.pattern_detector = PatternDetector(7)
        self.sharing = None

    def share_lexicon(self, prob=1, only_patterns=False):
        if np.random.random() <= prob and self.lexicon.n_words > 1: 
            if only_patterns:
                pattern, pidx = self.pattern_detector.random_pattern()
                word = self.lexicon.get_word(pidx, is_pattern=True)
                self.sharing = {'pattern' : pattern, 'word' : word}
            else:
                word, widx = self.lexicon.random_word()
                if widx in self.lexicon.word_patt_dict:
                    # Is pattern
                    pidx = self.lexicon.word_patt_dict[widx]
                    pattern = self.pattern_detector.pattern_mat[pidx].copy() 
                    self.sharing = {'pattern' : pattern, 'word' : word}
                else:
                    self.sharing = {'pattern' : None, 'word' : word}
        else:
            self.sharing = None

    def update_lexicon(self):
        # if not (self.pattern_detector.n_patterns > 1 and self.lexicon.n_words > 1):
        #     return
        neighbors = self.robot.neighbors
        for neigh in neighbors:
            if neigh.controller.sharing is not None:
                pattern= neigh.controller.sharing['pattern']
                word = neigh.controller.sharing['word'].copy()
                pattern_detected = False
                pattern_idx = None
                if pattern is not None: # None means abstract concept word
                    pattern_detected, pattern_idx = self.pattern_detector.step(pattern.copy())
                    if pattern_detected:
                        self.lexicon.update_word(pattern_idx, word, is_pattern=True) 
                    else:
                        self.lexicon.create_word(patt_idx=pattern_idx)
                else:
                    if self.lexicon.n_words < 2:
                        return
                    for w_idx1, w_idx2 in itertools.combinations(range(self.lexicon.n_words),2):
                        w1 = self.lexicon.words[w_idx1].copy()
                        w2 = self.lexicon.words[w_idx2].copy()
                        if self.lexicon.is_same_word(w1 + w2, word):
                            self.lexicon.create_relation(w_idx1,w_idx2)

                    

    def step(self, *args, **kwargs):
        obj = self.get_sensor_reading('object_sensor')
        wheels  = self.get_sensor_reading('encoder').round(3)
        # ground = self.get_sensor_reading('ground_sensor')
        # if self.pattern_traces is not None:

        #     self.pattern_traces = self.pattern_traces - (1 / 100) * self.pattern_traces
        self.lexicon.reset_visible_words()
        if obj is not None:
            size = np.round(obj['size'],2)
            MIN_SIZE, MAX_SIZE = 0.05, 0.5 
            size = (size - MIN_SIZE) / (MAX_SIZE - MIN_SIZE)
            # size = np.exp(2*size ) -1  + .1 
            shape = {'ball' : np.array([1,0,0]), 'cube' : np.array([0,1,0]), 'pyramid' : np.array([0,0,1])}.get(obj['shape'])
            shape = np.clip(np.random.normal(shape, 0.05), a_min=0, a_max=1)
            if isinstance(obj['color'], str):
                color = np.array({'red' : [0.8,0,0], 'green' : [0,0.8,0], 'blue' : [0,0,0.8]}.get(obj['color']))
                color = np.clip(np.random.normal(color, 0.1), a_min=0, a_max=1)
            else:
                color = np.array(obj['color'])
            score = {'green' : 1, 'red' : -1}.get(obj['color'], 0)
            score_val = np.clip(np.random.normal(score,0.05), a_min=-1, a_max=1)
               
            # phi = np.r_[color, size, shape]
            stimuli = []
            variables = [color, shape, score_val]
            for i, var in enumerate(variables):
                stim = np.hstack([(v, np.zeros_like(v))[i!=j] for j, v in enumerate(variables)])
                stimuli.append(stim)
            # stimuli = [ np.r_[color, 0, [0,0,0], [0,0]], np.r_[[0,0,0], size, [0,0,0], [0,0]], np.r_[[0,0,0],0,shape, [0,0]]]
            # stimuli.append(np.r_[[0,0,0],0,[0,0,0], wheels])

            for phi in stimuli:
                pattern_detected, pattern = self.pattern_detector.step(phi)
                if not pattern_detected: 
                    self.lexicon.create_word(patt_idx=pattern)
                self.lexicon.add_visible_word(pattern, is_pattern=True)
                # self.update_masses(phi)    
        
        # COMMUNICATION
        self.lexicon.step()
        self.share_lexicon()
        self.update_lexicon()

        if False and self.t > 14989 :
            print(" CULTURAL EVOLUTION OF ROBOT ", self.robot.id)
            print(self.pattern_detector.pattern_mat.round(3))
            print(self.lexicon.word_tree)
            # print(self.pattern_relation_mat.shape)
            if self.lexicon.n_words > 0:
                print(self.lexicon.words)
            # self.plot_concept_tree()
            if self.lexicon.nx_tree is not None:
                G = self.lexicon.nx_tree
                pos = nx.multipartite_layout(G, subset_key="layer")
                # for k, v in pos.items():
                #    pos[k][1] = v[1] + np.random.normal(0, 0.1)  
                # nx.draw(G, pos=pos)
                label_options = {"ec": "k", "fc": "white", "alpha": 0.7}
                nx.draw_networkx_nodes(G, pos=pos, )
                nx.draw_networkx_edges(G, pos=pos, )
                names = nx.get_node_attributes(G, "lab")
                labels = {i+1 : names[i+1] + " : ({:.2f}, {:.2f}, {:.2f})".format(self.lexicon.words[i][0], self.lexicon.words[i][1], self.lexicon.words[i][2])\
                        for i in range(self.lexicon.n_words)} 
                nx.draw_networkx_labels(G, pos, labels=labels, font_size=14, bbox=label_options)
                plt.show()
                __import__('pdb').set_trace()
        # if self.robot.id == 5:
        self.obstacle_avoider.step(*args,**kwargs)

    def lexicon_error(self):
        pass

    def reset(self):
        # self.robot.virtual_particle.mass = 0.01 #! OJJJJO
        self.obstacle_avoider.controller_owner = self.controller_owner 
        self.obstacle_avoider.reset()
        # self.lexicon.attach_particle(self.robot.virtual_particle)
    


