import string
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mereli.controllers import RobotController, BasicObstacleAvoider
from mereli.register import controller_registry
from mereli.utils import compute_angle, softmax, sigmoid
from mereli.globals import global_states 
from mereli.controllers.find_sensor_pattern import PatternDetector



CONSTANTS = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'r', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z', 'ch', 'xh', 'zh', 'wh']
VOWELS = ['a', 'e', 'i', 'o', 'u']

def create_random_word(num_syllables=3, uppercase=False):
    new_word = ''
    for i in range(num_syllables):
        ci = np.random.choice(len(CONSTANTS))
        vi = np.random.choice(len(VOWELS))
        syllable = CONSTANTS[ci] + VOWELS[vi]
        new_word += syllable
    if uppercase:
        new_word.upper()
    return new_word


# class SemioticSymbol:
#     def __init__(self):
#         self._type = None # Elementary or composed
#         self.word = None # Three syllable word
#         self.meaning = None # real vector

class Lexicon:
    def __init__(self, dim=3, h=4, w=4):
        self.dim = dim
        self.h = h
        self.w = w
        self.short_term_memory = 100
        self.forget_rate = 1e-3 
        self.learn_rate = 1e-2

        self.thoughts = np.empty((0, self.dim)) 
        self.words = []
        self.word_conf = {}
        self.visible_words = np.array([]) 
        self.word_traces = np.empty((0)).astype(float) 
        self.word_creation_probs = np.empty((0,0)).astype(float) 
        self.word_tree = np.empty((0,0)) 

        # MAPPINGS
        self.patt_word_dict = {} # Maps pattern indices to word indices
        self.word_thought_dict = {}

        # For plotting and debugging only
        self.nx_tree = None
        if True: #global_states.DEBUG:
            self.nx_tree = nx.DiGraph()

    @property
    def elementary_words(self):
        ewords = []
        inv_word_mean_dict = {v : k for k, v in self.word_thought_dict.items()}
        for i in range(len(self.nx_tree.nodes)):
            node = self.nx_tree.nodes[i+1]
            if node['layer'] == 0:
               ewords.append(inv_word_mean_dict[i]) 
        return ewords
            
    @property
    def composed_words(self):
        cwords = []
        inv_word_mean_dict = {v : k for k, v in self.word_thought_dict.items()}
        for i in range(len(self.nx_tree.nodes)):
            node = self.nx_tree.nodes[i+1]
            if node['layer'] > 0:
               cwords.append(inv_word_mean_dict[i]) 
        return cwords

    @property
    def assoc_list(self):
        alist = []
        for word in self.words:
            parents = self.parents(word)
            if parents[0] is not None and  parents[1] is not None:
                alist.append((parents[0], parents[1], word))
        return alist

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
        creation_idxs = np.where(self.word_creation_probs > 0.4)

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
        words_vector[[*self.patt_word_dict.values()]] = self.thoughts[[*self.patt_word_dict.values()]].copy()

        # for pidx, widx in self.patt_word_dict.items():
        #     words_vector[widx] = self.thoughts[widx]
        prev_vector = np.zeros(self.n_words) 
        prev_wvector = words_vector.copy() 
        i = 1
        while any(vector != prev_vector):
            prev_vector = vector.copy()
            prev_wvector = words_vector.copy() 
            # input to the graph always active 
            vector[input_indices] = 1
            words_vector[[*self.patt_word_dict.values()]] = self.thoughts[[*self.patt_word_dict.values()]].copy()
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
                            self.set_thought(j, new_word)
                        elif j not in input_indices:
                            new_word = words_vector[j]
                            self.set_thought(j, new_word)
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
   
    
    def add_thought(self, new_thought, patt_idx=None):
        """ Adds a new specific word to the lexicon """
        self.thoughts = np.vstack((self.thoughts, new_thought))
        self.word_traces = np.pad(self.word_traces, (0,1))
        self.word_tree = np.pad(self.word_tree, (0,1))
        self.word_creation_probs = np.pad(self.word_creation_probs, (0,1))
        if self.nx_tree is not None:
            self.nx_tree.add_node(len(self.thoughts), layer=0, lab=string.ascii_uppercase[len(self.patt_word_dict) - 1])

    def add_word(self, new_word):
        self.word_conf[new_word] = 0.0
        self.words.append(new_word)

    def link_semiotics(self, word, thought_idx, pattern_idx=None):
        if pattern_idx is not None:
            self.patt_word_dict[pattern_idx] = thought_idx 
        self.word_thought_dict[word] = thought_idx

    def create_word(self, use_word=None, patt_idx=None): 
        """ Creates a random word and adds it to the lexicon """
        if use_word is None:
            new_word = create_random_word()
            done = False
            # Check word not in lexicon
            while not done:
                if not new_word in self.words:
                    self.add_word(new_word)
                    done = True
                else:
                    new_word = create_random_word()
        else:
            new_word = use_word
            if new_word in self.words:
                print('CREATING A WORD THAT ALREADY EXISTS')
            self.add_word(new_word)
        new_thought = np.random.uniform(-2, 2, size=self.dim)
        # Check thought does not overlap
        done = False
        while not done:
            if not self.word_exists(new_thought, eps=0.1):
                self.add_thought(new_thought, patt_idx=patt_idx)
                done = True 
            else:
                new_word = np.random.uniform(-2, 2, size=self.dim)
        self.link_semiotics(new_word, len(self.thoughts) - 1, patt_idx)

    
    def create_relation(self, w1, w2, word=None):
        auxvar = np.zeros(self.word_tree.shape[0])
        auxvar[[w1,w2]] = 1
        # Check if there is a word already coming from w1 and w2 (check duplicates) 
        if any(np.abs(self.word_tree - auxvar).sum(1) == 0):
            if word is not None: # Change name of word
                idx =np.where(np.abs(self.word_tree - auxvar).sum(1) == 0)[0][0]
                self.set_word(idx, word, is_pattern=False)
            return
        # New category p1 + p2
        # State is no longer random, but rather s_p1 + s_p2
        new_thought = self.thoughts[w1] + self.thoughts[w2]
        if self.word_exists(new_thought):
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
        if word is None:
            new_word = create_random_word()
            done = False
            while not done: 
                if not new_word in self.words:
                    done = True
                    self.add_word(new_word)
                else:
                    new_word = create_random_word()
        else:
            new_word = word
            self.add_word(new_word)
        self.add_thought(new_thought.copy(), patt_idx=None)
        self.link_semiotics(new_word, len(self.thoughts) - 1, pattern_idx=None)

        # Update compositionality matrix
        self.word_tree[-1, [w1, w2]] = 1
        if self.nx_tree is not None:
            lay = max(self.nx_tree.nodes[w1+1]['layer'], self.nx_tree.nodes[w2+1]['layer']) + 1
            labels = nx.get_node_attributes(self.nx_tree, 'lab')
            lab =  ''.join(labels[w1+1].split('+')) + '+'  + ''.join(labels[w2+1].split('+'))
            self.nx_tree.add_node(len(self.word_tree), layer=lay, lab=lab, word=new_thought.round(3).tolist())
            self.nx_tree.add_edge(w1+1, len(self.word_tree))
            self.nx_tree.add_edge(w2+1, len(self.word_tree))
            
    def parents(self, word):
        thought_idx = self.word_thought_dict[word]
        tree_inputs = self.word_tree[thought_idx]
        if np.sum(tree_inputs) == 0:
            return (None, None)
        parent_ids = np.where(tree_inputs == 1)[0]
        assert len(parent_ids) == 2 
        return (self.words[parent_ids[0]], self.words[parent_ids[1]])

    def closest_word(self, word):
        distances = np.array([self.word_distance(word, ww) for ww in self.thoughts])
        sorted_words = np.argsort(distances)
        clst_word = self.thoughts[sorted_words[0]]
        word_dist = distances[sorted_words[0]]
        return clst_word, word_dist 
    
    def is_same_word(self, w1, w2, eps=1e-2):
        return self.word_distance(w1, w2) < eps

    def word_exists(self, word, eps=1e-2):
        return any([self.word_distance(word, ww) < eps for ww in self.thoughts])

    def reset_visible_words(self):
        # must be called at every time instant
        self.visible_words = np.array([]).astype(int)

    def add_visible_word(self, idx, is_pattern=True):
        # must be called at every time instant
        ii = idx if not is_pattern else self.patt_word_dict[idx]
        self.visible_words = np.append(self.visible_words, ii)
     
    def attach_particle(self, particle):
        self.particle = particle

    def set_thought(self, idx, new_word, is_pattern=False):
        ii = idx if not is_pattern else self.patt_word_dict[idx]
        self.thoughts[ii] = new_word.copy()
        # self.particle.landmarks[ii].state = new_word.copy()

    def set_word(self, idx, new_word, is_pattern=False):
        ii = idx if not is_pattern else self.patt_word_dict[idx]
        # if ii in self.word_thought_dict.values():# and self.word_thought_dict[ii] != new_word:
        #     print('YEPA')
        #     __import__('pdb').set_trace()
        prev_word = self.words[ii]
        if prev_word == new_word: 
            if self.word_thought_dict[prev_word] != ii:
                __import__('pdb').set_trace()
            return

        # if self.word_thought_dict[prev_word] != ii:
        #     __import__('pdb').set_trace()
        self.word_thought_dict[new_word] =ii# self.word_thought_dict.pop(prev_word)
        del self.word_thought_dict[prev_word]
        self.words[ii] = new_word

    def get_word(self, idx, is_pattern=True):
        ii = idx if not is_pattern else self.patt_word_dict[idx]
        return self.words[ii]
        # return self.particle.landmarks[ii].state.copy()

    def random_word(self):
        try:
            idx = np.random.choice(len(self.words))
            return self.words[idx], idx
        except:
            __import__('pdb').set_trace()
    
    def word_distance(self, w1, w2):
        return np.linalg.norm(w1 - w2)

    @property
    def n_words(self):
        return self.word_traces.shape[0] 
    
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




@controller_registry(name='build_lexicon_B') 
class BuildLexiconControllerB(RobotController):
    def __init__(self, *args, **kwargs):
        super(BuildLexiconControllerB, self).__init__(*args, **kwargs)
        self.obstacle_avoider = BasicObstacleAvoider() 
        self.lexicon = Lexicon()
        self.pattern_detector = PatternDetector(7)
        self.sharing = None

    def transmit_lexicon(self, prob=0.9, only_patterns=False):
        if np.random.random() <= prob and self.lexicon.n_words > 1: 
            word, widx = self.lexicon.random_word()
            if widx in self.lexicon.word_patt_dict:
                # Is pattern
                pidx = self.lexicon.word_patt_dict[widx]
                pattern = self.pattern_detector.pattern_mat[pidx].copy() 
                self.sharing = {'pattern' : pattern, 'word' : word, 'parents' : None}
            else:
                parents = self.lexicon.parents(word) 
                self.sharing = {'pattern' : None, 'word' : word, 'parents' : parents}
        else:
            self.sharing = None

    def receive_lexicon(self):
        neighbors = self.robot.neighbors
        if len(neighbors) == 0:
            return
        # neigh_idx = np.random.choice(len(neighbors))
        for neigh in neighbors:
            if neigh.controller.sharing is not None:
                pattern= neigh.controller.sharing['pattern']
                word = neigh.controller.sharing['word']
                parents = neigh.controller.sharing['parents']
                pattern_detected = False
                pattern_idx = None
                if pattern is not None: # None means abstract concept word
                    pattern_detected, pattern_idx = self.pattern_detector.step(pattern.copy())
                    if pattern_detected:
                        # self.lexicon.update_word(pattern_idx, word, is_pattern=True) 
                        self.lexicon.set_word(pattern_idx, word, is_pattern=True)
                    else:
                        if word in self.lexicon.words:
                            __import__('pdb').set_trace()
                        self.lexicon.create_word(use_word=word, patt_idx=pattern_idx)
                        # self.lexicon.set_word(word, pattern_idx=pattern_idx)
                else:
                    if self.lexicon.n_words < 2:
                        continue
                    if parents[0] in self.lexicon.words and parents[1] in self.lexicon.words:
                        widx1 = self.lexicon.word_thought_dict[parents[0]]
                        widx2 = self.lexicon.word_thought_dict[parents[1]]
                        if word in self.lexicon.words:
                            w_tr = self.lexicon.word_tree[self.lexicon.word_thought_dict[word]]
                            if w_tr[widx1] == 0 or w_tr[widx2] == 0:
                                __import__('pdb').set_trace()
                        self.lexicon.create_relation(widx1, widx2, word=word)
    

    def get_color(self, color, distorted_perception=False):
        num_color = None
        if isinstance(color, str):
            num_color= np.array({'red' : [0.8,0,0], 'green' : [0,0.8,0], 'blue' : [0,0,0.8]}.get(color))
            if distorted_perception: 
                num_color = np.array({'red' : [0.5, 0, 0.5], 'green' : [0,0.8,0], 'blue' : [0,0,0.8]}.get(color))
            num_color = np.clip(np.random.normal(num_color, 0.05), a_min=0, a_max=1)
        else:
            num_color = np.array(color)
        return num_color

    def get_shape(self, shape, perspective_noise=False, distorted_perception=False):
        if perspective_noise:
            rnd_aux = np.random.random() < 0.5
            num_shape = {'cube' : (np.array([0,1,0]), np.array([0.25,0,0.75]))[rnd_aux], 
                         'ball' : np.array([1,0,0]), 'pyramid' : np.array([0,0,1])}.get(shape)
        elif distorted_perception: 
            num_shape = {'cube' : np.array([0,0,1]),  'ball' : np.array([1,0,0]), 'pyramid' : np.array([0,0.5,0.5])}.get(shape)
        else:
            num_shape = {'ball' : np.array([1,0,0]), 'cube' : np.array([0,1,0]), 'pyramid' : np.array([0,0,1])}.get(shape)
        num_shape = np.clip(np.random.normal(num_shape, 0.05), a_min=0, a_max=1)
        return num_shape

    def step(self, *args, **kwargs):
        obj = self.get_sensor_reading('object_sensor')
        wheels  = self.get_sensor_reading('encoder').round(3)
        # ground = self.get_sensor_reading('ground_sensor')
        # if self.pattern_traces is not None:

        #     self.pattern_traces = self.pattern_traces - (1 / 100) * self.pattern_traces
        self.lexicon.reset_visible_words()
        if obj is not None:
            #### --- PERCEPTION --- #### 
            size = np.round(obj['size'],2)
            MIN_SIZE, MAX_SIZE = 0.05, 0.5 
            size = (size - MIN_SIZE) / (MAX_SIZE - MIN_SIZE)
            color = self.get_color(obj['color'])#, distorted_perception=self.robot.id in [22,23,24])
            shape = self.get_shape(obj['shape'], perspective_noise=True)#, distorted_perception=self.robot.id in [25,26])
            score = {'green' : 1, 'red' : -1}.get(obj['color'], 0)
            score_val = np.clip(np.random.normal(score,0.05), a_min=-1, a_max=1)
               
            # phi = np.r_[color, size, shape]
            stimuli = []
            variables = [color, shape, score_val]
            for i, var in enumerate(variables):
                stim = np.hstack([(v, np.zeros_like(v))[i!=j] for j, v in enumerate(variables)])
                stimuli.append(stim)

            #### --- DISCRIMINATION --- #### 
            for phi in stimuli:
                pattern_detected, pattern = self.pattern_detector.step(phi)
                if not pattern_detected: 
                    self.lexicon.create_word(patt_idx=pattern)
                self.lexicon.add_visible_word(pattern, is_pattern=True)
        # COMMUNICATION
        self.lexicon.step()
        self.receive_lexicon()
        self.transmit_lexicon()
        if False and  self.t > 17989:
        # if True and  self.t > 5989:
            __import__('pdb').set_trace()
            pp = self.lexicon.parents(self.lexicon.words[-1])
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
                labels = {i+1 : names[i+1] + " : ({:.2f}, {:.2f}, {:.2f})".format(self.lexicon.thoughts[i][0], self.lexicon.thoughts[i][1],
                                        self.lexicon.thoughts[i][2]) for i in range(self.lexicon.n_words)} 
                labels = {i+1 : names[i+1] + ' : ' + self.lexicon.words[i] for i in range(self.lexicon.n_words)} 
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
    


