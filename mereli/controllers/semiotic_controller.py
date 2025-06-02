
import string
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mereli.controllers import RobotController, BasicObstacleAvoider
from mereli.register import controller_registry from mereli.utils import compute_angle, softmax, sigmoid
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



class SemioticSymbol:
    def __init__(self, input_space_dim, meaning_space_dim=3):
        self.input_space_dim = input_space_dim
        self.meaning_space_dim = meaning_space_dim 
        self.tag = None
        self._id = None
        self._type = None # Elementary or composed
        self._word = None # Three syllable word
        self._meaning = None # real vector
        self._referent = None # Referent on which it is grounded (not accessible)
        
        self.parent_ids = ()
        self.parent_tags = ()

    @property
    def id(self):
        return self._id

    @property
    def type(self):
        return self._type

    @property
    def word(self):
        return self._word

    @property
    def meaning(self):
        return self._meaning 

    @property
    def referent(self):
        return self._referent

    @id.setter
    def id(self, new_id):
        self._id = new_id

    @type.setter
    def type(self, new_type):
        self._type = new_type



class SemioticSymbolSet:


    def __init__():
        self.symbols = set()

    def add(self, ):
        pass

    def remove(self,):
        pass
    
    def sample(self):
         " Randomly samples a semiotic symbol in the set"
        pass

    def get_by_id(self, id_num):
        pass

    @property
    def words(self):
        return [symbol.word for symbol in self.symbols]

    @property
    def meanings(self):
        return np.r_[symbol.meaning for symbol in self.symbols]

@controller_registry(name='SemioticController') 
class SemioticController(RobotController):
    def __init__(self, *args, **kwargs):
        super(SemioticController, self).__init__(*args, **kwargs)
        self.obstacle_avoider = BasicObstacleAvoider() 
        self.speaker = False # Flag that is active when the robot is transmitting some word 
        self.message = {} # Message transmitted if speaker=true.



        self.lexicon = Lexicon()
        self.pattern_detector = PatternDetector(7)

    def get_color(self, obj, distorted_perception=False):
        color = obj['color']
        num_color = None
        if isinstance(color, str):
            num_color= np.array({'red' : [0.8,0,0], 'green' : [0,0.8,0], 'blue' : [0,0,0.8]}.get(color))
            if distorted_perception: 
                num_color = np.array({'red' : [0.5, 0, 0.5], 'green' : [0,0.8,0], 'blue' : [0,0,0.8]}.get(color))
            num_color = np.clip(np.random.normal(num_color, 0.05), a_min=0, a_max=1)
        else:
            num_color = np.array(color)
        return num_color

    def get_shape(self, obj, perspective_noise=False, distorted_perception=False):
        if perspective_noise:
            rnd_aux = np.random.random() < 0.5
            num_shape = {'cube' : (np.array([0,1,0]), np.array([0.25,0,0.75]))[rnd_aux], 
                         'ball' : np.array([1,0,0]), 'pyramid' : np.array([0,0,1])}.get(obj['shape'])
        elif distorted_perception: 
            num_shape = {'cube' : np.array([0,0,1]),  'ball' : np.array([1,0,0]), 'pyramid' : np.array([0,0.5,0.5])}.get(obj['shape'])
        else:
            num_shape = {'ball' : np.array([1,0,0]), 'cube' : np.array([0,1,0]), 'pyramid' : np.array([0,0,1])}.get(obj['shape'])
        num_shape = np.clip(np.random.normal(num_shape, 0.05), a_min=0, a_max=1)
        return num_shape

    def step(self, *args, **kwargs):
        obj = self.get_sensor_reading('object_sensor')
        wheels  = self.get_sensor_reading('encoder').round(3)
        obj_detected = obj is not None

        if obj_detected:
            #### --- PERCEPTION --- #### 
            color = self.get_color(obj) 
            shape = self.get_shape(obj) 
            score = {'green' : 1, 'red' : -1}.get(obj['color'], 0)
            score_val = np.clip(np.random.normal(score,0.05), a_min=-1, a_max=1)
            
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

        # if self.robot.id == 5:
        self.obstacle_avoider.step(*args,**kwargs)

    def lexicon_error(self):
        pass

    def reset(self):
        # self.robot.virtual_particle.mass = 0.01 #! OJJJJO
        self.obstacle_avoider.controller_owner = self.controller_owner 
        self.obstacle_avoider.reset()
        # self.lexicon.attach_particle(self.robot.virtual_particle)
    




