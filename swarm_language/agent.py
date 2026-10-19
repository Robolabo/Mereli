""" A language-game agent: perception, discrimination, compositionality and the naming game
(paper Secs. 3.3-3.7 and Fig. 2). This is the piece to plug into any simulator.

Messages exchanged by agents are plain dicts:

* elementary word: ``{'pattern': prototype (np.ndarray), 'word': str, 'parents': None}``
* composed word:   ``{'pattern': None, 'word': str, 'parents': (parent_word_1, parent_word_2)}``
"""
import numpy as np
from .discrimination import PatternDetector
from .lexicon import Lexicon
from .perception import FEATURE_DIM, perceive_object


class LanguageAgent:
    """ Semiotic agent of the swarm.

    Usage, once per control step and agent (agents are stepped sequentially)::

        agent.step(observed_object, [neighbor.message for neighbor in neighbors])

    where ``observed_object`` is a dict with ``'color'`` and ``'shape'`` keys (or None when no
    object is perceived). After the call, ``agent.message`` holds what the agent broadcasts
    (or None); neighbors stepped later in the same cycle already hear it.

    :param float speak_prob: probability of broadcasting a word at each step.
    :param bool distorted_color: faulty color sensor (robustness experiment R2).
    :param bool distorted_shape: faulty shape perception.
    :param bool perspective_noise: perspective-dependent shape perception (experiment R1).
    :param dict lexicon_params: keyword arguments of :class:`Lexicon`.
    :param dict discrimination_params: keyword arguments of :class:`PatternDetector`.
    """
    def __init__(self, speak_prob=0.9, distorted_color=False, distorted_shape=False,
                 perspective_noise=False, lexicon_params=None, discrimination_params=None):
        self.speak_prob = speak_prob
        self.distorted_color = distorted_color
        self.distorted_shape = distorted_shape
        self.perspective_noise = perspective_noise
        self.lexicon = Lexicon(**(lexicon_params or {}))
        self.pattern_detector = PatternDetector(FEATURE_DIM, **(discrimination_params or {}))
        self.message = None

    def step(self, observed_object, neighbor_messages):
        """ Runs one full control step: perceive, learn compositions, hear, speak.

        :param dict observed_object: perceived object, or None.
        :param list neighbor_messages: current messages of the neighbors in communication
            range, in a fixed order. ``None`` entries are ignored.
        """
        self.perceive(observed_object)
        self.lexicon.step()
        self.hear(neighbor_messages)
        self.speak()

    def perceive(self, observed_object):
        """ Sensing and discrimination: activates the elementary meanings of the observed
        object, creating new prototypes and symbols when needed. """
        self.lexicon.reset_visible_words()
        if observed_object is None:
            return
        stimuli = perceive_object(observed_object, distorted_color=self.distorted_color,
                                  distorted_shape=self.distorted_shape,
                                  perspective_noise=self.perspective_noise)
        for phi in stimuli:
            pattern_detected, pattern = self.pattern_detector.step(phi)
            if not pattern_detected:
                self.lexicon.create_word(patt_idx=pattern)
            self.lexicon.add_visible_word(pattern, is_pattern=True)

    def speak(self):
        """ Speaker role: with probability ``speak_prob`` broadcasts a random word of the
        lexicon (Fig. 2B). Sets ``self.message``. """
        if np.random.random() <= self.speak_prob and self.lexicon.n_words > 1:
            word, widx = self.lexicon.random_word()
            if widx in self.lexicon.meaning_to_prototype:
                pidx = self.lexicon.meaning_to_prototype[widx]
                pattern = self.pattern_detector.pattern_mat[pidx].copy()
                self.message = {'pattern': pattern, 'word': word, 'parents': None}
            else:
                self.message = {'pattern': None, 'word': word,
                                'parents': self.lexicon.parents(word)}
        else:
            self.message = None

    def hear(self, messages):
        """ Hearer role (Fig. 2C): updates the lexicon with the received messages.

        Elementary words use the received prototype for joint attention: the hearer either
        renames the symbol of its matching prototype or creates a new symbol. Composed words
        are learned only if both parent words are known.

        Note: as in the published implementation, processing of the remaining messages stops
        when a known composed word arrives with parents that differ from the hearer's own.
        """
        for message in messages:
            if message is None:
                continue
            pattern = message['pattern']
            word = message['word']
            parents = message['parents']
            if pattern is not None:
                pattern_detected, pattern_idx = self.pattern_detector.step(pattern.copy())
                if pattern_detected:
                    self.lexicon.set_word(pattern_idx, word, is_pattern=True)
                else:
                    self.lexicon.create_word(use_word=word, patt_idx=pattern_idx)
            else:
                if self.lexicon.n_words < 2:
                    continue
                if parents[0] in self.lexicon.words and parents[1] in self.lexicon.words:
                    widx1 = self.lexicon.word_to_meaning[parents[0]]
                    widx2 = self.lexicon.word_to_meaning[parents[1]]
                    if word in self.lexicon.words:
                        own_parents = self.lexicon.man[self.lexicon.word_to_meaning[word]]
                        if own_parents[widx1] == 0 or own_parents[widx2] == 0:
                            break
                    self.lexicon.create_relation(widx1, widx2, word=word)
