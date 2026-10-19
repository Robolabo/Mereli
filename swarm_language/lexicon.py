""" Semiotic symbols, meaning space and Meaning Association Network (paper Secs. 3.4-3.6).

Each semiotic symbol ``i`` of an agent is stored by index:

* its **form** (word) is ``words[i]``,
* its private **meaning** is the vector ``meanings[i]`` in the meaning space S,
* its **referent** is implicit: elementary symbols are linked to a prototype through
  ``prototype_to_meaning``; composed symbols inherit the referents of their parents.

Name mapping with the paper:

==========================  ========================================================
Code                        Paper
==========================  ========================================================
``meanings``                meaning vectors ``s_i`` (Sec. 3.4)
``man``                     MAN adjacency ``G``: ``man[a, b] = 1`` iff ``(b, a) in E``
``traces``                  short-term traces ``beta_i`` (Eq. 7)
``assoc_weights``           association weights ``alpha_ij`` (Eq. 8)
``association_threshold``   ``theta_alpha``
``active``                  meanings activated at the current step
==========================  ========================================================
"""
import logging
import string
import numpy as np
import networkx as nx
from .words import create_random_word


def _node_label(idx):
    """ Letter used to label MAN nodes in ``nx_tree`` (plotting only). """
    if -26 <= idx < 26:
        return string.ascii_uppercase[idx]
    return 'P{}'.format(idx)


class Lexicon:
    """ Lexicon of one agent: semiotic symbols plus the Hebbian learning of compositions.

    The default values of every parameter are the ones of the implementation used for
    the published experiments.

    :param int dim: dimension of the meaning space (number of sensory channels).
    :param float short_term_memory: traces decay as ``beta <- beta - beta / short_term_memory``
        at each step in which the meaning is not active.
    :param float learn_rate: increase of ``alpha_ij`` per step in which meanings ``i`` and ``j``
        are simultaneously active.
    :param float forget_rate: decay of ``alpha_ij`` per step, ``alpha <- alpha - forget_rate * alpha``.
    :param float association_threshold: a composed symbol is created when ``alpha_ij`` exceeds it.
    :param float meaning_range: elementary meanings are sampled uniformly in
        ``[-meaning_range, meaning_range]^dim``.
    :param float new_meaning_min_dist: minimum distance between a new elementary meaning and
        the existing ones (a new sample is drawn otherwise).
    :param float same_meaning_eps: two meanings closer than this are considered the same
        (used to avoid duplicated composed meanings).
    """
    def __init__(self, dim=3, short_term_memory=100, learn_rate=1e-2, forget_rate=1e-3,
                 association_threshold=0.4, meaning_range=2, new_meaning_min_dist=0.1,
                 same_meaning_eps=1e-2):
        self.dim = dim
        self.short_term_memory = short_term_memory
        self.learn_rate = learn_rate
        self.forget_rate = forget_rate
        self.association_threshold = association_threshold
        self.meaning_range = meaning_range
        self.new_meaning_min_dist = new_meaning_min_dist
        self.same_meaning_eps = same_meaning_eps

        self.meanings = np.empty((0, self.dim))
        self.words = []
        self.active = np.array([]).astype(int)
        self.traces = np.empty((0)).astype(float)
        self.assoc_weights = np.empty((0, 0)).astype(float)
        self.man = np.empty((0, 0))

        self.prototype_to_meaning = {}  # prototype index -> meaning index (elementary symbols)
        self.word_to_meaning = {}       # word -> meaning index

        # Graph view of the MAN. Node ``i + 1`` is meaning ``i``; its ``layer`` attribute
        # (0 for elementary symbols) defines ``elementary_words`` and ``composed_words``.
        self.nx_tree = nx.DiGraph()

    # ------------------------------------------------------------------ #
    # Logged quantities (used to compute LSS, CSS and lexicon sizes)
    # ------------------------------------------------------------------ #
    @property
    def elementary_words(self):
        """ Words of elementary symbols (MAN layer 0), ordered by meaning index. """
        inv_word_meaning = {v: k for k, v in self.word_to_meaning.items()}
        return [inv_word_meaning[i] for i in range(len(self.nx_tree.nodes))
                if self.nx_tree.nodes[i + 1]['layer'] == 0]

    @property
    def composed_words(self):
        """ Words of composed symbols (MAN layer > 0), ordered by meaning index. """
        inv_word_meaning = {v: k for k, v in self.word_to_meaning.items()}
        return [inv_word_meaning[i] for i in range(len(self.nx_tree.nodes))
                if self.nx_tree.nodes[i + 1]['layer'] > 0]

    @property
    def assoc_list(self):
        """ Associations ``(parent_word_1, parent_word_2, word)`` of all composed words. """
        alist = []
        for word in self.words:
            parents = self.parents(word)
            if parents[0] is not None and parents[1] is not None:
                alist.append((parents[0], parents[1], word))
        return alist

    @property
    def n_words(self):
        return self.traces.shape[0]

    @property
    def meaning_to_prototype(self):
        return {v: k for k, v in self.prototype_to_meaning.items()}

    # ------------------------------------------------------------------ #
    # Hebbian learning of compositionality (Sec. 3.6)
    # ------------------------------------------------------------------ #
    def step(self):
        """ Updates traces and association weights with the currently active meanings and
        creates the composed symbols whose association weight exceeds the threshold.
        Must be called once per control step, after the active meanings have been set. """
        if len(self.traces) == 0:
            return
        self.propagate_tree()
        self.traces = self.traces - (1 / self.short_term_memory) * self.traces
        self.traces[self.active] = 1.0
        coactive = np.zeros(len(self.traces))
        coactive[self.active] = 1.0
        self.assoc_weights += self.learn_rate * np.outer(coactive, coactive)
        self.assoc_weights = self.assoc_weights + self.forget_rate * (-self.assoc_weights)

        creation_idxs = np.where(self.assoc_weights > self.association_threshold)
        for m1, m2 in zip(*creation_idxs):
            if m1 != m2:
                self.create_relation(m1, m2)

    def propagate_tree(self):
        """ Activates the composed meanings whose two parents are active, layer by layer,
        and recomputes their meaning vectors as the sum of their parents' meanings. """
        input_indices = self.active.astype(int).copy()
        vector = np.zeros(self.man.shape[0])
        vector[input_indices] = 1

        # Only elementary meanings are inputs of the network
        elementary = [*self.prototype_to_meaning.values()]
        meaning_vectors = np.zeros((self.man.shape[0], self.dim)).astype(float)
        meaning_vectors[elementary] = self.meanings[elementary].copy()

        prev_vector = np.zeros(self.n_words)
        while any(vector != prev_vector):
            prev_vector = vector.copy()
            vector[input_indices] = 1
            meaning_vectors[elementary] = self.meanings[elementary].copy()
            vector = self.man.dot(vector)
            meaning_vectors = self.man.dot(meaning_vectors)
            vector = np.where(vector == 2, 1, 0)

            # Update the meanings of the newly activated composed symbols
            if any(vector == 1):
                for j in np.where(vector == 1)[0]:
                    if prev_vector[j] == 0:
                        if j not in self.active:
                            self.active = np.append(self.active, j)
                            self.set_meaning(j, meaning_vectors[j])
                        elif j not in input_indices:
                            self.set_meaning(j, meaning_vectors[j])
            vector[input_indices] = 1

    def backpropagate_tree(self, meaning_idx):
        """ Binary vector marking ``meaning_idx`` and all its ancestors in the MAN. """
        vector = np.zeros(self.man.shape[0])
        prev_vector = np.zeros(self.man.shape[0])
        vector[meaning_idx] = 1
        while any(vector != prev_vector):
            prev_vector = vector.copy()
            vector[meaning_idx] = 1
            vector = self.man.T.dot(vector)
        vector[meaning_idx] = 1
        return vector

    # ------------------------------------------------------------------ #
    # Creation of semiotic symbols
    # ------------------------------------------------------------------ #
    def add_meaning(self, new_meaning):
        """ Appends a new meaning and grows all per-meaning structures. """
        self.meanings = np.vstack((self.meanings, new_meaning))
        self.traces = np.pad(self.traces, (0, 1))
        self.man = np.pad(self.man, (0, 1))
        self.assoc_weights = np.pad(self.assoc_weights, (0, 1))
        self.nx_tree.add_node(len(self.meanings), layer=0,
                              lab=_node_label(len(self.prototype_to_meaning) - 1))

    def add_word(self, new_word):
        self.words.append(new_word)

    def link_semiotics(self, word, meaning_idx, prototype_idx=None):
        """ Links a word (and optionally a prototype) to a meaning. """
        if prototype_idx is not None:
            self.prototype_to_meaning[prototype_idx] = meaning_idx
        self.word_to_meaning[word] = meaning_idx

    def create_word(self, use_word=None, patt_idx=None):
        """ Creates an elementary semiotic symbol: a random meaning in S, linked to prototype
        ``patt_idx`` and to the word ``use_word`` (or a new random word if None). """
        if use_word is None:
            new_word = create_random_word()
            while new_word in self.words:
                new_word = create_random_word()
            self.add_word(new_word)
        else:
            new_word = use_word
            if new_word in self.words:
                logging.warning('Creating word "%s", which already exists in the lexicon.', new_word)
            self.add_word(new_word)
        new_meaning = np.random.uniform(-self.meaning_range, self.meaning_range, size=self.dim)
        # Resample until the meaning does not overlap with an existing one
        while self.word_exists(new_meaning, eps=self.new_meaning_min_dist):
            new_meaning = np.random.uniform(-self.meaning_range, self.meaning_range, size=self.dim)
        self.add_meaning(new_meaning)
        self.link_semiotics(new_word, len(self.meanings) - 1, patt_idx)

    def create_relation(self, w1, w2, word=None):
        """ Creates the composed symbol of meanings ``w1`` and ``w2`` (Eq. 6), with the word
        ``word`` (or a new random word if None). It is not created if it already exists, if one
        meaning is an ancestor of the other, or if both share an ancestor (redundancy). If the
        composition already exists and ``word`` is given, the existing symbol is renamed. """
        parents_row = np.zeros(self.man.shape[0])
        parents_row[[w1, w2]] = 1
        # The composition w1 + w2 already exists
        if any(np.abs(self.man - parents_row).sum(1) == 0):
            if word is not None:
                idx = np.where(np.abs(self.man - parents_row).sum(1) == 0)[0][0]
                self.set_word(idx, word, is_pattern=False)
            return
        new_meaning = self.meanings[w1] + self.meanings[w2]
        if self.word_exists(new_meaning, eps=self.same_meaning_eps):
            return
        row1 = self.man[w1]
        row2 = self.man[w2]
        # One meaning is a parent of the other, or they share a parent
        if row1[w2] == 1 or row2[w1] == 1 or np.logical_and(row1 == 1, row1 == row2).any():
            return
        if any(self.backpropagate_tree(w1) + self.backpropagate_tree(w2) == 2):
            return

        # All conditions met => create the new symbol
        if word is None:
            new_word = create_random_word()
            while new_word in self.words:
                new_word = create_random_word()
        else:
            new_word = word
        self.add_word(new_word)
        self.add_meaning(new_meaning.copy())
        self.link_semiotics(new_word, len(self.meanings) - 1, prototype_idx=None)

        self.man[-1, [w1, w2]] = 1
        layer = max(self.nx_tree.nodes[w1 + 1]['layer'], self.nx_tree.nodes[w2 + 1]['layer']) + 1
        labels = nx.get_node_attributes(self.nx_tree, 'lab')
        lab = ''.join(labels[w1 + 1].split('+')) + '+' + ''.join(labels[w2 + 1].split('+'))
        self.nx_tree.add_node(len(self.man), layer=layer, lab=lab,
                              word=new_meaning.round(3).tolist())
        self.nx_tree.add_edge(w1 + 1, len(self.man))
        self.nx_tree.add_edge(w2 + 1, len(self.man))

    # ------------------------------------------------------------------ #
    # Queries and updates
    # ------------------------------------------------------------------ #
    def parents(self, word):
        """ Parent words of a composed word, or ``(None, None)`` for elementary words. """
        meaning_idx = self.word_to_meaning[word]
        parents_row = self.man[meaning_idx]
        if np.sum(parents_row) == 0:
            return (None, None)
        parent_ids = np.where(parents_row == 1)[0]
        assert len(parent_ids) == 2
        return (self.words[parent_ids[0]], self.words[parent_ids[1]])

    def word_distance(self, m1, m2):
        return np.linalg.norm(m1 - m2)

    def word_exists(self, meaning, eps=1e-2):
        """ Whether some stored meaning is closer than ``eps`` to ``meaning``. """
        return any([self.word_distance(meaning, mm) < eps for mm in self.meanings])

    def reset_visible_words(self):
        """ Clears the active meanings. Must be called at the start of every control step. """
        self.active = np.array([]).astype(int)

    def add_visible_word(self, idx, is_pattern=True):
        """ Activates a meaning, given by its own index or by its prototype index. """
        ii = idx if not is_pattern else self.prototype_to_meaning[idx]
        self.active = np.append(self.active, ii)

    def set_meaning(self, idx, new_meaning, is_pattern=False):
        ii = idx if not is_pattern else self.prototype_to_meaning[idx]
        self.meanings[ii] = new_meaning.copy()

    def set_word(self, idx, new_word, is_pattern=False):
        """ Changes the form of a symbol (given by meaning or prototype index) to ``new_word``. """
        ii = idx if not is_pattern else self.prototype_to_meaning[idx]
        prev_word = self.words[ii]
        if prev_word == new_word:
            if self.word_to_meaning[prev_word] != ii:
                logging.warning('Word "%s" is linked to meaning %d instead of %d.',
                                prev_word, self.word_to_meaning[prev_word], ii)
            return
        self.word_to_meaning[new_word] = ii
        del self.word_to_meaning[prev_word]
        self.words[ii] = new_word

    def get_word(self, idx, is_pattern=True):
        ii = idx if not is_pattern else self.prototype_to_meaning[idx]
        return self.words[ii]

    def random_word(self):
        """ Returns a random word of the lexicon and its meaning index. """
        idx = np.random.choice(len(self.words))
        return self.words[idx], idx
