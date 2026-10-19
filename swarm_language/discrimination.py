""" Iconisation and discrimination (paper Sec. 3.3, Eqs. 3-5).

Each robot incrementally partitions its sensory space into regions represented by
prototypes (a discrimination game). A stimulus is assigned to its closest prototype;
if it is too far from all of them, it becomes a new prototype.
"""
import numpy as np


class PatternDetector:
    """ Set of prototypes learned by one agent.

    :param int vector_dim: dimension of the feature vectors (7 in the paper: r, g, b, three
        shape probabilities and quality).
    :param float threshold: discrimination threshold. A stimulus farther than this from every
        prototype creates a new prototype.
    :param float adaptation_rate: step size used to move the closest prototype towards the
        stimulus, ``xi <- (1 - rate) * xi + rate * x``.

    :var np.ndarray pattern_mat: prototypes, one per row (shape ``(K, vector_dim)``).
    :var list names: one letter per prototype, only used for printing and plotting.
    """
    def __init__(self, vector_dim, threshold=0.4, adaptation_rate=0.25):
        self.vector_dim = vector_dim
        self.threshold = threshold
        self.adaptation_rate = adaptation_rate
        self.pattern_mat = np.empty((0, vector_dim)).astype(float)
        self.names = []

    def compute_distance(self, phi):
        """ Euclidean distance between ``phi`` and every prototype. """
        return np.linalg.norm(self.pattern_mat - phi, axis=1)

    def assign_name(self, idx):
        """ Short name of a prototype (plotting and printing only). """
        if idx < 26:
            return chr(97 + idx)
        return 'p{}'.format(idx)

    def step(self, phi):
        """ Classifies stimulus ``phi`` and updates the prototypes.

        :param np.ndarray phi: feature vector of length ``vector_dim``.
        :returns: tuple ``(pattern_detected, pattern)``. ``pattern_detected`` is False when
            ``phi`` created a new prototype. ``pattern`` is the index of the matched or
            newly created prototype.
        """
        if self.pattern_mat.shape[0] == 0:
            self.pattern_mat = np.vstack((self.pattern_mat, phi))
            self.names.append(self.assign_name(0))
            return False, 0

        dist = self.compute_distance(phi)
        closest = np.argmin(dist)
        if dist[closest] > self.threshold:
            self.pattern_mat = np.vstack((self.pattern_mat, phi))
            pattern = len(self.pattern_mat) - 1
            self.names.append(self.assign_name(pattern))
            return False, pattern
        rate = self.adaptation_rate
        self.pattern_mat[closest] = (1 - rate) * self.pattern_mat[closest] + rate * phi
        return True, closest

    def random_pattern(self):
        """ Returns a copy of a random prototype and its index. """
        i = np.random.choice(self.n_patterns)
        return self.pattern_mat[i].copy(), i

    @property
    def n_patterns(self):
        return self.pattern_mat.shape[0]

    def reset(self):
        self.pattern_mat = np.empty((0, self.vector_dim)).astype(float)
