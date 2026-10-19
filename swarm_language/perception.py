""" Sensing: from an observed object to channel-wise feature vectors (paper Secs. 3.2 and 4).

The feature vector is ``x = (r, g, b, p_cube_or_ball..., quality)`` of dimension 7: three color
components, three shape components and one quality value. Color and shape are noisy
estimates of the ground truth (Gaussian noise, clipped to [0, 1]).

Implementation details that matter to reproduce the published runs:

* Colors are encoded with intensity 0.8, e.g. red = (0.8, 0, 0).
* The shape components are ordered (ball, cube, pyramid).
* Quality is derived from the object color: green -> +1, red -> -1, any other -> 0, plus
  Gaussian noise, clipped to [-1, 1].
* Random numbers are drawn from the global ``np.random`` generator in this order: color,
  shape (preceded by one extra draw if ``perspective_noise`` is on), quality.
"""
import numpy as np

COLOR_CODES = {'red': [0.8, 0, 0], 'green': [0, 0.8, 0], 'blue': [0, 0, 0.8]}
# Faulty color sensor (robustness experiment R2): red is perceived as (0.5, 0, 0.5).
DISTORTED_COLOR_CODES = {'red': [0.5, 0, 0.5], 'green': [0, 0.8, 0], 'blue': [0, 0, 0.8]}

SHAPE_CODES = {'ball': [1, 0, 0], 'cube': [0, 1, 0], 'pyramid': [0, 0, 1]}
# Faulty shape perception.
DISTORTED_SHAPE_CODES = {'cube': [0, 0, 1], 'ball': [1, 0, 0], 'pyramid': [0, 0.5, 0.5]}
# Perspective-dependent perception (robustness experiment R1): with probability 0.5 a cube
# is perceived as an irregular shape between a cube and a pyramid.
PERSPECTIVE_CUBE_CODES = ([0, 1, 0], [0.25, 0, 0.75])

QUALITY_BY_COLOR = {'green': 1, 'red': -1}

NOISE_STD = 0.05
FEATURE_DIM = 7


def encode_color(color, distorted_perception=False, noise_std=NOISE_STD):
    """ Noisy RGB estimate of a color name. Non-string colors are returned unchanged (no noise). """
    if not isinstance(color, str):
        return np.array(color)
    codes = DISTORTED_COLOR_CODES if distorted_perception else COLOR_CODES
    num_color = np.array(codes.get(color))
    return np.clip(np.random.normal(num_color, noise_std), a_min=0, a_max=1)


def encode_shape(shape, perspective_noise=False, distorted_perception=False, noise_std=NOISE_STD):
    """ Noisy (ball, cube, pyramid) estimate of a shape name. """
    if perspective_noise:
        rnd_aux = np.random.random() < 0.5
        codes = dict(SHAPE_CODES, cube=PERSPECTIVE_CUBE_CODES[rnd_aux])
    elif distorted_perception:
        codes = DISTORTED_SHAPE_CODES
    else:
        codes = SHAPE_CODES
    num_shape = np.array(codes.get(shape))
    return np.clip(np.random.normal(num_shape, noise_std), a_min=0, a_max=1)


def encode_quality(color, noise_std=NOISE_STD):
    """ Noisy quality score in [-1, 1], derived from the object color. """
    score = QUALITY_BY_COLOR.get(color, 0)
    return np.clip(np.random.normal(score, noise_std), a_min=-1, a_max=1)


def split_channels(channels):
    """ Builds one stimulus per sensory channel: the full-length feature vector with every
    other channel set to zero. Discrimination is performed on each of them separately. """
    stimuli = []
    for i in range(len(channels)):
        stimuli.append(np.hstack([(v, np.zeros_like(v))[i != j] for j, v in enumerate(channels)]))
    return stimuli


def perceive_object(obj, distorted_color=False, distorted_shape=False, perspective_noise=False):
    """ Converts an observed object into the list of channel stimuli (color, shape, quality).

    :param dict obj: object description with at least the keys ``'color'`` (e.g. ``'red'``)
        and ``'shape'`` (``'ball'``, ``'cube'`` or ``'pyramid'``).
    :returns: list of three feature vectors of length 7.
    """
    color = encode_color(obj['color'], distorted_perception=distorted_color)
    shape = encode_shape(obj['shape'], perspective_noise=perspective_noise,
                         distorted_perception=distorted_shape)
    quality = encode_quality(obj['color'])
    return split_channels([color, shape, quality])
