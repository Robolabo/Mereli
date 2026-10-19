""" Random generation of word forms (paper Sec. 3.4: "arbitrary three syllable words").

A syllable is a consonant followed by a vowel. Words are drawn with the global
``np.random`` generator, one ``choice`` call per letter group, so the random stream
is identical to the original implementation.
"""
import numpy as np

# NOTE: 'r' appears twice. This is kept on purpose: the list length and order define
# how ``np.random.choice`` indices map to syllables, so changing it would change every
# word generated with a given seed (and the published results).
CONSONANTS = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'r', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't',
              'v', 'w', 'x', 'y', 'z', 'ch', 'xh', 'zh', 'wh']
VOWELS = ['a', 'e', 'i', 'o', 'u']


def create_random_word(num_syllables=3):
    """ Returns a random word made of ``num_syllables`` consonant-vowel syllables. """
    new_word = ''
    for _ in range(num_syllables):
        ci = np.random.choice(len(CONSONANTS))
        vi = np.random.choice(len(VOWELS))
        new_word += CONSONANTS[ci] + VOWELS[vi]
    return new_word
