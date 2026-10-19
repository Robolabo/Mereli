""" Cultural evolution of perceptually grounded compositional lexicons in robot swarms.

Standalone implementation of the system of Sendra-Arranz & Gutiérrez, Applied Soft
Computing 202 (2026) 115879. Only numpy and networkx are required. See README.md.
"""
from .words import create_random_word, CONSONANTS, VOWELS
from .discrimination import PatternDetector
from .lexicon import Lexicon
from .perception import perceive_object, encode_color, encode_shape, encode_quality, FEATURE_DIM
from .agent import LanguageAgent
from .toy_world import ToyWorld
from .metrics import (jaccard, lexicon_sharing_score, compositionality_sharing_score,
                      lexicon_size, swarm_metrics, report_line, REPORT_HEADER)
