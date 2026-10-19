import unittest
from swarm_language.metrics import (jaccard, lexicon_sharing_score, compositionality_sharing_score,
                                    lexicon_size)


class TestMetrics(unittest.TestCase):
    def test_jaccard(self):
        self.assertEqual(jaccard({'a', 'b'}, {'b', 'c'}), 1 / 3)
        self.assertEqual(jaccard([], []), 0.0)

    def test_lss(self):
        self.assertEqual(lexicon_sharing_score([['a', 'b'], ['a', 'b'], ['b', 'a']]), 1.0)
        # pairs: (1, 2) -> 1/2, (1, 3) -> 0, (2, 3) -> 1/2
        self.assertAlmostEqual(lexicon_sharing_score([['a', 'b'], ['b'], ['c']]), (0.5 + 0 + 0) / 3)
        with self.assertRaises(ValueError):
            lexicon_sharing_score([['a']])

    def test_css_ignores_parent_order(self):
        a1 = [('red', 'cube', 'redcube')]
        a2 = [('cube', 'red', 'redcube')]
        self.assertEqual(compositionality_sharing_score([a1, a2]), 1.0)
        self.assertEqual(compositionality_sharing_score([a1, [('red', 'ball', 'redcube')]]), 0.0)

    def test_lexicon_size(self):
        self.assertEqual(lexicon_size([['a', 'b'], ['b', 'c'], []]), 3)


if __name__ == '__main__':
    unittest.main()
