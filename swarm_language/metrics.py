""" Swarm-level assessment metrics (paper Sec. 5.1, Eqs. 10-13).

These functions are re-implemented from the equations in the paper; they are not the
scripts used to produce the published figures.
"""
import itertools


def jaccard(a, b):
    """ Jaccard similarity of two collections (Eq. 11). Two empty sets give 0. """
    a, b = set(a), set(b)
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _mean_pairwise_jaccard(sets):
    sets = [set(s) for s in sets]
    if len(sets) < 2:
        raise ValueError('At least two agents are required.')
    pairs = list(itertools.combinations(sets, 2))
    return sum(jaccard(a, b) for a, b in pairs) / len(pairs)


def lexicon_sharing_score(lexicons):
    """ LSS (Eq. 10): mean pairwise Jaccard similarity of the agents' lexicons.

    :param lexicons: one collection of words per agent.
    """
    return _mean_pairwise_jaccard(lexicons)


def compositionality_sharing_score(associations):
    """ CSS (Eq. 13): mean pairwise Jaccard similarity of the agents' association sets.

    :param associations: one collection of ``(parent_1, parent_2, word)`` tuples per agent.
        The order of the two parents is ignored, so ``(a, b, w)`` and ``(b, a, w)`` are the
        same association.
    """
    normalized = [{(frozenset((p1, p2)), w) for p1, p2, w in agent_assocs}
                  for agent_assocs in associations]
    return _mean_pairwise_jaccard(normalized)


def lexicon_size(lexicons):
    """ Number of distinct words in the swarm, ``upsilon`` (Eq. 12). """
    return len(set().union(*map(set, lexicons)))


REPORT_HEADER = '{:>8} {:>8} {:>6} {:>6} {:>8} {:>6} {:>6}'.format(
    'step', 'time(s)', 'LSS', 'CSS', 'upsilon', 'elem', 'comp')


def report_line(step, metrics, sim_time=None):
    """ One row of the lexicon emergence table printed by the demo and by Mereli. """
    time_str = '{:.1f}'.format(sim_time) if sim_time is not None else '-'
    return '{:>8} {:>8} {:>6.3f} {:>6.3f} {:>8} {:>6} {:>6}'.format(
        step, time_str, metrics['LSS'], metrics['CSS'], metrics['upsilon'],
        metrics['upsilon_elem'], metrics['upsilon_comp'])


def swarm_metrics(lexicons):
    """ All metrics of a list of :class:`~swarm_language.lexicon.Lexicon` objects.

    :returns: dict with ``LSS``, ``CSS``, ``upsilon``, ``upsilon_elem`` and ``upsilon_comp``.
    """
    return {
        'LSS': lexicon_sharing_score([lex.words for lex in lexicons]),
        'CSS': compositionality_sharing_score([lex.assoc_list for lex in lexicons]),
        'upsilon': lexicon_size([lex.words for lex in lexicons]),
        'upsilon_elem': lexicon_size([lex.elementary_words for lex in lexicons]),
        'upsilon_comp': lexicon_size([lex.composed_words for lex in lexicons]),
    }
