""" Plotting helpers (optional; require matplotlib). """
import networkx as nx


def plot_man(lexicon, show='words', ax=None):
    """ Draws the Meaning Association Network of one agent, one column per layer.

    :param Lexicon lexicon: lexicon of the agent.
    :param str show: node label contents, ``'words'`` (forms) or ``'meanings'`` (vectors).
    :param ax: matplotlib axis (a new figure is created if None).
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    graph = lexicon.nx_tree
    pos = nx.multipartite_layout(graph, subset_key='layer')
    names = nx.get_node_attributes(graph, 'lab')
    if show == 'meanings':
        labels = {i + 1: '{} : ({:.2f}, {:.2f}, {:.2f})'.format(names[i + 1], *lexicon.meanings[i][:3])
                  for i in range(lexicon.n_words)}
    else:
        labels = {i + 1: '{} : {}'.format(names[i + 1], lexicon.words[i]) for i in range(lexicon.n_words)}
    nx.draw_networkx_nodes(graph, pos=pos, ax=ax)
    nx.draw_networkx_edges(graph, pos=pos, ax=ax)
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10, ax=ax,
                            bbox={'ec': 'k', 'fc': 'white', 'alpha': 0.7})
    return ax
