import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA


def plot_state_plane(neural_net, t_start=0, t_end=500,\
    n_pc=3, downsampled=False, fig=None):
    """Plots the state plane projected into 2D with PCA. 
    The plotted trajectory within the specified time window.
    #! Only implemented for rate_models for the moment.
    ========================================================================================
    - Args:
        neural_net [NeuralNetwork]: neural network instance with recorded data. 
        t_start [int] : starting time step of the state plane trajectory. 
        t_end [int] : ending time step of the state plane trajectory. 
        n_pc [int] : number of principal components to project the state plane with PCA.
        downsampled [bool] : whether to downsample neuronal time regime to the 
                environment time scale.
        fig [matplotlib.pyplot.figure or None] : if not None it uses an existing figure to 
                plot the data. 
    - Returns:
        Matplotlib figure
    ========================================================================================
    """
    if len(neural_net.monitor) < 1:
        raise Exception(logging.error('No ANN data was recorded.'))
    if neural_net.is_spiking:
        raise Exception(logging.error('State plane visualization not currently '\
            'supported for spiking neural networks.'))
    outputs = np.stack(tuple(neural_net.monitor.get('outputs').values())).T
    pca = PCA(n_components=n_pc).fit(outputs)
    state_pca = pca.transform(outputs)

    # light = neural_net.monitor.get_inputs()[1][:, -6:].max(1)
    # colors = [('k', 'b')[ls > 0.4] for ls in light]
    
    if fig is None:
        fig = plt.figure()
        if n_pc >= 3:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
    else:
        ax = fig.axes[0]
    Ts = neural_net.time_scale if downsampled else 1
    if n_pc >= 3:
        ax.plot(state_pca[t_start * neural_net.time_scale : t_end * neural_net.time_scale : Ts, 0],\
                state_pca[t_start * neural_net.time_scale : t_end * neural_net.time_scale : Ts, 1],\
                state_pca[t_start * neural_net.time_scale : t_end * neural_net.time_scale : Ts, 2])
    else:
        ax.plot(state_pca[t_start * neural_net.time_scale : t_end * neural_net.time_scale : Ts, 0],\
                state_pca[t_start * neural_net.time_scale : t_end * neural_net.time_scale : Ts, 1])
    ax.set_xlabel(r"Principal Component 1")
    ax.set_ylabel(r"Principal Component 2")
    if n_pc >= 3:
        ax.set_zlabel(r"Principal Component 3")
    ax.set_title(r"Phase plane of projected ANN state.")
    # plot.scatter(state_pca[0, 0], state_pca[0, 1], color='r')
    # plot.scatter(state_pca[-1, 0], state_pca[-1, 1], color='g')
    # plot.show()
    return fig


def plot_spikes(neural_net, show_inputs=True):
    """ Plots an eventplot of spikes of neurons x time steps. 
    Can only be used if the neuron models are spiking. 
    =====================================================================================
    - Args:
        neural_net [NeuralNetwork]: neural network instance with recorded data. 
        show_inputs [bool]: whether to show the encoded stimuli in addition to the spikes.
    - Returns: None
    =====================================================================================
    """
    if len(neural_net.monitor) < 1:
         raise Exception(logging.error('No ANN data was recorded.'))
    if not neural_net.is_spiking:
        raise Exception(logging.error('Visualization of spikes is only available in '\
            'spiking neural networks.'))
    _, ax = plt.subplots(1)
    encoded_inputs = np.stack(tuple(neural_net.monitor.get('encoded_inputs').values()))
    spikes = np.stack(tuple(neural_net.monitor.get('spikes').values()))
    values = np.vstack((encoded_inputs, spikes)) if show_inputs else spikes
    colors = []
    counter = 0
    for _, ens_neurons in neural_net.subpop_neurons.items():
        rgb_color = tuple([np.random.random() for _ in range(3)])
        colors.extend([rgb_color for _ in range(ens_neurons)])
        counter += ens_neurons
    ax.eventplot([np.where(v)[0] for v in values],\
                lineoffsets=1, linelengths=.5, colors='k')
    # if self.time_scale > 1:
    #     for i in np.arange(0, values.shape[1], self.time_scale):
    #         ax.vlines(i, 0, values.shape[0], color='r')
    ax.set_xlabel('Time')
    ax.set_ylabel('Neurons')
    ax.set_ylim((0, values.shape[0]))
    ax.set_xlim((0, values.shape[1]))
    plt.show()

def plot_neuron(neural_net, neuron):
    """ Plots the relevant variables of a single neuron. 
    ============================================================================
    - Args:
        neural_net [NeuralNetwork]: neural network instance with recorded data. 
        neuron [int or str]: identification of a neuron either with an index or 
            with the neuron's name.
    - Returns: None
    ============================================================================
    """
    if len(neural_net.monitor) < 1:
        raise Exception(logging.error('No ANN data was recorded.'))
    current, voltage, recov, theta = None, None, None, None
    if isinstance(neuron, int):
        if neuron > neural_net.num_neurons:
            raise Exception(logging.error('The neuron index must be lower '\
                'than the number of neurons ({}).'.format(neural_net.num_neurons)))
        if neuron < 0:
            raise Exception(logging.error('The neuron index must greater than 0.'))
        current = np.stack(tuple(neural_net.monitor.get('currents').values()))[neuron]
        voltage = np.stack(tuple(neural_net.monitor.get('voltages').values()))[neuron]
        recov = np.stack(tuple(neural_net.monitor.get('recovery').values()))[neuron]
        theta = np.stack(tuple(neural_net.monitor.get('neuron_theta').values()))[neuron]
    elif isinstance(neuron, str):
        current = neural_net.monitor.get('currents')[neuron]
        voltage = neural_net.monitor.get('voltages')[neuron]
        recov = neural_net.monitor.get('recovery')[neuron]
        theta = neural_net.monitor.get('neuron_theta')[neuron]
    f, axes = plt.subplots(4, 1, figsize=(10, 12))
    for ax, vals, varname in zip(axes, [current, voltage, recov, theta],\
                ['Somatic Current', 'Membrane Voltage', 'Recovery Variable', 'Theta']):
        ax.plot(vals)
        ax.set_xlabel('Time')
        ax.set_ylabel(varname)
        ax.set_title('{} of Neuron {}'.format(varname, neuron))
        ax.set_xlim([0, len(vals)])
    f.subplots_adjust(left=0.08, right=0.96, top=0.97, bottom=0.05, hspace=0.28)
    plt.show()

def plot_activities(neural_net):
    """ Plots the activities of all the motor neurons in the neural net.
    ===========================================================================
    - Args:
        neural_net [NeuralNetwork]: neural network instance with recorded data.
    - Returns: None
    ===========================================================================
    """
    if len(neural_net.monitor) < 1:
        raise Exception(logging.error('No ANN data was recorded.'))
    if not neural_net.is_spiking:
        raise Exception(logging.error('Visualization of activities is only available in '\
            'spiking neural networks.'))
    activities = neural_net.monitor.get('activities')
    f, axes = plt.subplots(len(activities)//2, 2, figsize=(13, 13))
    for ax, (name, activs) in zip(axes.flatten(), activities.items()):
        ax.plot(activs)
        ax.set_title('Activity of neuron {}'.format(name))
        ax.set_xlabel('Time')
        ax.set_ylabel('Activity')
        f.subplots_adjust(left=0.08, right=0.96, top=0.97, bottom=0.05, hspace=0.42)
    plt.show()

def plot_weights(neural_net):
    """ Plots the weighted adjacency matrix of the ANN.
    ===========================================================================
    - Args:
        neural_net [NeuralNetwork]: neural network instance with recorded data.
    - Returns: None
    ===========================================================================
    """
    if len(neural_net.monitor) < 1:
        raise Exception(logging.error('No ANN data was recorded.'))
    y_labels = [n for ii, n in enumerate(neural_net.pointers.keys()) if ii > neural_net.n_inputs-1]
    x_labels = [n for i, n in enumerate(neural_net.pointers.keys())]
    y_ticks = [v - n//2 - neural_net.n_inputs for ii, (v, n) in enumerate(zip(\
        neural_net.pointers.values(), neural_net.subpop_neurons.values()))\
        if ii > neural_net.n_inputs-1]
    x_ticks = [v - n//2 for ii, (v, n) in enumerate(zip(\
        neural_net.pointers.values(), neural_net.subpop_neurons.values()))]
    ax = sns.heatmap(neural_net.weights, cmap="vlag", annot=False, center=0.)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xticklabels(x_labels)
    plt.show()

def plot_ntx(neural_net):
    """ Plots a matrix of the same shape as the adjacency matrix of the ANN
    that represents the synapse type of each synapse (AMPA, GABA, NDMA, ...). 
    More formally, it specifies the neurotransmitter of the synspses.
    ===========================================================================
    - Args:
        neural_net [NeuralNetwork]: neural network instance with recorded data.
    - Returns: None
    ===========================================================================
    """
    if len(neural_net.monitor) < 1:
        raise Exception(logging.error('No ANN data was recorded.'))
    if not neural_net.is_spiking:
        raise Exception(logging.error('Visualization of synapse neurotransmitters '\
            'is only available in spiking neural networks.'))
    y_labels = [n for ii, n in enumerate(neural_net.pointers.keys())\
            if ii > neural_net.n_inputs-1]
    x_labels = [n for i, n in enumerate(neural_net.pointers.keys())]
    y_ticks = [v - n//2 - neural_net.n_inputs for ii, (v, n) in enumerate(zip(\
        neural_net.pointers.values(), neural_net.subpop_neurons.values()))\
        if ii > neural_net.n_inputs-1]
    x_ticks = [v - n//2 for ii, (v, n) in enumerate(zip(\
        neural_net.pointers.values(), neural_net.subpop_neurons.values()))]
    _, ax = plt.subplots(1)
    ntx = neural_net.synapses.ampa_mask + 2*neural_net.synapses.gaba_mask
    im = plt.imshow(ntx, interpolation='none')
    values = np.unique(ntx.ravel())
    colors = [im.cmap(im.norm(value)) for value in values]
    labels = ['No Synapse', 'AMPA+NDMA', 'GABA']
    patches = [mpatches.Patch(color=colors[i], label=labels[i])\
                    for i in range(len(values))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.subplots_adjust(left=.02, right=.85)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xticklabels(x_labels, rotation=90)
    plt.show()