import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class AnimatedPlot:
    def __init__(self, name):
        self.name = name 
        self.axis = None
        self.t = 0
        self.yData = []
        self.xData = []

    def update(self, robot):
        pass

    def initialize(self):
        pass

    def set_xlim(self, xmin, xmax):
        self.axis.set_xlim([xmin, xmax])
    
    def set_ylim(self, ymin, ymax):
        self.axis.set_ylim([ymin, ymax])
        
    def draw_artist(self):
        for i in range(len(self.yData)):
            self.axis.draw_artist(self.axis.lines[i])
    
    @property
    def subplot_kw(self):
        return {self.name : {}}

class AnimatedSensorLineplot(AnimatedPlot):
    def __init__(self, *args, sensor='distance_sensor', sectors=[0], **kwargs):
        super(AnimatedSensorLineplot, self).__init__(*args, **kwargs)
        self.target_sensor = sensor
        self.sectors = sectors

    def update(self, robot):
        if 'light_sensor' in self.target_sensor:
            new_y = robot.sensors['light_sensor'].reading[self.target_sensor]
        else:    
            new_y = robot.sensors[self.target_sensor].reading
        if isinstance(new_y, float):
            self.yData[0] = np.r_[self.yData[0,1:], new_y]
        else:
            for i in range(len(self.sectors)):
                self.yData[i] = np.r_[self.yData[i,1:], new_y[self.sectors[i]]]
        for i in range(len(self.yData)):
            self.axis.lines[i].set_ydata(self.yData[i])
            # re-render the artist, updating the canvas state, but not the screen
            self.axis.draw_artist(self.axis.lines[i])

    def initialize(self):
        self.set_ylim(0,1)
        self.xData = np.arange(500) 
        self.yData = np.zeros([len(self.sectors), self.xData.shape[0]]) 
        for i in range(len(self.sectors)):
            ln = self.axis.plot(self.xData, self.yData[i], animated=True)

class AnimatedPolarEpuck(AnimatedPlot):
    def __init__(self, *args, sensors=['distance_sensor'], **kwargs):
        super(AnimatedPolarEpuck, self).__init__(*args, **kwargs)
        self.sensors=sensors
        self.colors = {'distance_sensor' : 'k', 'red_light_sensor' : 'r', 'blue_light_sensor' : 'b'}

    def update(self, robot):
        head_ori = robot.orientation[-1]
        aux_s = 'distance_sensor' if 'distance_sensor' in self.sensors else 'light_sensor'
        sensor_dirs = robot.sensors[aux_s].directions(head_ori)
        for j in range(len(self.sensors)):
            sensor = self.sensors[j]
            for i in range(8):
                if 'light_sensor' in sensor: 
                    new_y = robot.sensors['light_sensor'].reading[sensor]
                else:
                    new_y = robot.sensors[self.sensors[j]].reading
                self.axis.lines[j].set_ydata(np.r_[new_y, new_y[0]])
                self.axis.lines[j].set_xdata(np.r_[sensor_dirs, sensor_dirs[0]])
        for i in range(8):
            self.axis.lines[len(self.sensors)+1+i].set_xdata([sensor_dirs[i], sensor_dirs[i]])
            self.axis.lines[len(self.sensors)+1+i].set_ydata([0, 1])

            
        self.axis.lines[len(self.sensors)].set_xdata(np.array([head_ori, head_ori]))
        self.axis.lines[len(self.sensors)].set_ydata(np.array([0,0.1]))
        for ln in self.axis.lines:
            self.axis.draw_artist(ln)

    def initialize(self):
        self.axis.set_rorigin(-0.1)
        self.set_ylim(0,1)
        for sensor in self.sensors:
            ln = self.axis.plot([0,0], [0,0], color=self.colors[sensor])
        self.axis.plot([0,0.0],[0.,0.0],linewidth=5, color='k')
        # Initialize lines for sensor directions
        for i in range(8):
            ln = self.axis.plot([0,0], [0,0], color='r')

    @property
    def subplot_kw(self):
        return {self.name : {'projection' : 'polar'}}
        

class AnimatedCustomLineplot(AnimatedPlot):
    def __init__(self, *args, **kwargs):
        super(AnimatedCustomLineplot, self).__init__(*args, **kwargs)
        self.variables = ['virtual_particle@dist_clst_lmark', 'virtual_particle@dist_clst_neighbor', 'virtual_particle@dist_clst_lmark_av']
  
    def get_variable(self, robot, varcode):
        aux = varcode.split('@')
        varpath = aux[0]
        varname = aux[1]
        return getattr(getattr(robot, varpath), varname)
        
    def update(self, robot):
        new_y = []
        for i in range(len(self.variables)):
            new_y.append(self.get_variable(robot, self.variables[i]))
        if isinstance(new_y, float):
            self.yData[0] = np.r_[self.yData[0,1:], new_y]
        else:
            for i in range(len(self.variables)):
                self.yData[i] = np.r_[self.yData[i,1:], new_y[i]]
        for i in range(len(self.yData)):
            self.axis.lines[i].set_ydata(self.yData[i])
            # if self.t >= 500:
            #     self.axis.lines[i].set_xdata(np.arange(self.t-500, self.t))
            #     # self.set_xlim(self.t-500, self.t)
            # else:
            #     self.axis.lines[i].set_xdata(np.arange(self.t, self.t+500))
            #     self.set_xlim(self.t, self.t+500)
            # re-render the artist, updating the canvas state, but not the screen
            self.axis.draw_artist(self.axis.lines[i])
        self.t += 1

    def initialize(self):
        self.set_ylim(0,2)
        self.xData = np.arange(500) 
        self.yData = np.zeros([len(self.variables), self.xData.shape[0]]) 
        for i in range(len(self.variables)):
            ln = self.axis.plot(self.xData, self.yData[i], animated=True)

class AnimatedCommunicationSpace(AnimatedPlot):
    def __init__(self, *args, **kwargs):
        super(AnimatedCommunicationSpace, self).__init__(*args, **kwargs)
    
    def update(self, robot):
        own_state = robot.virtual_particle.state 
        # print(robot.virtual_particle.)
        neigh_states = [vv.state for vv in robot.virtual_particle.neighbors]
        lmarks = robot.virtual_particle.landmarks
        disabled = robot.virtual_particle.disabled_lmarks
        lm_colors = [('grey', 'red')[i not in disabled] for i in range(len(lmarks))] 
        self.axis.collections[0].set_offsets(own_state)
        self.axis.collections[1].set_offsets(neigh_states)
        self.axis.collections[2].set_offsets(lmarks)
        self.axis.collections[2].set_facecolor(lm_colors)
        
        for ln in self.axis.collections:
            self.axis.draw_artist(ln)

    def initialize(self):
        self.set_ylim(-1.1,1.1)
        self.set_xlim(-1.1,1.1)
        self.axis.scatter([], [], color='b',zorder=100, s=102, animated=True) # Own state
        self.axis.scatter([], [], color='k',zorder=100, s=102, animated=True) # Neigh states
        self.axis.scatter([], [], marker='*', edgecolors='k', s=250, zorder=101, color='r', animated=True) # Lmarks
    
class AnimatedImage(AnimatedPlot):
    def __init__(self, *args, **kwargs):
        super(AnimatedImage, self).__init__(*args, **kwargs)
    
    def update(self, robot):
        img = robot.sensors['camera'].reading
        # print(robot.virtual_particle.)
        # self.axis.collections[0].set_offsets(own_state)
        # self.axis.collections[1].set_offsets(neigh_states)
        # self.axis.collections[2].set_offsets(lmarks)
        # self.axis.collections[2].set_facecolor(lm_colors)
        self.axis.get_children()[0].set_data(img)
        self.axis.draw_artist(self.axis.get_children()[0])

    def initialize(self):
        self.axis.imshow(np.zeros((50,50)))
        self.axis.get_xaxis().set_visible(False)
        self.axis.get_yaxis().set_visible(False)
        # self.axis.scatter([], [], color='b',zorder=100, s=102, animated=True) # Own state
        
import networkx as nx
class AnimatedNeuralNetwork(AnimatedPlot):
    def __init__(self, *args, **kwargs):
        super(AnimatedNeuralNetwork, self).__init__(*args, **kwargs)
        self.graph_plotted = False

    def plot_graph(self, ann):
        layers = np.array(ann.input_ensemble_names + [k for k in ann.ensemble_names if k not in ann.motor_ensemble_names] + ann.motor_ensemble_names)
        G = nx.Graph()
        node_options = {"edgecolors": "red", "node_size": 800, "alpha": 0.9}
        for inp_name, inp in ann.graph['inputs'].items():
            G.add_node(inp_name, layer=np.where(layers == inp['ensemble'])[0][0], **node_options)
        for node_name, node in ann.graph['neurons'].items():
            G.add_node(node_name, layer=np.where(layers == node['ensemble'])[0][0], zorder=10, **node_options)
        for conn_name, conn in ann.graph['synapses'].items():    
            G.add_edge(conn['pre'], conn['post'], zorder=5)
        pos = nx.multipartite_layout(G, subset_key="layer")
        for k in pos:
            pos[k][0] *= 2
            pos[k][1] *= 1.5 
        # aaa = nx.draw_networkx(G, pos, min_source_margin=100,min_target_margin=100, node_size=500, node_color=['blue']*len(pos),ax=self.axis, )
        wmax = 5
        edge_color = [(1 + conn['weight'] / wmax) / 2 for conn in ann.graph['synapses'].values()]
        nx.draw_networkx_edges(G, pos, ax=self.axis)
        nx.draw_networkx_edges(
            G,
            pos,
            alpha=0.5,
            edge_color=edge_color,
            ax=self.axis,
            edge_cmap=plt.cm.RdBu,
        )
        aa = nx.draw_networkx_nodes(G, pos, node_size=500, node_color=['blue']*len(pos),ax=self.axis, )
        self.axis.collections[3].set_edgecolor("#000000")
        self.axis.collections[3].set_linewidth(3)
        self.axis.collections[1].set_linewidth(1.5)
        self.axis.collections[2].set_linewidth(5)

    def update(self, robot):
        ann = robot.controller.neural_network
        if not self.graph_plotted:
            self.plot_graph(ann)
            self.graph_plotted = True
        cmap_neu = plt.get_cmap('bwr')
        cmap_inp = plt.get_cmap('Purples')
        
        color_inputs = [cmap_inp(int( i  * 256) ) for i in ann.inputs]
        color_neurons = [cmap_neu(int(((i + 1) / 2) * 256)) for i in ann.spikes]
        nodes = self.axis.collections[3] 
        nodes.set_facecolor(color_inputs + color_neurons)
        # self.axis.collections[2].set_zorder(1) 
        # self.axis.collections[1].set_zorder(2)
        for ln in self.axis.collections:
            self.axis.draw_artist(ln)

    def initialize(self):
        self.axis.scatter([], [], color='k', s=103, ) # Own state
        # self.axis.scatter([], [], color='k',zorder=100, s=102, animated=True) # Neigh states
        # self.axis.scatter([], [], edgecolors='k', s=250, zorder=101, color='r', animated=True) # Lmarks


class AnimatedLayout:

    def __init__(self):
        self.plots = []
        self.figsize = (7,10)
        self.fig = None # plt.figure(figsize=self.figsize)
        self.grid = None #"AA;BC"
    
    def initialize(self):
        from mereli.utils import merge_dicts
        subplot_kw = merge_dicts([p.subplot_kw for p in self.plots])
        self.fig, axes = plt.subplot_mosaic(self.grid, per_subplot_kw=subplot_kw, figsize=self.figsize) 
        plt.subplots_adjust(top=0.97, bottom=0.08, left=0.10, right=0.97, hspace=0.1, wspace=0.15)
        for p in self.plots:
            p.axis = axes[p.name]
            # p.(self.fig, self.gs)
            p.initialize()

        plt.show(block=False)
        plt.pause(0.1)
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        for p in self.plots:
            p.draw_artist() 
        self.fig.canvas.blit(self.fig.bbox)

    def add_plot(self, plot_type='sensor_lineplot', name=None, **kwargs):
        new_plot = None
        if  plot_type == 'sensor_lineplot':
            new_plot = AnimatedSensorLineplot(name, **kwargs)
        elif plot_type == 'epuck_polar':
            new_plot = AnimatedPolarEpuck(name, **kwargs)
        elif plot_type == 'comm_space':
            new_plot = AnimatedCommunicationSpace(name, **kwargs)
        elif plot_type == 'custom_lineplot':
            new_plot = AnimatedCustomLineplot(name, **kwargs)
        elif plot_type == 'animated_ann':
            new_plot = AnimatedNeuralNetwork(name, **kwargs)
        elif plot_type == 'animated_image':
            new_plot = AnimatedImage(name, **kwargs)
        else:
            pass
        self.plots.append(new_plot)
    
    def add_plots(self, plots, grid=None):
        if grid is None:
            self.grid = "" #TODO
        else:
            self.grid = grid
        for plt_cfg in plots:
            self.add_plot(**plt_cfg)

    def update(self, robot):
        self.fig.canvas.restore_region(self.bg)
        for p in self.plots:
            p.update(robot)
        # copy the image to the GUI state, but screen might not be changed yet
        self.fig.canvas.blit(self.fig.bbox)
        # flush any pending GUI events, re-painting the screen if needed
        self.fig.canvas.flush_events()
        # you can put a pause in if you want to slow things down
        # plt.pause(.1)



    def reset(self): 
        pass
