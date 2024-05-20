import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from mereli.utils import compute_angle



class AnimatedPlot:
    def __init__(self, name, plot_buffer=500):
        self.name = name 
        self.plot_buffer = plot_buffer
        self.axis = None
        self.t = 0
        self.yData = []
        self.xData = []

    def update(self, robot):
        pass

    def initialize(self, world):
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

class GeneralInformation(AnimatedPlot):
    def __init__(self, *args, variables=None,  **kwargs):
        super(GeneralInformation, self).__init__(*args, **kwargs)
        if variables is None:
            self.variables = ['name', 'position']
        else:
            self.variables = variables 

    def get_variable(self, robot, varcode):
        aux = varcode.split('@')
        varpath = aux[0]
        aux_obj = robot
        for vp in varpath.split(':'):
            if hasattr(aux_obj, vp):
                aux_obj = getattr(aux_obj, vp)
            else:
                if isinstance(aux_obj, dict) and vp in aux_obj:
                    aux_obj = aux_obj[vp]
        varname = aux[1]
        return getattr(aux_obj, varname)

        
    def update(self, robot):
        pass

    def initialize(self,world):
        for var in range(len(self.variables)):
                self.axis.text(0.1, 0.9 - 0.1*var, self.variables[var], weight="bold")
        self.set_ylim(0,1)
        self.set_xlim(0,1)



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
        if isinstance(new_y, float) or isinstance(new_y, int):
            self.yData[0] = np.r_[self.yData[0,1:], new_y]
        else:
            for i in range(len(self.sectors)):
                self.yData[i] = np.r_[self.yData[i,1:], new_y[self.sectors[i]]]
        for i in range(len(self.yData)):
            self.axis.lines[i].set_ydata(self.yData[i])
            # re-render the artist, updating the canvas state, but not the screen
            self.axis.draw_artist(self.axis.lines[i])

    def initialize(self, world):
        self.set_ylim(-.1, 1.1)
        self.xData = np.arange(self.plot_buffer) 
        self.yData = np.zeros([len(self.sectors), self.xData.shape[0]]) 
        for i in range(len(self.sectors)):
            ln = self.axis.plot(self.xData, self.yData[i], animated=True)
        self.axis.set_title(f'{self.target_sensor} {self.sectors}')
        self.axis.get_xaxis().set_visible(False)

class AnimatedPolarEpuck(AnimatedPlot):
    def __init__(self, *args, sensors=['distance_sensor'], **kwargs):
        super(AnimatedPolarEpuck, self).__init__(*args, **kwargs)
        self.sensors=sensors
        self.colors = {'distance_sensor' : 'k', 'red_light_sensor' : 'r', 'blue_light_sensor' : 'b', 'green_light_sensor' : 'g'}

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

    def initialize(self, world):
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
    def __init__(self, *args, variables= None, **kwargs):
        super(AnimatedCustomLineplot, self).__init__(*args, **kwargs)
        self.variables = variables or []#['virtual_particle@dist_clst_lmark', 'virtual_particle@dist_clst_neighbor', 'virtual_particle@dist_clst_lmark_av']
  
    def get_variable(self, robot, varcode):
        """
        : -> route jump 
        @ -> resource definition
        () -> resource conditions

        """
        aux = varcode.split('@')
        varpath = aux[0]
        aux_obj = robot
        for vp in varpath.split(':'):
            if hasattr(aux_obj, vp):
                aux_obj = getattr(aux_obj, vp)
            else:
                if isinstance(aux_obj, dict) and vp in aux_obj:
                    aux_obj = aux_obj[vp]
        varname = aux[1]
        varaux = varname.split('?')
        varname = varaux[0]
        if len(varaux) == 1: 
            return getattr(aux_obj, varname)
        vargs = varaux[1]
        import re
        regex_keys = r'\{.+?\}|".+?"|\w+'
        regex_par = r'\(.+?\)|".+?"|\w+'
        varname_splt = re.findall(regex_par, vargs)
        indexing = re.findall(r'\(.*?\)', vargs)
        params = re.findall(r'\{.*?\}', vargs)
        if len(indexing) > 0:
            indexing = indexing[0][1:-1]
            if ',' in indexing:
                ivec = np.array([int(i) for i in indexing.split(',') ]) 
            else:
                ivec = int(indexing)
            return getattr(aux_obj, varname)[ivec] if not isinstance(ivec, int) else np.array([getattr(aux_obj, varname)[ivec]])    
        return getattr(aux_obj, varname)

        
    def update(self, robot):
        new_y = [] 
        j = 0
        for i in range(len(self.variables)):
            y = self.get_variable(robot, self.variables[i])
            if isinstance(y, float) or isinstance(y, int):
                new_y.append(y)
                self.yData[j] = np.r_[self.yData[j,1:], y]
                j += 1
            else:
                self.yData[j:j+len(y)] = np.hstack((self.yData[j:j+len(y),1:], np.expand_dims(y, 1)))
                j += len(y)
        for i in range(len(self.yData)):
            self.axis.lines[i].set_ydata(self.yData[i])
            # re-render the artist, updating the canvas state, but not the screen
            self.axis.draw_artist(self.axis.lines[i])
        self.t += 1

    def initialize(self, world):
        self.set_ylim(0,2)
        self.xData = np.arange(self.plot_buffer)
        robot = world.focused_robot()
        size = np.sum([self.get_variable(robot, vv).shape[0] for vv in self.variables]).astype(int)
        self.yData = np.zeros([size, self.xData.shape[0]]) 
        for i in range(self.yData.shape[0]):
            ln = self.axis.plot(self.xData, self.yData[i], animated=True)

class AnimatedCommunicationSpace(AnimatedPlot):
    def __init__(self, *args, H=2, W=2,**kwargs):
        super(AnimatedCommunicationSpace, self).__init__(*args, **kwargs)
        self.H = H 
        self.W = W
    
    def update(self, robot):
        own_state = robot.virtual_particle.state 
        # print(robot.virtual_particle.)
        neigh_states = [vv.state for vv in robot.virtual_particle.neighbors]
        # lmarks = robot.virtual_particle.landmarks # OLD IMPL
        lmarks = [lm.state for lm in robot.virtual_particle.landmarks] # PHYSICS BASED IMPL
        disabled = robot.virtual_particle.disabled_lmarks
        self.axis.collections[0].set_offsets(own_state)
        if len(neigh_states) > 0:
            self.axis.collections[1].set_offsets(neigh_states)
        self.axis.collections[2].set_offsets(lmarks)
        lm_colors = []
        # priorities = robot.virtual_particle.lmark_priorities
        for i in range(len(lmarks)):
            lm_color = 'red'
            if i in disabled:
                lm_color = 'grey'
            else:
                lm_color='blue'
                # lm_color = ['red', 'yellow', 'blue', 'green'][int(priorities[i])]
            lm_colors.append(lm_color)
        # lm_colors = [('grey', 'red')[i not in disabled] for i in range(len(lmarks))] 
        self.axis.collections[2].set_facecolor(lm_colors)
        
        for ln in self.axis.collections:
            self.axis.draw_artist(ln)

    def initialize(self, world):
        h, w =self.H, self.W 
        self.set_ylim(-h/2-.1,h/2+.1)
        self.set_xlim(-w/2-.1,w/2+.1)
        self.axis.scatter([], [], color='r',zorder=102, s=102, animated=True) # Own state
        self.axis.scatter([], [], color='k',zorder=100, s=102, animated=True) # Neigh states
        self.axis.scatter([], [], marker='*', edgecolors='k', s=250, zorder=101, color='r', animated=True) # Lmarks


import string
class AnimatedImage(AnimatedPlot):
    def __init__(self, *args, source=None, **kwargs):
        super(AnimatedImage, self).__init__(*args, **kwargs)
        # assert source is not None
        self.source = source
    
    def update(self, robot):
        # img = robot.sensors['camera'].reading
        img = robot.controller.pattern_detector.pattern_mat
        if img.shape[0] == 0:
            img = np.zeros((1, img.shape[1])).astype(float)
        
        # else:
        #     __import__('pdb').set_trace()
        img = img * 255
        # img = np.random.random((5,7)) * 255
        self.axis.get_children()[0].set_data(img)
        # self.axis.get_children()[0].set_extent((0, 7, len(img), 0))
        self.axis.grid( color='k', linestyle='-', linewidth=2)
        ylabels = [string.ascii_uppercase[i] for i in range(img.shape[0])]
        self.axis.set_yticklabels(ylabels)
        
        self.axis.draw_artist(self.axis.get_children()[0])
        self.axis.draw_artist(self.axis.get_yaxis())
        # for ch in self.axis.lines:
        #     self.axis.draw_artist(ch)

    def initialize(self, world):
        img = np.zeros((2,7)).astype(float)#
        img = np.random.random((2,7)) * 255
        self.axis.imshow(img, cmap='Reds',) 
        # self.axis.grid( color='k', linestyle='-', linewidth=2)
        # self.axis.get_xaxis().set_visible(False)
        self.axis.get_yaxis().set_visible(False)
        self.axis.set_xticklabels(['R', 'G', 'B', 'Size', 'Shape(0)', 'Shape(1)', 'Shape(2)'])
        self.axis.set_xticks(np.arange(7) -0.5)

class AnimatedCamera(AnimatedPlot):
    def update(self, robot):
        img = robot.sensors['camera'].reading
        img = img #* 255
        self.axis.get_children()[0].set_data(img)
        self.axis.draw_artist(self.axis.get_children()[0])

    def initialize(self, world):
        img = np.zeros((200,200)).astype(float)#
        # img = np.random.random((2,7)) * 255
        self.axis.imshow(img) 
        # self.axis.get_yaxis().set_visible(False)
        # self.axis.get_xaxis().set_visible(False)
        




        
class AnimatedNeuralNetwork(AnimatedPlot):
    def __init__(self, *args, **kwargs):
        super(AnimatedNeuralNetwork, self).__init__(*args, **kwargs)
        self.graph_plotted = False
        self.plot_time_series = True 
        self.add_names = False 
        self.interactive = True
        self.rect_data = np.zeros(50) if self.interactive else None
        self.rect_node_name  = None

    def plot_graph(self, ann):
        layers = np.array(ann.input_ensemble_names + [k for k in ann.ensemble_names if k not in ann.motor_ensemble_names] + ann.motor_ensemble_names)
        n_inputs = ann.num_inputs 
        n_outputs = ann.num_motor
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
            pos[k][0] *= 1.5 
            pos[k][1] *= 1.5 
        wmax = 5
        edge_color = [(1 + conn['weight'] / wmax) / 2 for conn in ann.graph['synapses'].values()]
        nx.draw_networkx_edges(G, pos, ax=self.axis)
        nx.draw_networkx_edges(G, pos,alpha=0.5, edge_color=edge_color, ax=self.axis, edge_cmap=plt.cm.RdBu)
        aa = nx.draw_networkx_nodes(G, pos, node_size=500, node_color=['blue']*len(pos),ax=self.axis, )
        self.axis.collections[3].set_edgecolor("#000000")
        self.axis.collections[3].set_linewidth(3)
        self.axis.collections[1].set_linewidth(1.5)
        self.axis.collections[2].set_linewidth(5)
        
        if self.add_names:
            nodes = self.axis.collections[3] 
            input_names = list(ann.graph['inputs'].keys())
            output_names = list(filter(lambda n: ann.is_motor(n), ann.graph['neurons']))
            for i in range(n_inputs): 
                xc = nodes.get_offsets()[i][0] - 0.15
                yc = nodes.get_offsets()[i][1] -0.05
                self.axis.text(xc, yc, input_names[i], weight="bold")
                xlims = self.axis.get_xlim()
            for i in range(n_outputs): 
                # idx = ann.motor_neurons[::-1][i]
                ii = i+1
                data = nodes.get_offsets().data
                data = data[np.argsort(data, axis=0)[:,0]]
                xc = data[-ii][0] + 0.05
                yc = data[-ii][1] -0.05
                self.axis.text(xc, yc, output_names[i], weight="bold",zorder=98)
            xlims = self.axis.get_xlim()
            self.axis.set_xlim(xlims[0]-0.2, xlims[1] + 0.2) 

            
        if self.interactive:
            def annotate(event):
                x = event.xdata
                y = event.ydata
                cursor_pos = np.r_[x,y]
                node_pos = self.axis.collections[3].get_offsets().data
                node_rad = 0.1 
                tmp = np.linalg.norm(cursor_pos - node_pos,axis=1) < node_rad
                rectangle = self.axis.patches[0] 
                if any(tmp):
                    idx = np.where(tmp)[0][0]
                    self.rect_node_name = idx
                    rect_pos = node_pos[idx].copy()
                    xlims = self.axis.get_xlim()
                    ylims = self.axis.get_ylim()
                    if ylims[1] - rect_pos[1] < rectangle.get_height():
                        rect_pos[1] -= rectangle.get_height()
                    if xlims[1] - rect_pos[0] < rectangle.get_width():
                        rect_pos[0] -= rectangle.get_width()
                    rectangle.set_visible(True)
                    rectangle.set_xy(rect_pos)
                else:
                    rectangle.set_visible(False)
                    self.axis.lines[0].set_data([],[])
                    self.axis.lines[1].set_data([],[])
                   
            self.axis.figure.canvas.mpl_connect('motion_notify_event', annotate)
            w = 0.2 * self.axis.get_data_ratio()
            h = 0.3
            rect = patches.Rectangle((0, 0), h, w, fc=(1,1,1, 1), ec=(0,0,0,1), lw=2, zorder=99)
            rect.set_alpha(0.8)
            rect.set_visible(False)
            self.axis.add_patch(rect) 
            self.axis.plot([],[], lw=2, color='r', zorder=101)
            self.axis.plot([],[], lw=2, color=[0,0,0,0.8], zorder=100)
        if self.plot_time_series:
            self.ydata = [ np.zeros(30) for i in range(n_inputs + n_outputs)]
            for i in range(n_inputs + n_outputs):
                self.axis.plot([],[], color='k', lw=2)
            xlims = self.axis.get_xlim()
            self.axis.set_xlim(np.round(xlims[0]-0.3,2), np.round(xlims[1] + 0.3,2)) 


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
        if self.plot_time_series:
            self.update_time_series(robot)
        if self.interactive:
            rectangle = self.axis.patches[0]
            if rectangle.get_visible():
                rect_pos = rectangle.get_xy() 
                w = rectangle.get_width()
                h = rectangle.get_height()
                if self.rect_node_name < ann.num_inputs:
                    new_data = ann.inputs[self.rect_node_name]
                else:
                    neuron_idx = self.rect_node_name - ann.num_inputs 
                    layer_num = np.argsort([neuron_idx in ann.ensemble_indices(lay) for lay in ann.ensemble_names])[::-1]
                    # neuron_idx 
                    # __import__('pdb').set_trace()
                      
                    new_data = ann.voltages[neuron_idx] / 3 
                new_data *= h/2
                self.rect_data = np.r_[self.rect_data[1:], new_data]
                self.axis.lines[0].set_ydata(self.rect_data + rect_pos[1] + h/2)
                self.axis.lines[0].set_xdata(np.linspace(rect_pos[0], rect_pos[0] + w, 50))
                self.axis.lines[1].set_data([rect_pos[0], rect_pos[0] + w], [rect_pos[1]+h/2,rect_pos[1]+h/2])
                # self.axis.lines[1].set_xdata(np.linspace(rect_pos[0], rect_pos[0] + w, 50))
            else:
                self.rect_data = np.zeros(50)

        for ln in self.axis.collections:
            self.axis.draw_artist(ln)
        for p in self.axis.patches:
            self.axis.draw_artist(p)
        for ln in self.axis.lines:
            self.axis.draw_artist(ln)
        if self.add_names:
            for tx in self.axis.texts:
                self.axis.draw_artist(tx)

    def update_time_series(self, robot):
        ann = robot.controller.neural_network
        n_inputs = ann.num_inputs 
        n_outputs = ann.num_motor
        nodes = self.axis.collections[3] 
        loffset = 2 if self.interactive else 0
        for i in range(n_inputs):
            xc = nodes.get_offsets()[i][0]
            yc = nodes.get_offsets()[i][1]
            xdat = xc + np.linspace(-0.3, -0.1, 30)
            ynew = 0.2 * robot.controller.neural_network.inputs[i] 
            self.ydata[i] = np.r_[self.ydata[i][1:],ynew]
            ydat = yc + self.ydata[i]
            self.axis.lines[i + loffset].set_xdata(xdat)
            self.axis.lines[i + loffset].set_ydata(ydat)
        for i in range(n_outputs):
            idx = ann.motor_neurons[::-1][i]
            ii = i+1
            xc = nodes.get_offsets()[-ii][0]
            yc = nodes.get_offsets()[-ii][1]
            xdat = xc + np.linspace(0.1, 0.3, 30)
            ynew = 0.1 * np.round(robot.controller.neural_network.spikes[idx],3)
            self.ydata[-ii] = np.r_[self.ydata[-ii][1:],ynew]
            ydat = yc + self.ydata[-ii]
            self.axis.lines[-ii].set_xdata(xdat)
            self.axis.lines[-ii].set_ydata(ydat)
        

    def initialize(self, world):
        self.axis.scatter([], [], color='k', s=103, ) # Own state
        self.axis.get_xaxis().set_visible(False)
        self.axis.get_yaxis().set_visible(False)
        # self.axis.scatter([], [], color='k',zorder=100, s=102, animated=True) # Neigh states
        # self.axis.scatter([], [], edgecolors='k', s=250, zorder=101, color='r', animated=True) # Lmarks

class AnimatedEventPlot(AnimatedPlot):
    def __init__(self, *args, **kwargs):
        super(AnimatedEventPlot, self).__init__(*args, **kwargs)
    
    def update(self, robot): 
        robot_ctrl = robot.controller
        names = [*robot_ctrl.priorities.keys()]
        names.sort(key=robot_ctrl.priorities.get)
        for i, k in enumerate(names):
            if robot_ctrl.routines[k].flag:
                ecl = self.axis.collections[i]
                data = ecl.get_positions()
                data.append(robot.t)
                ecl.set_positions(data[-self.plot_buffer:])
                break

        self.axis.set_xlim(robot.t-self.plot_buffer-10, robot.t + 10)
        for coll in self.axis.collections:
            self.axis.draw_artist(coll)
        self.axis.draw_artist(self.axis.yaxis)

    def initialize(self, world):
        robot = world.focused_robot()
        ctrl = robot.controller 
        n_ev = len(ctrl.routines)
        lineoffsets = np.arange(1, n_ev * 2,2).tolist()
        self.axis.eventplot(np.array([[-10]* n_ev]).T, colors=[f'C{i}' for i in range(n_ev)], lineoffsets=lineoffsets, linelength=1, linewidth=10)
        self.axis.get_xaxis().set_visible(False)
        # self.axis.get_yaxis().set_visible(False)
        self.axis.set_xlim(0, self.plot_buffer)
        self.axis.set_yticks(lineoffsets, ctrl.routines.keys()) 

class AnimatedGraph(AnimatedPlot):
    def __init__(self, *args, **kwargs):
        super(AnimatedGraph, self).__init__(*args, **kwargs)
        self.graph_plotted = False
        self.interactive = True

    def plot_graph(self, G, robot):
        if len(G.nodes) == 0:
            return
        H, W = 5, 2.6 
        pos = nx.multipartite_layout(G, subset_key="layer")
        for i in pos.keys():
            pos[i] *= 10 
        pos_np = np.stack([*pos.values()])
        self.axis.set_ylim(pos_np[:,1].min() - 5, pos_np[:,1].max()+6)
        self.axis.set_xlim(pos_np[:,0].min() - 3, pos_np[:,0].max()+3)
        for v in self.axis.patches + self.axis.lines + self.axis.texts:
            v.remove()
        n_patches = len(self.axis.patches)
        n_nodes = len(pos)
        for i in range(n_nodes):
            pi = pos[i+1]
            text = nx.get_node_attributes(G, "lab")[i+1]
            self.axis.text(pi[0], pi[1]+H/2, text, fontweight='bold',fontsize='large')
            st = robot.controller.lexicon.words[i].copy().round(2)
            self.axis.text(pi[0], pi[1]+H/2-2, f"({st[0]}, {st[1]})", fontweight='bold',fontsize='large')
            rect = patches.Rectangle(pi, W, H, fc=(1,1,1, 1), ec=(0,0,0,1), lw=2, zorder=99)
            self.axis.add_patch(rect)
        n_edges = len(G.edges)
        if n_edges == 0:
            return
        for i in range(n_edges):
            edge = [*G.edges][i]
            p1, p2 = pos[edge[0]], pos[edge[1]] 
            self.axis.plot([p1[0] +   2, p2[0]], [p1[1] + H/2, p2[1]+ H/2], lw=2, color='k')

    def update(self, robot):
        graph = robot.controller.lexicon.nx_tree
        if not self.graph_plotted:
            self.plot_graph(graph, robot)
        for v in self.axis.patches + self.axis.lines + self.axis.texts:
            self.axis.draw_artist(v)

    def initialize(self, world):
        # self.axis.scatter([], [], color='k', s=103, ) # Own state
        self.axis.get_xaxis().set_visible(False)
        self.axis.get_yaxis().set_visible(False)
        self.axis.set_xlim(-4,8)
        self.axis.set_ylim(-8,8)
        # self.axis.scatter([], [], color='k',zorder=100, s=102, animated=True) # Neigh states
        # self.axis.scatter([], [], edgecolors='k', s=250, zorder=101, color='r', animated=True) # Lmarks

class AnimatedEventPlot(AnimatedPlot):
    def __init__(self, *args, **kwargs):
        super(AnimatedEventPlot, self).__init__(*args, **kwargs)
    
    def update(self, robot): 
        robot_ctrl = robot.controller
        names = [*robot_ctrl.priorities.keys()]
        names.sort(key=robot_ctrl.priorities.get)
        for i, k in enumerate(names):
            if robot_ctrl.routines[k].flag:
                ecl = self.axis.collections[i]
                data = ecl.get_positions()
                data.append(robot.t)
                ecl.set_positions(data[-self.plot_buffer:])
                break

        self.axis.set_xlim(robot.t-self.plot_buffer-10, robot.t + 10)
        for coll in self.axis.collections:
            self.axis.draw_artist(coll)
        self.axis.draw_artist(self.axis.yaxis)

    def initialize(self, world):
        robot = world.focused_robot()
        ctrl = robot.controller 
        n_ev = len(ctrl.routines)
        lineoffsets = np.arange(1, n_ev * 2,2).tolist()
        self.axis.eventplot(np.array([[-10]* n_ev]).T, colors=[f'C{i}' for i in range(n_ev)], lineoffsets=lineoffsets, linelength=1, linewidth=10)
        self.axis.get_xaxis().set_visible(False)
        # self.axis.get_yaxis().set_visible(False)
        self.axis.set_xlim(0, self.plot_buffer)
        self.axis.set_yticks(lineoffsets, ctrl.routines.keys()) 

class AnimatedVirtualForces(AnimatedPlot):
    def __init__(self, *args, sensors=['distance_sensor'], **kwargs):
        super(AnimatedVirtualForces, self).__init__(*args, **kwargs)

    def update(self, robot):
        head_ori = robot.orientation[-1]
        # arrow = self.axis.get_children()[0]
        # arrow.set_data(x=0, y=0, dx=np.cos(head_ori), dy=np.sin(head_ori))
        # aux_s = 'distance_sensor' if 'distance_sensor' in self.sensors else 'light_sensor'
        # sensor_dirs = robot.sensors[aux_s].directions(head_ori)
        # for j in range(len(self.sensors)):
        #     sensor = self.sensors[j]
        #     for i in range(8):
        #         if 'light_sensor' in sensor: 
        #             new_y = robot.sensors['light_sensor'].reading[sensor]
        #         else:
        #             new_y = robot.sensors[self.sensors[j]].reading
        #         self.axis.lines[j].set_ydata(np.r_[new_y, new_y[0]])
        #         self.axis.lines[j].set_xdata(np.r_[sensor_dirs, sensor_dirs[0]])
        # for i in range(8):
        self.axis.lines[0].set_xdata([head_ori, head_ori])
        self.axis.lines[0].set_ydata([0, 1])
        ctrl = robot.controller 
        nforces = len(ctrl.routines)
        for i, force in enumerate(ctrl.forces.values()):
            fangle = compute_angle(force)
            self.axis.lines[i+1].set_xdata([fangle, fangle])
            self.axis.lines[i+1].set_ydata([0, 1])


            
        # self.axis.lines[len(self.sensors)].set_xdata(np.array([head_ori, head_ori]))
        # self.axis.lines[len(self.sensors)].set_ydata(np.array([0,0.1]))
        for ln in self.axis.lines:
            self.axis.draw_artist(ln)

    def initialize(self, world):
        robot = world.focused_robot()
        ctrl = robot.controller 
        nforces = len(ctrl.routines)
        
        self.axis.set_rorigin(-0.1)
        self.set_ylim(0,1)
        self.axis.plot([0,0], [0,0], color='k', lw=2)
        for i in range(nforces):
            self.axis.plot([0,0], [0,0], color=f'C{i}', lw=2)

    @property
    def subplot_kw(self):
        return {self.name : {'projection' : 'polar'}}

class AnimatedLayout:

    def __init__(self, figsize=None):
        self.plots = []
        self.figsize = figsize 
        self.px_col = 3 
        self.px_row =2.8 
        self.fig = None # plt.figure(figsize=self.figsize)
        self.grid = None #"AA;BC"
    
    def initialize(self, world):
        from mereli.utils import merge_dicts
        subplot_kw = merge_dicts([p.subplot_kw for p in self.plots])
        if self.figsize is None:
            ncols = len(self.grid.split(';'))
            nrows = len(self.grid.split(';')[0])
            self.figsize = (nrows * self.px_row, ncols* self.px_col) 
            if nrows == 1 and ncols == 1:
                self.figsize = (7,7)
        self.fig, axes = plt.subplot_mosaic(self.grid, per_subplot_kw=subplot_kw, figsize=self.figsize,)
        plt.subplots_adjust(top=0.97, bottom=0.08, left=0.1, right=0.97, hspace=0.3, wspace=0.2)
        for p in self.plots:
            p.axis = axes[p.name]
            # p.(self.fig, self.gs)
            p.initialize(world)

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
        elif plot_type == 'animated_graph':
            new_plot = AnimatedGraph(name, **kwargs)
        elif plot_type == 'animated_image':
            new_plot = AnimatedImage(name, **kwargs)
        elif plot_type == 'animated_camera':
            new_plot = AnimatedCamera(name, **kwargs)
        elif plot_type == 'animated_event_plot':
            new_plot = AnimatedEventPlot(name, **kwargs)
        elif plot_type == 'animated_virtual_forces':
            new_plot = AnimatedVirtualForces(name, **kwargs)
        elif plot_type == 'general_info':
            new_plot = GeneralInformation(name, **kwargs)
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
