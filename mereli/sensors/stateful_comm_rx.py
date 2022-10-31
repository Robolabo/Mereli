import copy
import numpy as np
from mereli.register import sensor_registry
from mereli.sensors import DirectionalSensor, Sensor
from mereli.objects import Robot
from mereli.neural_networks import NeuralNetwork

@sensor_registry(name='stateful_rx')
class StatefulCommRX(Sensor):
    """ 
    """
    def __init__(self, *args,  range=4, state_dim=5, **kwargs):
        super(StatefulCommRX, self).__init__(*args, **kwargs)
        self.range = range
        self.state_dim = state_dim
        self.attention_network = None
        self.state = None
    
    def build_attention(self, topology):
        self.attention_network = NeuralNetwork(topology['dt'], time_scale=topology['time_scale'],\
        neuron_model=topology['neuron_model'], synapse_model=topology['synapse_model'])
        self.neural_network.build_from_dict(topology)

    def reset(self):
        self.state = np.zeros(self.state_dim)
        if self.attention_network is not None:
            self.attention_network.reset()

    def step(self, neighborhood):
        """ 
        """
        neigh_state = []
        own_state = self.sensor_owner.actuators['stateful_tx'].state
        for obj in neighborhood: 
            if self.sensor_owner.id != obj.id and isinstance(obj, Robot):
                if 'stateful_tx' in obj.actuators:
                    dist = np.linalg.norm(obj.position - self.sensor_owner.position)
                    if dist < self.range:
                        neigh_state.append(obj.actuators['stateful_tx'].state.copy())
        if len(neigh_state) == 0:
            neigh_state = np.array([0] * self.state_dim)
        else:
            if self.attention_network is not None:
                weights = np.hstack([self.attention_network.step({"stateful_rx:state" : st})['weight'] for st in neigh_state])
                if np.isnan(weights.sum()):
                    print(weights,self.attention_network.weights)
                    import pdb; pdb.set_trace()
                neigh_state = np.dot(weights, neigh_state)
               
            else:
                # neigh_state = np.stack(neigh_state)
                #neigh_state = neigh_state[np.random.choice(len(neigh_state))]
                state_diffs = [st - own_state for st in neigh_state]
                neigh_mean = np.mean(neigh_state, 0)
        # state_agg = np.mean(state_diffs, 0)
        state_agg = neigh_mean
        self.state = own_state
        closest_state = state_diffs[np.argmin([np.linalg.norm(st_df) for st_df in state_diffs])] 
        target_points = np.array([[.75, .75], [-.75, -.75], [-.75, .75], [.75, -.75]])
                                
        closest_tar = target_points[np.argmin([np.linalg.norm(pt - own_state) for pt in target_points])] 
        inside_area = np.linalg.norm(closest_tar - own_state) < 0.4
        area_full = np.sum([np.linalg.norm(st - closest_tar) < 0.4 for st in neigh_state]) > 3
        return {'mean_neigh_state' : state_agg,# + np.random.randn(self.state_dim) * 0.05,
                'closest_state' : closest_state,#  + np.random.randn(self.state_dim) * 0.05,
                'closest_target' : closest_tar - own_state, 
                'inside_area' : np.array([int(inside_area)]),
                'area_full' : np.array([int(area_full)]) if inside_area else np.array([0.]),
                'own_state' : own_state}#+ np.random.randn() * 0.05}

@sensor_registry(name='ori_stateful_rx')
class OrientStatefulCommRX(Sensor):
    """ 
    """
    def __init__(self, *args, range=4, state_dim=5, **kwargs):
        super(OrientStatefulCommRX, self).__init__(*args, **kwargs)
        self.range = range
        self.state_dim = state_dim
        self.state = None
        self.target_spots = []

    def reset(self):
        self.state = np.zeros(self.state_dim)
        self.target_spots = []

    def step(self, neighborhood):
        """ 
        """
        neigh_state = []
        neigh_oris = []
        own_state = self.sensor_owner.actuators['ori_stateful_tx'].state
        own_ori = self.sensor_owner.actuators['ori_stateful_tx'].orientation
        for obj in neighborhood: 
            if self.sensor_owner.id != obj.id and isinstance(obj, Robot):
                if 'ori_stateful_tx' in obj.actuators:
                    dist = np.linalg.norm(obj.position - self.sensor_owner.position)
                    if dist < self.range:
                        neigh_state.append(obj.actuators['ori_stateful_tx'].state.copy())
                        neigh_oris.append(obj.actuators['ori_stateful_tx'].orientation)
        if len(neigh_state) == 0:
            neigh_state = 0.0
        else:
            state_diffs = [st - own_state for st in neigh_state]
            neigh_mean = np.mean(neigh_state, 0)
        heading_vec = np.r_[np.cos(own_ori), np.sin(own_ori)]
        # state_agg = np.mean(state_diffs, 0)
        state_agg = neigh_mean
        self.state = own_state
        thresh = 0.2
        closest_state = state_diffs[np.argmin([np.linalg.norm(st_df) for st_df in state_diffs])] 
        # target_points = np.array([[0.866,0.5], [0, 1], [-0.866, 0.5], [-0.866, -0.5], [0, -1], [0.866,-0.5]])
        # target_points = np.array([[0.3, 0.1],[-0.5,.2],[0.3, 0.7],[-0.1, -0.6],[0.9, 0.2],[-0.8, -.5]])
        # # target_points = np.array([[.6, .6], [-.6, -.6], [-.6, .6], [.6, -.6]])
        # target_points = np.array([[ 0.6951128 , -0.68198036],
        #    [ 0.81699608, -0.41782854],
        #    [ 0.30321314, -0.11263831],
        #    [-0.76069942,  0.28863416],
        #    [-0.74204351, -0.9134813 ],
        #    [ 0.85693412,  0.25682784],
        #    [ 0.18767415,  0.68150247],
        #    [-0.59958989, -0.41436703],
        #    [-0.2025987 ,  0.19248337],
        #    [ 0.44274261,  0.5605687 ]])
        target_points = self.target_spots if len(self.target_spots) > 0 else np.array([[0.5,0.5]])
        idle_tars = [not any([np.linalg.norm(st - tar) < thresh for st in neigh_state]) for tar in target_points]
        target_points_av = target_points[idle_tars]
        closest_tar = target_points[np.argmin([np.linalg.norm(pt - own_state) for pt in target_points])] - self.state
        if np.sum(idle_tars) == 0:
            closest_tar_av = closest_tar.copy()
        else:
            closest_tar_av = target_points_av[np.argmin([np.linalg.norm(pt - own_state) for pt in target_points_av])] - self.state


        phi_closest_st = np.arccos(closest_state.dot(heading_vec) / np.linalg.norm(closest_state)) if np.linalg.norm(closest_state) > 0 else 0.0
        phi_closest_tar = np.arccos(closest_tar.dot(heading_vec) / np.linalg.norm(closest_tar)) if np.linalg.norm(closest_tar) > 0 else 0.0
        phi_closest_tar_av = np.arccos(closest_tar_av.dot(heading_vec) / np.linalg.norm(closest_tar_av)) if np.linalg.norm(closest_tar_av) > 0 else 0.0
        dist_closest_st =  np.linalg.norm(closest_state) 
        dist_closest_tar = np.linalg.norm(closest_tar)
        dist_closest_tar_av = np.linalg.norm(closest_tar_av)
        
        inside_area = np.linalg.norm(closest_tar - own_state) < thresh 
        area_full = np.sum([np.linalg.norm(st - closest_tar) < thresh for st in neigh_state]) > 3
        return {'mean_neigh_state' : state_agg,# + np.random.randn(self.state_dim) * 0.05,
                'closest_state' : closest_state,#  + np.random.randn(self.state_dim) * 0.05,
                'closest_target' : closest_tar - own_state, 
                'phi_closest_st' : np.array([phi_closest_st / (2*np.pi)]),
                'phi_closest_tar' : np.array([phi_closest_tar / (2*np.pi)]),
                'dist_closest_st' : np.array([dist_closest_st]), 
                'dist_closest_tar' : np.array([dist_closest_tar]), 
                'dist_closest_tar_av' : np.array([dist_closest_tar_av]), 
                'phi_closest_tar_av' : np.array([phi_closest_tar_av / (2*np.pi)]),
                'inside_area' : np.array([int(inside_area)]),
                'area_full' : np.array([int(area_full)]) if inside_area else np.array([0.]),
                'own_state' : own_state}#+ np.random.randn() * 0.05}



@sensor_registry(name='comm_rx_a')
class CommRXTypeA(Sensor):
    """ 
    """
    def __init__(self, *args,  range=4, n=5, **kwargs):
        super(CommRXTypeA, self).__init__(*args, **kwargs)
        self.range = range
        self.n = n

    def step(self, neighborhood):
        """
        """
        reading = np.zeros(self.n)
        for idx, ent in enumerate(filter(lambda y: issubclass(type(y), Robot), sorted(neighborhood, key=lambda x: x.id))):
            reading[idx] = ent.actuators['comm_tx_a'].msg
        return reading
