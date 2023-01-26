import copy
import numpy as np
from mereli.register import sensor_registry
from mereli.sensors import DirectionalSensor, Sensor
from mereli.objects import Robot
from mereli.neural_networks import NeuralNetwork
from mereli.utils import torus_distance, torus_angle, ring_angle, ring_distance

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


@sensor_registry(name='ori_stateful_rx_new')
class NewOrientStatefulCommRX(Sensor):
    """ 
    """
    def __init__(self, *args, range=4, state_dim=5, use_estimation=True, **kwargs):
        super(NewOrientStatefulCommRX, self).__init__(*args, **kwargs)
        self.range = range
        self.state_dim = state_dim
        self.state = None
        self.landmarks = []
        self.t = 1

    def reset(self):
        self.t = 1
        self.state = np.zeros(self.state_dim)

    def step(self, neighborhood):
        """ 
        """
        neighbors = []
        neigh_states = []
        neigh_oris = []
        own_state = self.sensor_owner.actuators['ori_stateful_tx'].state
        own_ori = self.sensor_owner.actuators['ori_stateful_tx'].orientation
        for obj in neighborhood: 
            if self.sensor_owner.id != obj.id and isinstance(obj, Robot):
                if 'ori_stateful_tx' in obj.actuators:
                    dist = np.linalg.norm(obj.position - self.sensor_owner.position)
                    if dist < self.range:
                        neigh_states.append(obj.actuators['ori_stateful_tx'].state.copy())
                        neigh_oris.append(obj.actuators['ori_stateful_tx'].orientation)
                        neighbors.append(obj)
        if len(neigh_states) == 0:
            neigh_states = 0.0
        else:
            state_diffs = [st - own_state for st in neigh_states]
            neigh_mean = np.mean(neigh_states, 0)
        heading_vec = np.r_[np.cos(own_ori), np.sin(own_ori)]
        state_agg = neigh_mean
        self.state = own_state
        thresh = 0.2
        if len(self.landmarks) == 0: self.landmarks = np.zeros(5) #Provisional #Provisional
        dist_landmarks = np.sum([[np.exp(-50*np.linalg.norm(lmark - st)**2) for lmark in self.landmarks] for st in neigh_states], axis=0)
        own_dist_lmarks = np.array([np.exp(-50*np.linalg.norm(lmark - self.state)**2) for lmark in self.landmarks])
        # if len(self.landmarks) > 0:__import__('pdb').set_trace()
        self.t += 1
        return {'neigh_dist_lmarks' : dist_landmarks,# + np.random.randn(self.state_dim) * 0.05,
                'own_dist_lmarks' : own_dist_lmarks,#  + np.random.randn(self.state_dim) * 0.05,
                'own_state' : own_state}#+ np.random.randn() * 1.05}



@sensor_registry(name='ori_stateful_rx')
class OrientStatefulCommRX(Sensor):
    """ 
    """
    def __init__(self, *args, range=4, state_dim=5, use_estimation=True, **kwargs):
        super(OrientStatefulCommRX, self).__init__(*args, **kwargs)
        self.range = range
        self.state_dim = state_dim
        self.state = None
        self.use_estimation = use_estimation
        self.target_spots = []
        self.swarm_table = {}
        self.t = 1

    def reset(self):
        self.t = 1
        self.state = np.zeros(self.state_dim)
        self.target_spots = []
        self.swarm_table = {}

    def update_table(self, neighbors):
        my_id = self.sensor_owner.id
        # self.swarm_table.update({my_id : {'st' : self.state, 'hops' : 0}})
        for neigh in neighbors:
            self.swarm_table.update({neigh.id : {'st' : neigh.actuators['ori_stateful_tx'].state.copy(), 'hops' : 1, 'timeout' : 5}})

    def merge_tables(self, new_table):
        my_id = self.sensor_owner.id
        setA = set(self.swarm_table.keys())
        setB = set(new_table.keys())
        # for aid in setA.union(setB):
        for aid in new_table.keys():
            if aid == my_id: 
                continue
            hops = new_table[aid]['hops'] + 1
            # if hops > 2:
            #     __import__('pdb').set_trace()
            if aid not in self.swarm_table: # Not in table
                self.swarm_table.update({aid : {'st' : new_table[aid]['st'].copy(), 'hops' : hops, 'timeout' : 5}})
            else:
                if hops < self.swarm_table[aid]['hops']:
                    self.swarm_table.update({aid : {'st' : new_table[aid]['st'].copy(), 'hops' : hops, 'timeout' : 5}})
                elif hops == self.swarm_table[aid]['hops']:
                    self.swarm_table.update({aid : {'st' : new_table[aid]['st'].copy(), 'hops' : hops, 'timeout' : 5}})

    def step(self, neighborhood):
        """ 
        """
        neigh_state = []
        neigh_oris = []
        own_state = self.sensor_owner.actuators['ori_stateful_tx'].state
        own_ori = self.sensor_owner.actuators['ori_stateful_tx'].orientation
        neighbors = []
        for obj in neighborhood: 
            if self.sensor_owner.id != obj.id and isinstance(obj, Robot):
                if 'ori_stateful_tx' in obj.actuators:
                    dist = np.linalg.norm(obj.position - self.sensor_owner.position)
                    if dist < self.range:
                        neigh_state.append(obj.actuators['ori_stateful_tx'].state.copy())
                        neigh_oris.append(obj.actuators['ori_stateful_tx'].orientation)
                        if self.use_estimation and self.t % 5 == 0:    
                            rx_table = obj.sensors['ori_stateful_rx'].swarm_table
                            self.merge_tables(rx_table)
                        neighbors.append(obj)
        if self.use_estimation: 
            self.update_table(neighbors)
            neigh_state = [entry['st'] for entry in self.swarm_table.values()]
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
        closest_state = neigh_state[np.argmin([np.linalg.norm(st_df) for st_df in state_diffs])] 
        
        target_points = self.target_spots if len(self.target_spots) > 0 else 0.5 * np.ones(self.state_dim).reshape(1,-1)
        idle_tars = [not any([np.linalg.norm(st - tar) < thresh for st in neigh_state]) for tar in target_points]
        target_points_av = target_points[idle_tars]
        closest_tar = target_points[np.argmin([np.linalg.norm(pt - own_state) for pt in target_points])] 
        if np.sum(idle_tars) == 0:
            closest_tar_av = closest_tar.copy()
        else:
            closest_tar_av = target_points_av[np.argmin([np.linalg.norm(pt - own_state) for pt in target_points_av])] 
        angle_fn = torus_angle if self.state_dim == 2 else ring_angle
        dist_fn = torus_distance if self.state_dim == 2 else ring_distance
        phi_closest_st = angle_fn(closest_state, self.state, ref_vec=heading_vec) 
        phi_closest_tar = angle_fn(closest_tar, self.state, ref_vec=heading_vec) 
        phi_closest_tar_av = angle_fn(closest_tar_av, self.state, ref_vec=heading_vec) 
        dist_closest_st = dist_fn(closest_state, self.state) 
        dist_closest_tar = dist_fn(closest_tar, self.state) 
        dist_closest_tar_av = dist_fn(closest_tar_av, self.state) 
        if self.state_dim == 2:
            phi_closest_st = np.array([phi_closest_st / (2*np.pi)])
            phi_closest_tar = np.array([phi_closest_tar / (2*np.pi)])
            phi_closest_tar_av = np.array([phi_closest_tar_av / (2*np.pi)])
            dist_closest_st = np.array([dist_closest_st])
            dist_closest_tar = np.array([dist_closest_tar])
            dist_closest_tar_av = np.array([dist_closest_tar_av])
        
        inside_area = np.linalg.norm(closest_tar - own_state) < thresh 
        area_full = np.sum([np.linalg.norm(st - closest_tar) < thresh for st in neigh_state]) > 3
        self.t += 1
        return {'mean_neigh_state' : state_agg,# + np.random.randn(self.state_dim) * 0.05,
                'closest_state' : closest_state,#  + np.random.randn(self.state_dim) * 0.05,
                'closest_target' : closest_tar - own_state, 
                'phi_closest_st' : phi_closest_st, 
                'phi_closest_tar' :phi_closest_tar, 
                'dist_closest_st' : dist_closest_st, 
                'dist_closest_tar' : dist_closest_tar, 
                'dist_closest_tar_av' : dist_closest_tar_av, 
                'phi_closest_tar_av' : phi_closest_tar_av, 
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
