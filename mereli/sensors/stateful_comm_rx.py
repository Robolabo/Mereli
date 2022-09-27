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
        self.state = np.array([0] * self.state_dim)
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
