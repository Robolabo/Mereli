import numpy as np
from mereli.register import sensor_registry
from mereli.sensors import DirectionalSensor, Sensor
from mereli.objects import Robot

@sensor_registry(name='stateful_rx')
class StatefulCommRX(Sensor):
    """ 
    """
    def __init__(self, *args,  range=4, state_dim=5, **kwargs):
        super(StatefulCommRX, self).__init__(*args, **kwargs)
        self.range = range
        self.state_dim = state_dim

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
                        neigh_state.append(obj.actuators['stateful_tx'].state)
        if len(neigh_state) == 0:
            neigh_state = np.array([0] * self.state_dim)
        else:
            # neigh_state = np.stack(neigh_state)
            # neigh_state = np.mean(neigh_state, 0)
            neigh_state = sorted(neigh_state, key=lambda x: np.sum(x), reverse=True)[0]

        return {'mean_neigh_state' : neigh_state,# + np.random.randn(self.state_dim) * 0.0,
                'own_state' : own_state}# + np.random.randn()* 0.0}