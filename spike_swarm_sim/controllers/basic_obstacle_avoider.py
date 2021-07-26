import numpy as np
from spike_swarm_sim.controllers import RobotController
from spike_swarm_sim.register import controller_registry

@controller_registry(name='basic_obstable_avoider')
class BasicObstacleAvoider(RobotController):
    def __init__(self, *args, **kwargs):
        super(BasicObstacleAvoider, self).__init__(*args, **kwargs)

    def step(self, state, reward=0.0):
        sens = 0.15
        action = np.ones(2)
        st_ds = state['distance_sensor3D']
        max_dir = np.argmax(st_ds)
        if st_ds[max_dir] > sens:
            action = np.array({
                0 : [-1,-1],
                1 : [-1, 1],
                3 : [1, -1],
                2: [1, 1]
            }.get(max_dir, [1,1]))
        return {'joint_velocity_actuator' : action}