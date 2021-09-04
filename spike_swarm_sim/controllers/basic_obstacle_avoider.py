import numpy as np
from spike_swarm_sim.controllers import RobotController
from spike_swarm_sim.register import controller_registry

@controller_registry(name='basic_obstable_avoider')
class BasicObstacleAvoider(RobotController):
    def __init__(self, *args, **kwargs):
        super(BasicObstacleAvoider, self).__init__(*args, **kwargs)

    def step(self, state, reward=0.0):
        sens = 0.15
        
        st_ds = state['distance_sensor']
        # print(st_ds)
        if any(st_ds[[0,1]] > sens):
            # print('Turn Left')
            action = np.array([1., -1])
        elif any(st_ds[[6,7]] > sens):
            # print('Turn Right')
            action = np.array([-1, 1.])
        else:
            # print('GO straight')
            action = np.array([1., 1.])
        if 'led_actuator' in self.enabled_actuators:
            led_action = st_ds > sens
            return {'joint_velocity_actuator' : action, 'led_actuator' : led_action.astype(int)}
        else:    
            return {'joint_velocity_actuator' : action}