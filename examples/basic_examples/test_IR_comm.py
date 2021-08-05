import numpy as np
from spike_swarm_sim.world import World3D
from spike_swarm_sim.objects import Epuck3D, GroundArea
from spike_swarm_sim.utils.initializers import RandomUniformInitializer
from spike_swarm_sim.controllers import RobotController
# from spike_swarm_sim.register import controller_registry


class IRCommController(RobotController):
    def __init__(self, *args, **kwargs):
        super(IRCommController, self).__init__(*args, **kwargs)

    def step(self, state, reward=0.0):
        st_ds = state['distance_sensor']
        rx_msg = state['IR_receiver']['msg']
        rx_ori = state['IR_receiver']['receiving_direction']

        # Plan actions
        tx_frame = {k : v.copy() if isinstance(v, np.ndarray) else v for k, v in state['IR_receiver'].items()}
        tx_frame['msg'] = [1 if state['ground_sensor'] == 1 else  rx_msg]
        action_joints = np.ones(2).astype(float)
        if state['ground_sensor'] == 1:
            action_joints = np.array([0., 0.])
        elif rx_msg == 1.:
            action_joints = np.array({
                (1, 0) : [1,1],
                (0, 1) : [-1, 1],
                (-1, 0) : [-1, -1],
                (0, -1) : [1, -1]
            }.get(tuple(rx_ori), [1,1]))
        else: # Apply basic obstacle avoider
            sens = 0.15
            max_dir = np.argmax(st_ds)
            if st_ds[max_dir] > sens:
                action_joints = np.array({
                    0 : [-1,-1],
                    1 : [-1, 1],
                    3 : [1, -1],
                    2 : [1, 1]
                }.get(max_dir, [1,1]))
        return {'joint_velocity_actuator' : action_joints, 'IR_transmitter' : tx_frame}


        
n_robots = 10
world = World3D(height=10,  width=10)

ini_ori = RandomUniformInitializer(n_robots, low=0, high=6.28, size=1, engine='3D', variable='orientations')
ini_pos = RandomUniformInitializer(n_robots, low=[-3,-3], high=[3, 3], size=2, engine='3D',  variable='positions')
world.set_initializer('swarm', ini_pos, initializer_ori=ini_ori)
for i, (pos, ori) in enumerate(zip(ini_pos(), ini_ori())):
    ctlr = IRCommController()
    ctlr.add_sensor("distance_sensor", {"n_sectors" : 4, "range" : 1})
    ctlr.add_sensor("IR_receiver", {"n_sectors" : 4, "range" : 1, "msg_length" : 1, "selection_scheme" : "random"})
    ctlr.add_sensor("ground_sensor", {})
    ctlr.add_actuator("joint_velocity_actuator",  {"joint_ids" : [0, 1], "max_velocity" : 13})
    ctlr.add_actuator("IR_transmitter",  {"quantize": False, "range" : 1, "msg_length":1})
    ent = Epuck3D(pos, ori, controller=ctlr)
    world.register_entity('swarm_' + str(i), ent, group='swarm')

# Add ground area
ground_area = GroundArea(np.array([0, 3.5, 0]), np.zeros(3), color='grey', radius=1.5)
world.register_entity('grey_area_0', ground_area, group='grey_areas')

# Connect, reset and simulate
world.connect()
world.reset()
while(True):
    state, action = world.step()