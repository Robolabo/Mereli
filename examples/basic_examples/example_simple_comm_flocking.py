from spike_swarm_sim.communication.IR_communication import IRCommunication
import numpy as np
from spike_swarm_sim.world import World3D
from spike_swarm_sim.objects import Epuck3D, GroundArea
from spike_swarm_sim.utils.initializers import RandomUniformInitializer
from spike_swarm_sim.controllers import RobotController, BasicObstacleAvoider
from spike_swarm_sim.communication import IRCommunication


class ControllerA(RobotController):
    def __init__(self, *args, **kwargs):
        super(ControllerA, self).__init__(*args, **kwargs)

    def step(self, state, reward=0.0):
        st_ds = state['distance_sensor']
        rx_msg = state['IR_receiver']['msg']
        rx_ori = state['IR_receiver']['receiving_direction']

        # Plan Obstacle Avoider
        sens = 0.15
        max_dir = np.argmax(st_ds)
        action_joints = np.array([1., 1.])
        if st_ds[max_dir] > sens:
            action_joints = np.array({
                0 : [-1,-1],
                1 : [-1, 1],
                3 : [1, -1],
                2 : [1, 1]
            }.get(max_dir, [1,1]))


        # Plan COMM msg
        tx_msg = np.array([1.])
        return {'joint_velocity_actuator' : action_joints/2, 'IR_transmitter' : tx_msg}

class ControllerB(RobotController):
    def __init__(self, *args, **kwargs):
        super(ControllerB, self).__init__(*args, **kwargs)
    
    def step(self, state, reward=0.0):
        st_ds = state['distance_sensor']
        rx_msg = state['IR_receiver']['msg']
        rx_ori = state['IR_receiver']['receiving_direction']

        # Plan actions
        print(rx_msg)
        tx_msg = rx_msg
        action_joints = np.array([1., 1.])
        if rx_msg > 0.:
            action_joints = np.array({
                (1, 0) : [1, 1],
                (0, 1) : [1, -1],
                (-1, 0) : [-1, -1],
                (0, -1) : [-1, 1]
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
        return {'joint_velocity_actuator' : action_joints, 'IR_transmitter' : tx_msg}

n_robots = 5
world = World3D(height=10,  width=10)

ctlrA = ControllerA()
ctlrA.add_sensor("distance_sensor", {"n_sectors" : 4, "range" : 1})
ctlrA.add_sensor("IR_receiver", {"n_sectors" : 4, "range" : 2, "msg_length" : 1, "selection_scheme" : "cyclic"})
ctlrA.add_actuator("joint_velocity_actuator",  {"joint_ids" : [0, 1], "max_velocity" : 13})
ctlrA.add_actuator("IR_transmitter",  {"range" : 2, "msg_length":1})
entA = Epuck3D(np.array([0., 2., 0.]), np.zeros(3), controller=ctlrA)
entA.add_communication(IRCommunication())
world.register_entity('swarmA_0', entA, group='swarmA')

ini_ori = RandomUniformInitializer(n_robots, low=0, high=6.28, size=1, engine='3D', variable='orientations')
ini_pos = RandomUniformInitializer(n_robots, low=[-3,-3], high=[3, 1.5], size=2, engine='3D',  variable='positions')
world.set_initializer('swarm', ini_pos, initializer_ori=ini_ori)
for i, (pos, ori) in enumerate(zip(ini_pos(), ini_ori())):
    # Create controller
    ctlrB = ControllerB()
    # Add sensors and actuators
    ctlrB.add_sensor("distance_sensor", {"n_sectors" : 4, "range" : 1})
    ctlrB.add_sensor("IR_receiver", {"n_sectors" : 4, "range" : 2, "msg_length" : 1, "selection_scheme" : "cyclic"})
    ctlrB.add_actuator("joint_velocity_actuator",  {"joint_ids" : [0, 1], "max_velocity" : 13})
    ctlrB.add_actuator("IR_transmitter",  {"range" : 2, "msg_length":1})
    # Create and register epuck
    entB = Epuck3D(pos, ori, controller=ctlrB)
    entB.add_communication(IRCommunication())
    world.register_entity('swarmB_' + str(i), entB, group='swarmB')


# Connect, reset and simulate
world.connect()
world.reset()
while(True):
    state, action = world.step()