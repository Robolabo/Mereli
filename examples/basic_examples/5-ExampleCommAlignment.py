import time
import numpy as np
from mereli import SquareArena
from mereli.physics_engines import PybulletEngine
from mereli.objects import Epuck
from mereli.utils.initializers import RandomUniformInitializer, RandomGraphInitializer
from mereli.controllers import RobotController
from mereli.communication import IRCommunication
from mereli.utils import angle_diff
""" EXAMPLE DESCRIPTION:

"""

class BasicAlignment(RobotController):

    def __init__(self, *args, **kwargs):
        super(BasicAlignment, self).__init__(*args, **kwargs)
    
    def step(self, state, reward=0.0):
        rx_msg = state['IR_receiver']['msg']
        if rx_msg == 1:
            rx_ori = state['IR_receiver']['raw_rx_angle']
            tx_ori = state['IR_receiver']['raw_tx_angle']
            rot_vel = angle_diff(rx_ori, tx_ori+ np.pi)
            joint_ac = np.array([rot_vel, -rot_vel]) if rx_ori > tx_ori + np.pi else np.array([-rot_vel, rot_vel])
        else:
            joint_ac = np.array([0,0])
        tx_msg = np.array([1.]) # Send always a 1 (not really used due to purely situated comm.)
        return {'IR_transmitter' : tx_msg, 'joint_velocity_actuator' : .5*joint_ac}

n_robots = 5
# Create physics engine with 0.02sec of discretization and a period the robot
# control loop of 0.14sec.
phy_engine = PybulletEngine(dt=0.02, T_control=0.14)
world = SquareArena(phy_engine, height=10,  width=10)

ini_ori = RandomUniformInitializer(n_robots, low=0, high=6.28, size=1, engine='3D', variable='orientations')
#* To guarantee swarm communication compactness the robots are initialized as a random spatial graph.
ini_pos = RandomGraphInitializer(n_robots, max_rad=2, initial_pos=[0, 0], engine='3D',  variable='positions')
world.set_initializer('swarm', ini_pos, initializer_ori=ini_ori)
for i, (pos, ori) in enumerate(zip(ini_pos(), ini_ori())):
    # Create controller
    ctlr = BasicAlignment()
    # Add sensors and actuators
    ctlr.add_sensor("IR_receiver", {"n_sectors" : 8, "range" : 1.5, "msg_length" : 1, "selection_scheme" : "random"})
    ctlr.add_actuator("IR_transmitter",  {"range" : 1.5, "msg_length":1})
    ctlr.add_actuator("joint_velocity_actuator",  {"joint_ids" : [0, 1], "max_velocity" : 1})
    # Create and register epuck
    ent = Epuck(pos, ori, controller=ctlr)
    ent.add_communication(IRCommunication())
    world.register_entity('swarm_' + str(i), ent, group='swarm')

t0 = time.time()
with world:
    world.reset()
    for t in range(2000):
        state, action = world.step()
print('\nSIMULATION DURATION: ', np.round(time.time() - t0, 4))