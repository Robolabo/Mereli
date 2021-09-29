import numpy as np
from spike_swarm_sim import SquareArena
from spike_swarm_sim.objects import Epuck
from spike_swarm_sim.physics_engines import PybulletEngine
from spike_swarm_sim.utils.initializers import RandomUniformInitializer, RandomGraphInitializer
from spike_swarm_sim.controllers import RobotController
from spike_swarm_sim.communication import IRCommunication

class WOSPLeader(RobotController):
    def __init__(self, *args, **kwargs):
        super(WOSPLeader, self).__init__(*args, **kwargs)
        self.leader = True
        self.t_ref = 10
        self.t_ping = 5
        self.t_max = 200
        self.timer_main = np.random.randint(1, self.t_max)
        self.timer_ref = self.t_ref
        self.timer_ping = self.t_ping
        self.state = 0 # 0: inactive, 1: active, 2: refractory.
    
    def step(self, state, reward=0.0):
        rx_msg = state['IR_receiver']['msg']
        tx_msg = np.array([0.])
        if self.state == 0:
            if rx_msg == 1:
                self.leader = False     
                self.state = 1
            elif self.timer_main == 0:
                self.timer_main = np.random.randint(1, self.t_max)
                self.leader = True
                self.state = 1
            else:
                self.timer_main = max(0, self.timer_main - 1)
        elif self.state == 1:
            tx_msg = np.array([1.])
            if self.timer_ping == 0:
                self.state = 2
                self.timer_ping = self.t_ping
            self.timer_ping = max(0, self.timer_ping - 1)
        elif self.state == 2:
            if self.timer_ref == 0:
                self.state = 0
                self.timer_ref = self.t_ref
            self.timer_ref = max(0, self.timer_ref - 1)
        led_action = self.leader*np.ones(8)
        return {'IR_transmitter' : tx_msg, 'led_actuator' : led_action}

n_robots = 3
# Create physics engine with 0.02sec of discretization and a period the robot
# control loop of 0.14sec.
phy_engine = PybulletEngine(dt=0.02, T_control=0.14)
world = SquareArena(phy_engine, height=10,  width=10)

ini_ori = RandomUniformInitializer(n_robots, low=0, high=6.28, size=1, engine='3D', variable='orientations')
ini_pos = RandomGraphInitializer(n_robots, max_rad=3, initial_pos=[0, 0], engine='3D',  variable='positions')
# ini_pos = FixedInitializer(n_robots, fixed_values=[[0,0], [0,1]], engine='3D',  variable='positions')
# ini_ori = FixedInitializer(n_robots, fixed_values=[0,0], engine='3D',  variable='orientations')
world.set_initializer('swarm', ini_pos, initializer_ori=ini_ori)
for i, (pos, ori) in enumerate(zip(ini_pos(), ini_ori())):
    # Create controller
    ctlr = WOSPLeader()
    # Add sensors and actuators
    ctlr.add_sensor("IR_receiver", {"n_sectors" : 8, "range" : 1.5, "msg_length" : 1, "selection_scheme" : "random"})
    ctlr.add_actuator("IR_transmitter",  {"range" : 1.5, "msg_length":1})
    ctlr.add_actuator("led_actuator",  {})
    # Create and register epuck
    ent = Epuck(pos, ori, controller=ctlr)
    ent.add_communication(IRCommunication())
    world.register_entity('swarm_' + str(i), ent, group='swarm')

with world:
    world.reset()
    for t in range(1000):
        state, action = world.step()