from spike_swarm_sim.globals import global_states
from spike_swarm_sim import SquareArena, Engine3D
from spike_swarm_sim.utils.initializers import RandomUniformInitializer
from spike_swarm_sim.controllers import NeuralController
from spike_swarm_sim.objects import Epuck
from spike_swarm_sim.neural_networks import NeuralNetwork
from spike_swarm_sim.objectives import ObstacleAvoidance
from spike_swarm_sim.algorithms.evolutionary import GeneticAlgorithm

global_states.set_states(render=False)

n_robots = 1

#* ---- DESIGN and CREATE NEURAL NET ----
dt = 0.1
ann = NeuralNetwork(dt, neuron_model='rate_model', synapse_model='static_synapse')
ann.add_stimuli('I', 8, sensor='distance_sensor') #* 8 because it receives the distance sensor reading (dim. 8).
ann.add_ensemble('O', 2, activation='tanh') #* 2 neurons to control the actions of the joints
ann.set_motor('O')

ann.add_synapse('I-O', 'I', 'O', weight='random') #* Add synapses between inputs and outputs
ann.add_decoder('IdentityDecoding', 'O', 'A')

#! OJO ESTA LINEA ES TEMPORAL HASTA MEJORAR CODIGO LEARNING RULES
ann.build()
ann.reset()



#* ---- CREATE AND BUILD WORLD -----
phy_engine = Engine3D(dt=0.02, T_control=0.14)
world = SquareArena(phy_engine, height=5, width=5)
# Set initializers
ini_ori = RandomUniformInitializer(n_robots, low=0, high=6.28, size=1, engine='3D', variable='orientations')
ini_pos = RandomUniformInitializer(n_robots, low=[-1, -1], high=[1, 1], size=2, engine='3D',  variable='positions')
world.set_initializer('swarm', ini_pos, initializer_ori=ini_ori)

for i, (pos, ori) in enumerate(zip(ini_pos(), ini_ori())):
    controller = NeuralController()
    controller.add_neural_network(ann, {'A' : 'joint_velocity_actuator'})
    controller.add_sensor("distance_sensor", {"n_sectors" : 8, "range" : 0.7})
    controller.add_actuator("joint_velocity_actuator",  {"joint_ids" : [0, 1], "max_velocity" : 8})
    ent = Epuck(pos, ori, controller=controller)
    world.register_entity('swarm_' + str(i), ent, group='swarm')

populations = {
    "p1" : {
        "objects" : ["synapses:weights:all", "neurons:bias:all",  "neurons:tau:all"],
        "max_vals" : [5,  3, 0.5],
        "min_vals" : [-5, -3, -1],
        "params" : {
            "encoding" : "real",
            "selection_operator" : "nonlin_rank",
            "crossover_operator" : "blxalpha",
            "mutation_operator" : "gaussian",
            "mating_operator" : "random",
            "mutation_prob" : 0.05,
            "crossover_prob" : 0.9,
            "num_elite" : 5
        }
    }
}      

#* --- CREATE GENETIC ALGORITHM ------
fitness_fn = ObstacleAvoidance()
genetic_alg = GeneticAlgorithm(populations, world,
                population_size=10, n_generations=50, 
                eval_steps=100, num_evaluations=1,
                fitness_fn=fitness_fn)

genetic_alg.run()
world.physics_engine.render = True 
import pdb; pdb.set_trace()
with world:
    world.reset()
    for _ in range(1000):
        state, action = world.step()