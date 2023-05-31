from mereli.globals import global_states
from mereli import SquareArena 
from mereli.physics_engines import PybulletEngine
from mereli.utils.initializers import RandomUniformInitializer
from mereli.controllers import NeuralController
from mereli.objects import Epuck
from mereli.neural_networks import NeuralNetwork
from mereli.objectives import RewardIntegration 
from mereli.algorithms.evolutionary import GeneticAlgorithm
from mereli.tasks import TaskManager

global_states.set_states(render=False)

n_robots = 5


topology = {
    "main_ann" : {
        "dt" : 0.1,
        "time_scale" : 1,
        "stimuli": {
            "I1" : {"n" : 8, "sensor" : "distance_sensor"},
        },
        "neuron_model" : "rate_model",
        "synapse_model" : "static_synapse",
        "ensembles": {
            "OUT" : {"n" : 2, "params" : {"activation" : "tanh", "gain" : 4, "tau" : 1}}
        },
        "outputs" : {
            "outA" : {"ensemble" : "OUT", "actuator" : "joint_velocity_actuator", "enc": "real"}
        },
        "synapses" :  {
            "I1-OUT" : {"pre":"I1","post":"OUT", "trainable":True, "p":1.0},
        },
        "encoding" : {
            "I1" : {"scheme" : "IdentityEncoding"},
        },
        "decoding" : {
            "outA" : {"scheme" : "IdentityDecoding"}
        },
        "enconding" : {},
        "learning_rule" : {}
    } 
}



#* ---- CREATE AND BUILD WORLD -----
phy_engine = PybulletEngine(dt=0.05, T_control=0.05)
world = SquareArena(phy_engine, height=3, width=3)
task_manager = {
            "total_duration" : 1500,
            "num_slots" : 1,
            "use_done" : False,
            "tasks" : [{"name" : "obstacle_avoidance", "params" : {}}]
}
task_manager = TaskManager(duration=1500, num_slots=1, use_done=False) 
task_manager.add_task('obstacle_avoidance')

# Set initializers
ini_ori = RandomUniformInitializer(n_robots, low=0, high=6.28, size=1, engine='3D', variable='orientations')
ini_pos = RandomUniformInitializer(n_robots, low=[-1, -1], high=[1, 1], size=2, engine='3D',  variable='positions')
world.set_initializer('swarm', ini_pos, initializer_ori=ini_ori)
world.task_manager = task_manager

for i, (pos, ori) in enumerate(zip(ini_pos(), ini_ori())):
    controller = NeuralController()
    controller.add_sensor("distance_sensor", {"n_sectors" : 8, "range" : 0.7})
    controller.add_actuator("joint_velocity_actuator",  {"joint_ids" : [0, 1], "max_velocity" : 8})
    controller.add_ann_from_dict(topology['main_ann'])
    ent = Epuck(pos, ori, controller=controller)
    world.register_entity('swarm_' + str(i), ent, group='swarm')
    __import__('pdb').set_trace()

#* --- CREATE GENETIC ALGORITHM ------
fitness_fn = RewardIntegration()

targets = [{"topology" : "main_ann", "object" : "robotA:controller@neural_network"}]
gene_info = {
	    "main_ann:connections:weight:all" : {
            "normalization" : {"type": "linear", "min_val" : -8, "max_val" : 8}, 
            "initialization" : {"type" : "gaussian", "sigma" : 2},
            "mutation" : {"type" : "gaussian", "mutation_prob": 0.1, "sigma" : 0.05}
        },
	    "main_ann:nodes:bias:all" : {
            "normalization" : {"type": "linear", "min_val" : -8, "max_val" : 8},
            "initialization" : {"type" : "gaussian", "sigma" : 2},
            "mutation" : {"type" : "gaussian", "mutation_prob": 0.1, "sigma" : 0.05}
        }
}
__import__('pdb').set_trace()
opt_alg = GeneticAlgorithm(world, 200, 
            50, targets,
            num_evaluations=1,
            resume=False,
            fitness_fn=fitness_fn,
            **alg_params)
opt_alg.initialize(gene_info, topology)


genetic_alg.run()
world.physics_engine.render = True 
__import__('pdb').set_trace()
with world:
    world.reset()
    for _ in range(1000):
        state, action = world.step()



