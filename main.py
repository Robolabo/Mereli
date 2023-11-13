import click
import os
import subprocess
import logging
import time
from datetime import datetime
import numpy as np
try:
    from mpi4py import MPI
    USE_MPI = True
except:
    USE_MPI = False
from mereli import MultiWorldWrapper
from mereli.register import fitness_functions
from mereli.config_parser import json_parser
from mereli.register import algorithms, worlds, physics_engines
from mereli.globals import global_states

def get_irin_exp(num):
    experiments = ["irin/TestWheels.json", "irin/TestContact.json", "irin/TestPoximity.json", "irin/TestRedLightSensor.json",
     "irin/TestBlueLightSensor.json", "irin/TestGreenLightSensor.json", "irin/TestLED.json", "irin/TestBattery.json", 
     "irin/TestEncoder.json", "irin/ObstacleAvoidance.json", "irin/SubsumptionLightExp.json", "irin/SubsumptionGarbageExp.json", 
     "irin/NeuronEvoAvoidExp.json"] 
    if num > len(experiments):
        print('Experiment Code does not exist!')
        exit(0)
    return experiments[num-1]

def print_welcome():
    print("")
    print("WELCOME TO MERELI:\n")
    print("You forgot the configuration file required to properly run an experiment. ") 
    print("In the table below you can find some basic experiments. \n")
    print("In order to run an experiment please execute:\n ")
    print("python main.py -f CODE ")
    print("or")
    print("python main.py -f config_file_path")
    print("")
    print("+------------------+-----+--------------------------------------------+") 
    print("| EXPERIMENT          | CODE|      CONFIG FILE LOCATION           |")
    print("+------------------+-----+--------------------------------------------+") 
    print("| TEST WHEELS         |  1  | mereli/config/irin/TestWheels.json  |")
    print("+------------------+-----+--------------------------------------------+") 
    print("| TEST CONTACT        |  2  | mereli/config/irin/TestContact.json  |")
    print("+------------------+-----+--------------------------------------------+") 
    print("| TEST PROXIMITY      |  3  | mereli/config/irin/TestProximity.json  |")
    print("+------------------+-----+--------------------------------------------+") 
    print("| TEST RED            |  4  | mereli/config/irin/TestRedLightSensor.json  |")
    print("| LIGHT SENSOR        |     |                                     |")
    print("+------------------+-----+--------------------------------------------+") 
    print("| TEST BLUE           |  5  | mereli/config/irin/TestBlueLightSensor.json  |")
    print("| LIGHT SENSOR        |     |                                     |")
    print("+------------------+-----+--------------------------------------------+") 
    print("| TEST GREEN          |  6  | mereli/config/irin/TestGreenLightSensor.json  |")
    print("| LIGHT SENSOR        |     |                                     |")
    print("+------------------+-----+--------------------------------------------+") 
    print("| TEST LED            |  7  | mereli/config/irin/TestLED.json  |")
    print("+------------------+-----+--------------------------------------------+") 
    print("| TEST BATTERY        |  8  | mereli/config/irin/TestBattery.json  |")
    print("+------------------+-----+--------------------------------------------+") 
    print("| TEST ENCONDER       |  9  |  mereli/config/irin/TestEncoder.json           |")
    print("+------------------+-----+--------------------------------------------+") 
    print("| BASIC OBSTACLE      | 10  |  mereli/config/irin/ObstacleAvoidance.json     |")
    print("|   AVOIDANCE         |     |                                                |")
    print("+------------------+-----+--------------------------------------------+") 
    print("| SUBSUMPTION LIGHT   | 11  |  mereli/config/irin/SubsumptionLightExp.json   |")
    print("+---------------------+-----+--------------------------------------------+") 
    print("| SUBSUMPTION GARBAGE | 12  |  mereli/config/irin/SubsumptionGarbageExp.json |")
    print("+---------------------+-----+--------------------------------------------+") 
    print("| EVOLVED OBSTACLE    | 13  |  mereli/config/irin/NeuronEvoAvoidExp.json     |")
    print("|   AVOIDANCE         |     |                                                |")
    print("+---------------------+-----+--------------------------------------------+") 
    print("")


@click.command()
@click.option('-R', '--render', default=False, is_flag=True, help='Execute in render mode.')
@click.option('-d', '--debug', default=False, is_flag=True,  help='Execute in debug mode.')
@click.option('-r', '--resume', default=False, is_flag=True,\
        help='Resume optimization stored in the checkpoint settled in the JSON config.')
@click.option('-e', '--eval', default=False, is_flag=True, \
        help='Execute in eval mode. No optimization will be carried out.')
@click.option('-v', '--verbose', default=False, is_flag=True,\
        help='Execute in verbose mode (info msgs enabled).')
@click.option('-l', '--log', default=False, is_flag=True, help='Log data into a file.')
@click.option('-n', '--ncpu', default=1, help='Number of CPU cores.')
@click.option('-f', '--cfg', default=None, help='Name of the JSON config. file.')
@click.option('-i', '--interactive', default=False,is_flag=True, help='Run in interactive mode.')
def main(render, resume, cfg, debug, eval, verbose, log, interactive, ncpu):
    if cfg is None:
       print_welcome() 
       exit(0)
    # if interactive:
    #     process = subprocess.Popen(["panel", "serve", "--port" , "8086", 'mereli/dashboard/dashboard.py'])
    #* Set globals
    if len(cfg) <=3: # Is an exp code
        if not render and int(cfg) != 11:
            render = True
        cfg = get_irin_exp(int(cfg))
    global_states.set_states(render=render, eval=eval, debug=debug, log=log, info=verbose, interactive=interactive)
    
    #* Parse JSON
    cfg_dict = json_parser(cfg)
    if log:
        logs_folder = cfg_dict.get('logging', {}).get('file', cfg)
        logs_path = os.path.join(os.getcwd(), 'mereli', 'logs', logs_folder)
        if not os.path.isdir(logs_path):
            os.mkdir(logs_path)
        now = datetime.now()
        logs_path = os.path.join(logs_path, logs_folder + now.strftime("_%d-%m-%Y_%H:%M:%S")) 
        global_states.set_data_logging(logs_path)
        # if not os.path.isdir(logs_path):
        #     os.mkdir(logs_path)

    # Set loggings
    # if log:
    #     logs_folder = cfg_dict.get('logging', {}).get('file', cfg)
    #     logs_path = os.path.join(os.getcwd(), 'mereli', 'logs', logs_folder)
    #     if not os.path.isdir(logs_path):
    #         os.mkdir(logs_path)

    # __import__('pdb').set_trace()

    #* Create World
    physics_engine = physics_engines[cfg_dict['world'].get('engine', 'pybullet')](
                        dt=cfg_dict['world'].get('physics_dt', 0.02), 
                        T_control=cfg_dict['world'].get('T_control', 0.1))
    world_cls = worlds[cfg_dict['world'].get('name', 'square_arena')]
    arena_params = cfg_dict['world'].get('arena_params', {})
    world = world_cls(physics_engine, **arena_params)
    world.build_from_dict(cfg_dict['world'], ann_topology=cfg_dict.get('topology', {}))
    if log:
        world.config_data_logger(cfg_dict['logging']['data'])
    if render:
        simulation_config = cfg_dict.get('simulation', {})
        world.start_paused = simulation_config.get('start_paused', False)
    #     if 'animated_layout' in simulation_config:
    #         anim_config = simulation_config.get('animated_layout')
    #         world.create_animated_layout()
    #         world.animated_layout.add_plots(anim_config.get('plots'), grid=anim_config['grid'])
    #         world.animated_layout.initialize()

    # Create virtual space (if any)
    if 'virtual_space' in cfg_dict:
        # is_neural_ctlr = cfg_dict['virtual_space']['controller']['name'] == 'neural_controller'
        topology_name = cfg_dict['virtual_space']['controller'].get('topology')
        world.create_virtual_space(**cfg_dict['virtual_space'], topology=cfg_dict['topology'].get(topology_name))

    # import copy
    # world2 = copy.deepcopy(world)
    # world.connect()
    # world2.connect()
    # print(world.physics_engine.client)
    # print(world2.physics_engine.client)
    # import pdb; pdb.set_trace()

    if cfg_dict.get('algorithm', False) and len(cfg_dict['algorithm']):
        alg_config = cfg_dict['algorithm']
        if alg_config['name'] == 'multi_EA':
            opt_alg = algorithms['multi_EA'](world, alg_config['generations'], alg_config['population_size'], None, 
                        num_evaluations=alg_config['num_evaluations'],
                        resume=resume, fitness_fn=alg_config['fitness_function'], 
                        novelty_search=alg_config.get('novelty_search'), 
                        checkpoint_name=cfg_dict["checkpoint_file"])
            for i, alg_cfg in enumerate(alg_config['algs'].values()):
                alg_cls = algorithms[alg_cfg['name']]
                alg = alg_cls(world, alg_config['generations'], 
                        alg_config['population_size'], alg_cfg['targets'],
                        resume=resume, fitness_fn=alg_config['fitness_function'], 
                        checkpoint_name=cfg_dict["checkpoint_file"]+'_'+str(i+1), **alg_cfg["alg_params"])
                alg.initialize(alg_cfg['gene_info'], cfg_dict['topology'])
                opt_alg.add_algorithm(alg)
        else:
            algorithm_cls = algorithms[cfg_dict['algorithm']['name']]
            opt_alg = algorithm_cls(world, alg_config['generations'], 
                        alg_config['population_size'], alg_config['targets'],
                        num_evaluations=alg_config['num_evaluations'],
                        resume=resume, fitness_fn=alg_config['fitness_function'], 
                        novelty_search=alg_config.get('novelty_search'), 
                        checkpoint_name=cfg_dict["checkpoint_file"], **alg_config["alg_params"])
            # opt_alg.create_world(cfg_dict['world'], ann_config=cfg_dict['topology'])
            opt_alg.initialize(alg_config["gene_info"], cfg_dict['topology'])
        #* Run GA
        if render:
            simulation_config = cfg_dict.get('simulation', {})
            opt_alg.evaluator.world.start_paused = simulation_config.get('start_paused', False)
            if 'animated_layout' in simulation_config and simulation_config['animated_layout'].get('enabled', False):
                anim_config = simulation_config.get('animated_layout')
                opt_alg.evaluator.world.create_animated_layout()
                opt_alg.evaluator.world.animated_layout.add_plots(anim_config.get('plots'), grid=anim_config['grid'])
                opt_alg.evaluator.world.animated_layout.initialize()
        if not eval:
            opt_alg.run()
        else:
            opt_alg.validate()
    else: #* Non-optimizable simulation
        simulation_config = cfg_dict.get('simulation', {})
        seed = simulation_config.get('seed', None)
        world.connect()
        print('Connected!')
        timesteps = simulation_config.get('timesteps', 10000)
        trials = simulation_config.get('trials', 1)
        if render:
            world.start_paused = simulation_config.get('start_paused', False)
            if 'camera_options' in simulation_config:
                world.physics_engine.set_camera_options(**simulation_config['camera_options'])
            if 'animated_layout' in simulation_config and simulation_config['animated_layout'].get('enabled', False):
                anim_config = simulation_config.get('animated_layout')
                world.create_animated_layout()
                world.animated_layout.add_plots(anim_config.get('plots'), grid=anim_config['grid'])
                world.animated_layout.initialize()
        np.random.seed(seed)
        for tr in range(trials):
            world.reset()
            t0 = time.time()
            while (world.t < timesteps):
                state, action = world.step()
            time_elapsed = time.time() - t0 
            print(np.hstack([rob.position[:2] for rob in world.robots.values()]))
            print(f'Simulation of trial {tr} ended in {time_elapsed}')
if __name__ == "__main__":
    main()
