import click
import os
import subprocess
import logging
import time
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

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
    experiments = ["irin/HelloWorld.json", "irin/TestWheels.json", "irin/TestContact.json", "irin/TestProximity.json", "irin/TestRedLightSensor.json",
     "irin/TestBlueLightSensor.json", "irin/TestGreenLightSensor.json", "irin/TestLED.json", "irin/TestBattery.json", 
     "irin/TestEncoder.json", "irin/ObstacleAvoidance.json", "irin/SubsumptionLightExp.json", "irin/SubsumptionGarbageExp.json", 
     "irin/MotorSchemas1Exp.json", "irin/MotorSchemas2Exp.json", "irin/NeuronEvoAvoidExp.json", "irin/AStorekeeperExp.json", "irin/SubsumptionLucia.json", "irin/AStorekeeperExpLLM", "irin/AStorekeeperExpLLM2", "irin/AStoreKeeperLLMcentral"]
    if num > len(experiments):
        print('Experiment Code does not exist!')
        exit(0)
    return experiments[num]

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
    print("+------------------------------------------------------------------------------+") 
    print("                          BASIC IRIN EXAMPLES                                   ")
    print("+----------------------+------+------------------------------------------------+") 
    print("| EXPERIMENT           | CODE |      CONFIG FILE LOCATION                      |")
    print("+----------------------+------+------------------------------------------------+") 
    print("| HELLO WORLD          |  0   | mereli/config/irin/HelloWorld.json             |")
    print("+----------------------+------+------------------------------------------------+") 
    print("| TEST WHEELS          |  1   | mereli/config/irin/TestWheels.json             |")
    print("+----------------------+------+------------------------------------------------+") 
    print("| TEST CONTACT         |  2   | mereli/config/irin/TestContact.json            |")
    print("+----------------------+------+------------------------------------------------+") 
    print("| TEST PROXIMITY       |  3   | mereli/config/irin/TestProximity.json          |")
    print("+----------------------+------+------------------------------------------------+") 
    print("| TEST RED             |  4   | mereli/config/irin/TestRedLightSensor.json     |")
    print("| LIGHT SENSOR         |      |                                                |")
    print("+----------------------+------+------------------------------------------------+") 
    print("| TEST BLUE            |  5   | mereli/config/irin/TestBlueLightSensor.json    |")
    print("| LIGHT SENSOR         |      |                                                |")
    print("+----------------------+------+------------------------------------------------+") 
    print("| TEST GREEN           |  6   | mereli/config/irin/TestGreenLightSensor.json   |")
    print("| LIGHT SENSOR         |      |                                                |")
    print("+----------------------+------+------------------------------------------------+") 
    print("| TEST LED             |  7   | mereli/config/irin/TestLED.json                |")
    print("+----------------------+------+------------------------------------------------+") 
    print("| TEST BATTERY         |  8   | mereli/config/irin/TestBattery.json            |")
    print("+----------------------+------+------------------------------------------------+") 
    print("| TEST ENCONDER        |  9   |  mereli/config/irin/TestEncoder.json           |")
    print("+----------------------+------+------------------------------------------------+") 
    print("| BASIC OBSTACLE       | 10   |  mereli/config/irin/ObstacleAvoidance.json     |")
    print("|   AVOIDANCE          |      |                                                |")
    print("+----------------------+------+------------------------------------------------+") 
    print("| SUBSUMPTION LIGHT    | 11   |  mereli/config/irin/SubsumptionLightExp.json   |")
    print("+----------------------+------+------------------------------------------------+") 
    print("| SUBSUMPTION GARBAGE  | 12   |  mereli/config/irin/SubsumptionGarbageExp.json |")
    print("+----------------------+------+------------------------------------------------+") 
    print("| MOTOR SCHEMAS LIGHT  | 13   |  mereli/config/irin/MotorSchemas1Exp.json      |")
    print("+----------------------+------+------------------------------------------------+") 
    print("| MOTOR SCHEMAS        | 14   |  mereli/config/irin/MotorSchemas2Exp.json      |")
    print("|   GARBAGE            |      |                                                |")
    print("+----------------------+------+------------------------------------------------+") 
    print("| EVOLVED OBSTACLE     | 15   |  mereli/config/irin/NeuronEvoAvoidExp.json     |")
    print("|   AVOIDANCE          |      |                                                |")
    print("+----------------------+------+------------------------------------------------+") 
    print("| A STOREKEEPER        | 16   |  mereli/config/irin/AStorekeeperExp.json       |")
    print("+----------------------+------+------------------------------------------------+")
    print("| SUBSUMPTION LUCIA    | 17   |  mereli/config/irin/SubsumptionLucia.json      |")
    print("+----------------------+------+------------------------------------------------+")
    print("| ASTOREKEEPERLLM JSON | 18   |  mereli/config/irin/AStorekeeperExpLLM.json   |")
    print("+----------------------+------+------------------------------------------------+")
    print("| ASTOREKEEPERLLM2 DICT| 19   |  mereli/config/irin/AStorekeeperExpLLM2.json  |")
    print("+----------------------+------+------------------------------------------------+")
    print("| ASTOREKEEPERLLMCENTRAL| 20  |  mereli/config/irin/AStoreKeeperLLMcentral.json|")
    print("+----------------------+------+------------------------------------------------+")

    print("")

def generate_plots(folder):
    """ Función interna para generar las gráficas tras la simulación """
    csv_path = os.path.join(folder, "recorrido_robot.csv")
    if not os.path.exists(csv_path):
        return
    
    df = pd.read_csv(csv_path)
    # Gráfica Trayectoria
    plt.figure(figsize=(8, 8))
    plt.plot(df['x'], df['y'], color='green', alpha=0.6)
    # Punto de inicio en rojo
    plt.scatter(df['x'].iloc[0], df['y'].iloc[0], color='red', s=100, label='Inicio', zorder=5)
    # Punto de fin en azul
    plt.scatter(df['x'].iloc[-1], df['y'].iloc[-1], color='blue', s=100, label='Fin', zorder=5)
    plt.title(f'Trayectoria - {os.path.basename(folder)}')
    plt.savefig(os.path.join(folder, "trayectoria.png"))
    plt.close()

    # Gráfica Batería AZUL
    if 'bat_azul' in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df['step'], df['bat_azul'], color='blue')
        plt.title(f'Batería Azul - {os.path.basename(folder)}')
        plt.savefig(os.path.join(folder, "bateria_azul.png"))
        plt.close()

    # Gráfica Batería ROJA
    if 'bat_roja' in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df['step'], df['bat_roja'], color='red')
        plt.title(f'Batería Roja - {os.path.basename(folder)}')
        plt.savefig(os.path.join(folder, "bateria_roja.png"))
        plt.close()
    print(f"✅ Gráficas generadas automáticamente en {folder}")

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
    
    # if log:
    #     logs_folder = cfg_dict.get('logging', {}).get('file', cfg)
    #     logs_path = os.path.join(os.getcwd(), 'mereli', 'logs', logs_folder)
    #     if not os.path.isdir(logs_path):
    #         os.mkdir(logs_path)
    #     now = datetime.now()
    #     logs_path = os.path.join(logs_path, logs_folder + now.strftime("_%d-%m-%Y_%H:%M:%S")) 
    #     global_states.set_data_logging(logs_path)

    # Set loggings
    # if log:
    #     logs_folder = cfg_dict.get('logging', {}).get('file', cfg)
    #     logs_path = os.path.join(os.getcwd(), 'mereli', 'logs', logs_folder)
    #     if not os.path.isdir(logs_path):
    #         os.mkdir(logs_path)

    # __import__('pdb').set_trace()

    # --- MODIFICACIÓN: Crear carpeta de experimento ---
    now = datetime.now().strftime("%m%d_%H%M%S")
    exp_folder = os.path.join("outputs", f"run_{now}")
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)
    
    # Guardamos la ruta en global_states o una variable para que el controlador la use
    # Por ahora, la pasaremos de forma sencilla
    os.environ["CURRENT_EXP_FOLDER"] = exp_folder 
    print(f"📁 Iniciando experimento en: {exp_folder}")

    # --- Redirigir consola a un archivo log ---
    import sys
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w")
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            pass

        def fileno(self):
            return self.terminal.fileno()

    sys.stdout = Logger(os.path.join(exp_folder, "consola.log"))

    #* Create World
    physics_engine = physics_engines[cfg_dict['world'].get('engine', 'pybullet')](
                        dt=cfg_dict['world'].get('physics_dt', 0.02), 
                        T_control=cfg_dict['world'].get('T_control', 0.1))
    world_cls = worlds[cfg_dict['world'].get('name', 'square_arena')]
    arena_params = cfg_dict['world'].get('arena_params', {})
    world = world_cls(physics_engine, **arena_params)
    world.build_from_dict(cfg_dict['world'], ann_topology=cfg_dict.get('topology', {}))
    
    # Interacción inicial con LLM si es el experimento central
    if hasattr(world, 'llm_shared_data') and world.llm_shared_data is not None:
        print("🧠 [CEREBRO CENTRAL]: Ingresa instrucciones iniciales para el LLM (ej: 'Manda 2 robots a cargar batería y otro a apagar luces'): ")
        user_instructions = input().strip()
        if user_instructions == "":
            user_instructions = "No hay instrucciones específicas. Asigna tareas por defecto."
        # Actualizar el prompt
        world.llm_rules = world.llm_rules.replace("{user_instructions}", user_instructions)
        print(f"🧠 [CEREBRO CENTRAL]: Instrucciones registradas: {user_instructions}")
    
    if log:
        world.config_data_logger(cfg_dict['logging']['data'])
        world.data_logger.set_log_file(cfg_dict.get('logging', {}).get('file', cfg))
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
        topology_name = cfg_dict['virtual_space'].get('controller',{}).get('topology')
        world.create_virtual_space(**cfg_dict['virtual_space'], topology=cfg_dict.get('topology',{}).get(topology_name))

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
                opt_alg.evaluator.world.animated_layout.initialize(world)
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
                world.create_animated_layout(figsize=anim_config.get('figsize'))
                world.animated_layout.add_plots(anim_config.get('plots'), grid=anim_config['grid'])
                world.animated_layout.initialize(world)
        np.random.seed(seed)
        # Interacción inicial con LLM ya se hizo antes
        try: 
            for tr in range(trials):
                world.reset()
                t0 = time.time()
                while (world.t < timesteps):
                    if world.t == timesteps - 1:
                        world.is_done = True
                    state, action = world.step()
                time_elapsed = time.time() - t0 
                # print(np.hstack([rob.position[:2] for rob in world.robots.values()]))
                print(f'Simulation of trial {tr} ended in {time_elapsed} after {timesteps} cycles. ')
        except KeyboardInterrupt:
            print("\n🛑 Simulación interrumpida.")
        finally:
            # Esto se ejecuta SIEMPRE al terminar o al pulsar Ctrl+C
            generate_plots(exp_folder)
if __name__ == "__main__":
    main()

