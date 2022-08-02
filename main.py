import click
import logging
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

@click.command()
@click.option('-R', '--render', default=False, is_flag=True, help='Execute in render mode.')
@click.option('-d', '--debug', default=False, is_flag=True,  help='Execute in debug mode.')
@click.option('-r', '--resume', default=False, is_flag=True,\
        help='Resume optimization stored in the checkpoint settled in the JSON config.')
@click.option('-e', '--eval', default=False, is_flag=True, \
        help='Execute in eval mode. No optimization will be carried out.')
@click.option('-v', '--verbose', default=False, is_flag=True,\
        help='Execute in verbose mode (info msgs enabled).')
@click.option('-n', '--ncpu', default=1, help='Number of CPU cores.')
@click.option('-f', '--cfg', default='default', help='Name of the JSON config. file.')
def main(render, resume, cfg, debug, eval, verbose, ncpu):
    #* Set globals
    global_states.set_states(render=render, eval=eval, debug=debug, info=verbose)
    #* Parse JSON
    cfg_dict = json_parser(cfg)
    
    #* Create World
    physics_engine = physics_engines[cfg_dict['world'].get('engine', 'pybullet')](
                        dt=cfg_dict['world'].get('physics_dt', 0.02), 
                        T_control=cfg_dict['world'].get('T_control', 0.1))
    world_cls = worlds[cfg_dict['world'].get('name', 'square_arena')]
    arena_params = cfg_dict['world'].get('arena_params', {})
    world = world_cls(physics_engine, **arena_params)
    world.build_from_dict(cfg_dict['world'], ann_topology=cfg_dict['topology'])

    # import copy
    # world2 = copy.deepcopy(world)
    # world.connect()
    # world2.connect()
    # print(world.physics_engine.client)
    # print(world2.physics_engine.client)
    # import pdb; pdb.set_trace()

    if cfg_dict['algorithm'] is not None and len(cfg_dict['algorithm']):
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
        if not eval:
            opt_alg.run()
        else:
            #* Evaluate after evolution
            opt_alg.validate()
    else: #* Non-optimizable simulation
        world.connect()
        world.reset()
        while(True):
            state, action = world.step()
        
               
if __name__ == "__main__":
    main()
