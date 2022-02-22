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
    #* Create the world
    physics_engine = physics_engines[cfg_dict['world'].get('engine', 'pybullet')](
                        dt=cfg_dict['world'].get('physics_dt', 0.02), 
                        T_control=cfg_dict['world'].get('T_control', 0.1))
    world_cls = worlds[cfg_dict['world'].get('name', 'square_arena')]
    arena_params = cfg_dict['world'].get('arena_params', {})
    world = world_cls(physics_engine, **arena_params)
    if ncpu > 1 or USE_MPI and MPI.COMM_WORLD.Get_size() > 1:
        world = MultiWorldWrapper(max(ncpu, MPI.COMM_WORLD.Get_size()), world)
    world.build_from_dict(cfg_dict['world'], ann_topology=cfg_dict['topology'])
    if cfg_dict['algorithm'] is not None and len(cfg_dict['algorithm']):
        ga_config = cfg_dict['algorithm']
        fitness = fitness_functions[ga_config['fitness_function']]()
        algorithm_cls = algorithms[cfg_dict['algorithm']['name']]
        opt_alg = algorithm_cls(cfg_dict['algorithm']['populations'], world,\
                    population_size=ga_config['population_size'], n_generations=ga_config['generations'],\
                    eval_steps=ga_config['evaluation_steps'], num_evaluations=ga_config['num_evaluations'],\
                    n_processes=ncpu, resume=resume, fitness_fn=fitness, checkpoint_name=cfg_dict["checkpoint_file"])
        #* Run GA
        if not eval:
            opt_alg.run()
        else:
            #* Evaluate after evolution
            opt_alg.evaluate()
    else: #* Non-optimizable simulation
        world.connect()
        world.reset()
        t = 0
        while(True):
            state, action = world.step()
            t += 1
            if t == 1000:
                print(world.robots['robotA_0'].position)
                import pdb; pdb.set_trace()
               
if __name__ == "__main__":
    main()