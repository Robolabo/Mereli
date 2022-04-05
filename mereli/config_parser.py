import os 
import json
import logging
import mereli.register as reg
from mereli.sensors.utils import check_sensor_cfg, autocomplete_sensor_cfg
from mereli.actuators.utils import check_actuator_cfg, autocomplete_actuator_cfg
from mereli.neural_networks.utils import neural_net_checker, autocomplete_neural_net

def ExceptionDuplicates(kv_pairs):
    dct = {}
    for key, val in kv_pairs:
        if key in dct:
           raise Exception(logging.error('Duplicate key "{}" in JSON config. file.'.format(key)))
        else:
            dct.update({key : val})
    return dct

def json_parser(file):
    """ Converts the JSON configuration file into a dictionary. """
    json_path = os.path.join('mereli', 'config', file + '.json')
    with open(json_path) as json_file:
        config_dict = json.load(json_file, object_pairs_hook=ExceptionDuplicates)
    config_dict = config_checker(config_dict)
    config_dict = config_autocompletion(config_dict)
    return config_dict

def config_checker(cfg_dict):
    #* General checks
    #! COmprobar que si neural controller => topology != None
    if 'topology' not in cfg_dict.keys() or cfg_dict['topology'] is None:
        logging.warning('ANN Topology not specified in configuration file.')
    if 'algorithm' not in cfg_dict.keys() or cfg_dict['algorithm'] is None:
        logging.warning('Optimization Algorithm not specified in configuration file.')
    if 'world' not in cfg_dict.keys() or cfg_dict['world'] is None:
        raise Exception(logging.error('World/Environment not specified in configuration file.'))
    
    #!check types
    #* Checker Algorithm
    if 'algorithm' in cfg_dict.keys() and cfg_dict['algorithm'] is not None and len(cfg_dict['algorithm']):
        alg_cfg = cfg_dict['algorithm']
        if 'name' not in alg_cfg.keys():
            raise Exception(logging.error('Algorithm type of not specified.'))
        if alg_cfg['name'] not in reg.algorithms.keys():
            raise Exception(logging.error('The algorithm {} is not currently implemented in the simulator. '\
                'Available algorithms are: {}.'.format(alg_cfg['name'], tuple(reg.algorithms.keys()))))
        if 'fitness_function' not in alg_cfg.keys():
            raise Exception(logging.error('Fitness function not specified.'))
        if alg_cfg['fitness_function'] not in reg.fitness_functions.keys():
            raise Exception(logging.error('The Fitness Function {} is not currently implemented in the simulator. '\
                'Available fitness functions are: {}.'.format(alg_cfg['fitness_function'], tuple(reg.fitness_functions.keys()))))
        for var in ['population_size', 'generations', 'evaluation_steps', 'num_evaluations']:
            if alg_cfg[var] < 1:
                raise Exception(logging.error('Parameter {} of algorithm {} '\
                    'must be greater than 0.'.format(var, alg_cfg['name'])))
       
    #* Checker World
    world_cfg = cfg_dict['world']
    if 'objects' not in world_cfg.keys() or world_cfg['objects'] is None:
        logging.warning('No world entities were specified. Empty World will be created.')
    for obj_name, obj in world_cfg['objects'].items():
        if 'num_instances' not in obj.keys() or obj['num_instances'] is None:
            raise Exception(logging.error('Specify the number of instances of object {}.'.format(obj_name)))
        if obj['num_instances'] < 1:
            raise Exception(logging.error('The number of instances of object {} must be greater than .'.format(obj_name)))
        if 'type' not in obj.keys():
            raise Exception(logging.error('Entity type of {} not specified.'.format(obj_name)))
        # if obj['type'] not in reg.world_objects[world_cfg['engine']].keys():
        #     raise Exception(logging.error('Entity type {} is not implemented. '\
        #         'Available entities are: {}.'.format(obj['type'], tuple(reg.world_objects[world_cfg['engine']].keys()))))
            #! TO BE EXTENDED TO OTHER CONTROLLABLE ENTITIES
            raise Exception(logging.error('The number of instances of entity {} '\
                'must be greater than 0.'.format(obj['type'])))
        if obj['type'] == 'robot':
            if 'controller' not in obj.keys():
                logging.warning('No controller was settled for {}. Dummy controller will be used. '.format(obj_name))
            if obj['controller'] not in reg.controllers.keys():
                logging.warning('Controller {} of entity {} is not implemented. '\
                    'Dummy controller will be used instead.'.format(obj_name, obj['controller']))
            # #! Meter dummy controller
            # for var, possible_vals in zip(['sensors', 'actuators'], [reg.sensors, reg.actuators]):
            #     if var not in obj.keys() or len(obj[var]) == 0 or obj[var] is None:
            #         raise Exception(logging.error('No {} was settled for entity {}.'.format(var[:-1], obj_name)))
            #     else:
            #         for key in obj[var].keys():
            #             if key not in possible_vals.keys():
            #                 raise Exception(logging.error('{} {} is not implemented. Available {} are {}.'\
            #                     .format(var[:-1], key, var, tuple(possible_vals))))
            # Check actuators params
            for act_name, act_params in obj['actuators'].items():
                check_actuator_cfg(act_name, act_params)
            # Check sensors params
            for sens_name, sens_params in obj['sensors'].items():
                check_sensor_cfg(sens_name, sens_params)
            # Check both comm. transmitter and receiver are created
            if 'IR_transmitter' in obj['actuators'].keys() and 'IR_receiver' not in obj['sensors'].keys():
                logging.warning('A communication transmitter was created '\
                    'but no communication receiver was specified. ')
            if 'IR_receiver' in obj['sensors'].keys() and 'IR_transmitter' not in obj['actuators'].keys():
                logging.warning('A communication receiver was created '\
                    'but no communication transmitter was specified. ')
    if 'topology' in cfg_dict.keys() and cfg_dict['topology'] is not None and len(cfg_dict['topology']):
        neural_net_checker(cfg_dict['topology'])
    return cfg_dict  

def config_autocompletion(cfg_dict):
    """ Autocompletes omitted optional fields in the config file to 
    their default values. """
    #* Autocomplete Algorithm config
    if 'algorithm' in cfg_dict.keys() and cfg_dict['algorithm'] is not None and len(cfg_dict['algorithm']):
        alg_dict = cfg_dict['algorithm']
        # Autocomplete alg. paramters to their defaults.
        for var, default in zip(['population_size', 'generations',\
                'evaluation_steps', 'num_evaluations'], [50, 1000, 600, 1]):
            if var not in alg_dict.keys() or alg_dict[var] is None:
                alg_dict[var] = default
        #TODO Autocomplete each population
        # for pop in alg_dict['populations'].values():
        #     #! En el futuro comprobar que el alg hereda de Evolutionary Alg.
        #     for var, default in zip(['selection_operator', 'crossover_operator', 'mutation_operator',\
        #             'mating_operator', 'mutation_prob', 'crossover_prob', 'num_elite'],\
        #             ['nonlin_rank', 'blxalpha', 'gaussian', 'random', 0.05, 1, 1]):
        #         if var not in pop:
        #             pop[var] = default

    #* Autocomplete World config
    world_dict = cfg_dict['world']
    if 'engine' not in world_dict:
        world_dict['engine'] = '2D'
    # Autocomplete world paramters to their defaults.
    for var, default in zip(['world_delay', 'render_connections', 'height', 'width'], [1, False, 1000, 1000]):
        if var not in world_dict.keys() or world_dict[var] is None:
            world_dict[var] = default
    if 'objects' in world_dict.keys():
        for obj in world_dict['objects'].values():
            if obj['type'] == 'robot':
                # Autocomplete sensors' params
                if 'sensors' in obj:
                    for sens_name, sens in obj['sensors'].items():
                        obj['sensors'][sens_name] = autocomplete_sensor_cfg(sens_name, sens)
                # Autocomplete actuators' params
                if 'actuators' in obj:
                    for act_name, act in obj['actuators'].items():
                        obj['actuators'][act_name] = autocomplete_actuator_cfg(act_name, act)
                # Set TX and RX msgs to the same length
                if 'IR_receiver' in obj['sensors'].keys() and 'IR_transmitter' in obj['actuators'].keys():
                    if obj['sensors']['IR_receiver']['msg_length'] != obj['actuators']['IR_transmitter']['msg_length']:
                        obj['actuators']['IR_transmitter']['msg_length'] = obj['sensors']['IR_receiver']['msg_length']
                        logging.warning('The length of communication receiver and communication transmitter messages '\
                            'was not the same. Fixing message to a length of {}'.format(obj['sensors']['IR_receiver']['msg_length']))
                # Check robot env perturbations
                if  obj['type'] == 'robot' and 'perturbations' not in obj.keys():
                    obj['perturbations'] = {}
    if 'topology' in cfg_dict.keys() and cfg_dict['topology'] is not None and len(cfg_dict['topology']):
        cfg_dict['topology'] = autocomplete_neural_net(cfg_dict['topology'])
    return cfg_dict
