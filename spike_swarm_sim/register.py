from functools import wraps
# Set of registers for easing class and function automatic discovery
world_objects = {}
sensors = {}
actuators = {}
neuron_models = {}
synapse_models = {}
controllers = {}
encoders = {}
decoders = {}
evo_operators = {}
fitness_functions = {}
algorithms = {}
initializers = {}
env_perturbations = {}
receptive_fields = {}


def world_object_registry(*args, **kwargs):
    def wrapper(cls):
        name = (cls.__name__, kwargs['name'])['name' in kwargs.keys()]
        world_objects[name] = cls
        return cls
    return wrapper

def fitness_func_registry(*args, **kwargs):
    def wrapper(cls):
        name = (cls.__name__, kwargs['name'])['name' in kwargs.keys()]
        fitness_functions[name] = cls
        return cls 
    return wrapper 


# def sensor_registry(*args, **kwargs): 
#     def wrapper(cls):
#         name = (cls.__name__, kwargs['name'])['name' in kwargs.keys()]
#         sensors[name] = cls
#         @wraps(cls)
#         def _wrapper(*args, **kwargs):      
#             return cls
#         return _wrapper
#     return wrapper 

def sensor_registry(*args, **kwargs):
    def wrapper(cls):
        name = (cls.__name__, kwargs['name'])['name' in kwargs.keys()]
        sensors[name] = cls
        return cls 
    return wrapper 

def actuator_registry(*args, **kwargs):
    def wrapper(cls):
        name = (cls.__name__, kwargs['name'])['name' in kwargs.keys()]
        actuators[name] = cls
        return cls 
    return wrapper 


def controller_registry(*args, **kwargs):
    def wrapper(cls):
        name = (cls.__name__, kwargs['name'])['name' in kwargs.keys()]
        controllers[name] = cls
        return cls 
    return wrapper 

def neuron_model_registry(*args, **kwargs):
    def wrapper(cls):
        name = (cls.__name__, kwargs['name'])['name' in kwargs.keys()]
        neuron_models[name] = cls
        return cls 
    return wrapper

def synapse_registry(*args, **kwargs):
    def wrapper(cls):
        name = (cls.__name__, kwargs['name'])['name' in kwargs.keys()]
        synapse_models[name] = cls
        return cls
    return wrapper

def algorithm_registry(*args, **kwargs):
    def wrapper(cls):
        name = (cls.__name__, kwargs['name'])['name' in kwargs.keys()]
        algorithms[name] = cls
        return cls 
    return wrapper
 
def encoding_registry(cls):
    encoders[cls.__name__] = cls
    return cls

def decoding_registry(cls):
    decoders[cls.__name__] = cls
    return cls

def evo_operator_registry(*args, **kwargs):
    def decorator(cls):
        name = (cls.__name__, kwargs['name'])['name' in kwargs.keys()]
        evo_operators[name] = cls
        return cls
    return decorator

def initializer_registry(*args, **kwargs):
    def decorator(cls):
        name = (cls.__name__, kwargs['name'])['name' in kwargs.keys()]
        initializers[name] = cls
        return cls
    return decorator

def env_perturbation_registry(*args, **kwargs):
    def decorator(cls):
        name = (cls.__name__, kwargs['name'])['name' in kwargs.keys()]
        env_perturbations[name] = cls
        return cls
    return decorator

def receptive_field_registry(*args, **kwargs):
    def decorator(cls):
        name = (cls.__name__, kwargs['name'])['name' in kwargs.keys()]
        receptive_fields[name] = cls
        return cls
    return decorator