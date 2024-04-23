from .controller import Controller, RobotController


#* Robot Controllers
from .random_movement_controller import RandomMovementController
from .braitenberg import *  
from .basic_obstacle_avoider import BasicObstacleAvoider
from .wall_follower import WallFollowerController
from .aggregation_controller import *
from .neural_controller import NeuralController
from .phototaxis import Phototaxis
from .basic_goto_coords import BasicGOTOCoords
#* Light Controllers
from .light_position_controller import LightOrbitController, LightRndPositionController
from .motor_schemas import * 
from .subsumption import * 
from .forage import ForageCommSpace
from .slam import * 
#* Test Controllers 
from .test import * 
from .movement_patterns import *
# from .flocking_controller import FlockingController
# from .cascade_controller import  CascadeController
from .formation import EllipseFormationController 

from .irin_controllers import *
from .find_sensor_pattern import *  
from .learn_lexicon import * 
