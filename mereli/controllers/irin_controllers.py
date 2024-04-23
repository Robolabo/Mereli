import numpy as np
from mereli.controllers import RobotController
from mereli.register import controller_registry

"""
This module can be used by the IRIN students to code their own robot controllers. 
"""


@controller_registry(name='template_controller')
class TemplateController(RobotController):
    def __init__(self, *args, **kwargs):
        """ 
        The __init__ method is the constructor in python. As it inherits from RobotController, the constructor is optional. 
        It can be used to initialize some new variables used in the coded controller. 
        """
        super(TemplateController, self).__init__(*args, **kwargs)
        self.template_var = 0.0

    def step(self):
        # Executes one iteration of the robot controller. It is run every simulation cycle. 
        # 1. Read the sensors
        
        # 2. Processing and logic of the controller

        # 3. Modify the actuator actions (e.g. the motor wheels) 
        
        # Extra considerations:
        # You can access the current simulation time step (read only variable, do not modify):
        print(self.t)
        # You can access the instance of the robot (try to avoid, only in extreme cases)
        print(self.robot)

    def reset(self):
        # This method is executed once per robot at the beginning of every simulation

        # Reset the controller variables when needed. 
        self.template_var = 0.0
        


