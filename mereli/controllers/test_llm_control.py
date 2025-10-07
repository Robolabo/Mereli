import re
import json
import numpy as np
from mereli.controllers import RobotController, BasicObstacleAvoider
from mereli.register import controller_registry
from google import genai
  

@controller_registry(name='trajectory-llm')
class TrajectoryByLLM(RobotController):
    """
    """
    def __init__(self, *args, **kwargs):
        """ 
        The __init__ method is the constructor in python. As it inherits from RobotController, the constructor is optional. 
        It can be used to initialize some new variables used in the coded controller. 
        """
        super(TrajectoryByLLM, self).__init__(*args, **kwargs)
        self.template_var = 0.0
        self.client = genai.Client(api_key="AIzaSyBtddr1pj2IBpYu8xPbtMadDtrF5-Mp-Vg")
        self.prev_action = ""
        self.duration = 0 
        self.planned_actions = [0,0]
     

    def step(self, state, reward=None):
        
        gps= self.get_sensor_reading('gps')
        prompt = """ You are a two-wheeled mobile robot. Specifically the e-puck 2 robot that has two motor wheels 
        that can be controlled each one with actions in the range [-1,1]. Negative values for rotating counter clockwise
        and positive for rotating clockwise. Your task is to describe a 1mx1m square trajectory in a flat terrain. 
        Thus you have to generate the actions for controlling the motor wheels accordingly. Periodically I will ask you 
        the next set of actions that you must execute in order to accomplish the mission. You must also specify the duration 
        of the actions in seconds. 
        To do so, I will give you the current execution time and the previous action. I will give you also the GPS info of your positioning 
        Your response should be in the following json format (filling the gaps in upper case): 
        "{"action_left" : {VALUE_1}, "action_right" : {VALUE_2}, "duration" : {DURATION}}"

        Please in the response avoid any explanation, just the json format above. Dont include any explicit json word or anything 
        that may hinder the direct parsing of the json with code. For example, avoid this type of reponses: 
        ```json\n{"action_left" : XXXXX, "action_right" : YYYYY, "duration" : ZZZZ}\n``` 
        In the previous control cycle you executed: 
        """ + self.prev_action + f" and you are currently positioned in coordinates {gps}"

        dt = 0.01
        self.counter += dt 

        if self.counter >= self.duration:
            response = self.client.models.generate_content(
                  model="gemini-2.0-flash", contents=prompt
            ).text
            response = re.sub('\n', '', response)
            response = re.sub('json', '', response)
            response = re.sub('```', '', response)
            response = re.sub('´´´', '', response)
            print(response)
            print(gps)
            # print(prompt)
            res_json = json.loads(response)
            aL = res_json["action_left"]
            aR = res_json["action_right"]
            self.duration = res_json["duration"]
            self.counter = 0
            self.planned_actions = [aL,aR]
            self.prev_action = response
        self.get_actuator('joint_velocity_actuator').action = np.array(self.planned_actions)

         


    def reset(self):
        self.counter = 0
        self.planned_actions = [0,0]
        self.prev_action = "{action_left : 0.3, action_right : 0.3, duration : UNDEFINED}"
