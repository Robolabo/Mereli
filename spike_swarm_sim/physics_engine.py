import time
import numpy as np
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc
from spike_swarm_sim.globals import global_states
from spike_swarm_sim.objects import Wall

class PybulletEngine:
    def __init__(self):
        self.connected = True
        self.render = global_states.RENDER
        self.engine = bc.BulletClient(connection_mode=p.GUI if self.render else p.DIRECT)
        self.engine.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.engine.setGravity(0, 0, -9.8)
        self.planeId = p.loadURDF("plane.urdf", physicsClientId=self.engine._client)
        self.gui_params = {}
        if self.render:
            self.gui_params['robot_focus'] = self.engine.addUserDebugParameter('Robot focus', 1, -1, 1)
            self.engine.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=30,\
                    cameraPitch=-60, cameraTargetPosition=[0, 0, 0])
    # def id(self):

    def step_physics(self):
        self.engine.stepSimulation(physicsClientId=self.engine._client)

    def step_render(self):
        # if self.physics_client.readUserDebugParameter(self.gui_params['robot_focus']) == 1:
        #     self.physics_client.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=30,\
        #         cameraTargetPosition=self.robots['robotA_0'].position, cameraPitch=-70)#-60,)
        time.sleep(1/50.)

    def connect(self, objects): #!
        self.engine = bc.BulletClient(connection_mode=p.GUI if self.render else p.DIRECT)
        self.engine.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.engine.setGravity(0, 0, -9.8)
        planeId = p.loadURDF("plane.urdf", physicsClientId=self.engine._client)
        self.add_objects(objects)#!
        self.connected = True
        # self.gui_params = {}
        if self.render:
            # self.gui_params['robot_focus'] = self.physics_client.addUserDebugParameter('Robot focus', 1, -1, 1)
            self.engine.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=30,\
                    cameraPitch=-60, cameraTargetPosition=[0, 0, 0])
        
    def disconnect(self):
        self.engine.disconnect()
        self.connected = False

    def add_objects(self, objects):
        for obj in objects:
            obj.add_physics(self.engine._client)



    #! USELESS?
    def initialize_render(self):
        # self.gui_params['robot_focus'] = self.physics_client.addUserDebugParameter('Robot focus', 1, -1, 1)
        self.engine.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=30,\
                    cameraPitch=-60, cameraTargetPosition=[0, 0, 0])



# class Engine2D:
#     def __init__(self, height=1000, width=1000):
#         self.height = height
#         self.width = width
#         self.world_delay = 1 #! TO BE REMOVED
#         self.render = global_states.RENDER
#         if self.render:
#             self.initialize_render()
        
#     def initialize_render(self):
#         self.root = tk.Tk(className='SpikeSwarmSim')
#         self.root.geometry(str(self.width) + 'x' + str(self.height))
#         self.canvas = tk.Canvas(self.root, height=self.height, width=self.width, bg='grey')
#         self.canvas.pack(side='left')
#         frame = tk.Frame(self.root)
#         frame.pack(side='right')
#         # Create limiting walls
#         self.canvas.create_rectangle(0, 0, 20, self.height, fill='black')
#         self.canvas.create_rectangle(0, 0, self.width, 20, fill='black')
#         self.canvas.create_rectangle(self.width - 20, 0, self.width, self.height, fill='black')
#         self.canvas.create_rectangle(0, self.height - 20, self.width, self.height, fill='black')


#     def step_physics(self):
#         pass

#     def step_render(self):
#         # if self.render_connections:
#         #         self.draw_connections()
#         self.canvas.update()
#         self.root.after(self.world_delay)

#     def connect(self, objects): #!

#     def disconnect(self):

#     def add_objects(self, objects):
