import numpy as np
from mereli.controllers import NeuralController 

class VirtualParticle:
    def __init__(self):
        self.state = None
        self.orientation = None
        self.controller = None
    
    def attach_to_robot(self, real_robot):
        self.real_robot = real_robot
        self.real_robot.virtual_particle = self

    def set_controller(self, topology):
        self.controller = NeuralController()
        self.controller.add_ann_from_dict(topology) 
    
    def step_control(self, neigh_states):
        # Aggregate info
        
        pass

    def step_dynamics(self, control):
        delta_ori = control[0]
        speed = (control[1] + 1) / 2
        self.orientation += (self.dt / self.tau_ori) * (2*np.pi*delta_ori - self.orientation)
        self.orientation = np.clip(self.orientation, a_min=0, a_max=2*np.pi)
        heading_ori = np.r_[np.cos(self.orientation), np.sin(self.orientation)]
        if speed > 0.5:
            self.state += (self.dt / self.tau_st) * heading_ori

    def reset(self):
       self.state = np.random.uniform(-1, 1, size=2)
       self.orientation = np.random.uniform(0, 2*np.pi)

class CommunicationSpace:

    def __init__(self):
        self.particles = {}
        self.lmarks = []

    def add_particle(self, robot_name, real_robot):
        particle = VirtualParticle()
        particle.attach_to_robot(real_robot)
        self.particles[robot_name] = particle 
    
    def add_landmark(self, position):
        self.lmark.append(position)

    def distance(self, pointA, pointB):
        raise NotImplementedError
    

    def step(self):
        for particle in self.particles:
            particle.step_control()
        for particle in self.particles:
            particle.step_dynamics()

    def generate_rnd_lmarks(self, n_lmarks, min_dist, spc_dim):
        points = []
        while len(points) < n_points:
            new_candidate = np.random.uniform(low=-1, high=1, size=spc_dim)
            if len(points) == 0:
                points.append(new_candidate)
            else:
                distances = np.array([np.linalg.norm(pt - new_candidate) for pt in points])
                if all(distances > min_dist):
                    points.append(new_candidate)
        self.lmarks = np.vstack(points)

    def reset(self):
        pass

