import numpy as np
from mereli.utils.alg_utils import torus_distance, torus_angle, ring_distance, ring_angle
from mereli.neural_networks import NeuralNetwork 
from mereli.register import comm_space_registry

class VirtualParticle:
    def __init__(self):
        self.state = None
        self.orientation = None
        self.controller = None
        self.real_robot = None
        self.control = None
        self.neighbors = []
        self.dist_clst_neighbor = None 
        self.dist_clst_lmark = None
        self.dist_clst_lmark_av = None 
        self.lmark = None
   
    def attach_to_robot(self, real_robot):
        self.real_robot = real_robot
        self.real_robot.virtual_particle = self

    def set_controller(self, topology):
        self.controller = NeuralNetwork(topology['dt'], time_scale=topology['time_scale'],\
                neuron_model=topology['neuron_model'], synapse_model=topology['synapse_model'])
        self.controller.build_from_dict(topology)
    
    def reset(self):
        self.neighbors = []
        self.controller.reset()
        self.control = None
        self.dist_clst_neighbor = None
        self.dist_clst_lmark = None
        self.dist_clst_lmark_av = None 
        self.lmark = None
    
    def step_control(self, stimuli):
        control = self.controller.step(stimuli)
        self.control = np.array(control['out'])

    def simulate_dynamic_neighborhood(self, base_neighbors):
        min_neighs = max(2, len(base_neighbors) // 4)
        # __import__('pdb').set_trace()
        num_neighbors = np.random.choice(range(min_neighs, len(base_neighbors)))
        random_sample = np.random.choice(len(base_neighbors), size=num_neighbors, replace=False)
        self.neighbors = np.array(list(base_neighbors))[random_sample] 

    @property
    def heading_vector(self):
        return np.r_[np.cos(self.orientation), np.sin(self.orientation)]

class CommunicationSpace: 
    def __init__(self, threshold=0.2, randomize_neighbors=False, a=1, b=1): 
        self.threshold = threshold
        self.randomize_neighbors = randomize_neighbors 
        self.a = a
        self.b = b
        self.dt = 0.1
        self.particles = {}
        self.landmarks = []
        self.t = 1

    def step(self):
        import time 
        for particle in self.particles.values():
            stimuli = self.perceive(particle)
            particle.step_control(stimuli)
        self.step_dynamics()
        self.t += 1

    def perceive(self, particle):
        import time 
        # Aggregate info
        #MAYBE PROPERTY
        if self.randomize_neighbors:
            if self.t == 1 or self.t % 100  == 0:
                particle.neighbors = [neigh.virtual_particle for neigh in particle.real_robot.neighbors]
                particle.simulate_dynamic_neighborhood(particle.neighbors)     
                # particle.neighbors = [neigh.virtual_particle for neigh in particle.real_robot.neighbors]
        else:
            particle.neighbors = [neigh.virtual_particle for neigh in particle.real_robot.neighbors]
        neigh_states = []
        neigh_oris = []
        for ngh in particle.neighbors:
            neigh_states.append(ngh.state.copy())
            neigh_oris.append(ngh.orientation)
        if len(neigh_states) == 0:
            neigh_states = [particle.state.copy()] 

        # Compute closest state and landmark
        clst_state = neigh_states[np.argmin([self.distance(st, particle.state) for st in neigh_states])] 
        sorted_lmarks = np.argsort([self.distance(pt, particle.state) for pt in self.landmarks])

        clst_lmark_idx = sorted_lmarks[0] 
        clst_lmark = self.landmarks[clst_lmark_idx] 
         
        
        # t0 = time.time()
        clst_lmark_av = None
        # ord_lmarks = 
        for lm_idx in sorted_lmarks:
            lm = self.landmarks[lm_idx] 
            is_empty = False
            for st in neigh_states:
                if self.distance(st, lm) < self.threshold:
                    is_empty = True
                    break 
            if is_empty == 0:
                clst_lmark_av = lm.copy()
                break
        if clst_lmark_av is None: 
            clst_lmark_av = clst_lmark.copy()
        # print('New : ', time.time() - t0)


        
        # Obtain distances and angles
        phi_clst_st = self.angle(particle, clst_state)
        phi_clst_lmark = self.angle(particle, clst_lmark)
        phi_clst_lmark_av = self.angle(particle, clst_lmark_av)
        dist_clst_st = self.distance(particle.state, clst_state)
        dist_clst_lmark = self.distance(particle.state, clst_lmark)
        dist_clst_lmark_av = self.distance(particle.state, clst_lmark_av)
        particle.dist_clst_neighbor = dist_clst_st
        particle.dist_clst_lmark = dist_clst_lmark
        particle.dist_clst_lmark_av = dist_clst_lmark_av
        particle.lmark = clst_lmark_idx # np.argmin([self.distance(pt, particle.state) for pt in self.landmarks])
        # Normalize 
        phi_clst_st = np.array([1 / (self.b*phi_clst_st+1)])
        phi_clst_lmark = np.array([1 / (self.b*phi_clst_lmark + 1)])
        phi_clst_lmark_av = np.array([1 / (self.b*phi_clst_lmark_av + 1)])
        dist_clst_st = np.array([1 / (self.a*dist_clst_st + 1)]).flatten()
        dist_clst_lmark = np.array([1 / (self.a*dist_clst_lmark+1)]).flatten()
        dist_clst_lmark_av = np.array([1 / (self.a*dist_clst_lmark_av+1)]).flatten()
        return {
            'neigh_states' : neigh_states,
            'phi_clst_st' : phi_clst_st, 
            'phi_clst_lmark' : phi_clst_lmark, 
            'phi_clst_lmark_av' : phi_clst_lmark_av, 
            'dist_clst_st' : dist_clst_st, 
            'dist_clst_lmark' : dist_clst_lmark, 
            'dist_clst_lmark_av' : dist_clst_lmark_av, 
        }


    def add_particle(self, robot_name, real_robot):
        particle = VirtualParticle()
        particle.attach_to_robot(real_robot)
        self.particles[robot_name] = particle 
    
    def add_landmark(self, position):
        self.landmark.append(position)

    def step_dynamics(self): pass

    def reset(self, seed=None):
        self.t = 1
        self.generate_rnd_lmarks(len(self.particles), self.threshold)
        for particle in self.particles.values():
            particle.reset()
            self.initialize_particle(particle, seed=seed)

    def initialize_particle(self, particle, seed=None):
        pass

    def distance(self, pointA, pointB):
        pass
   
    def angle(self, particle, pointB):
        pass

    def generate_rnd_lmarks(self, n_lmarks, min_dist):
        pass



@comm_space_registry(name='torus2D')
class Torus2dSpace(CommunicationSpace):
    def __init__(self, H=2, W=2, tau_st=10, tau_ori=10, **kwargs):
        super(Torus2dSpace, self).__init__(**kwargs)
        self.H = H 
        self.W = W 
        self.tau_st = tau_st 
        self.tau_ori = tau_ori 

    def step_dynamics(self):
        for particle in self.particles.values():
            control = particle.control
            target_orientation = 2*np.pi*control[0]
            speed = (control[1] + 1) / 2
            particle.orientation += (self.dt / self.tau_ori) * (target_orientation - particle.orientation)
            particle.orientation = np.clip(particle.orientation, a_min=0, a_max=2*np.pi)
            if speed > 0.5:
                particle.state += (self.dt / self.tau_st) * particle.heading_vector

            # Apply torus teleportation
            if particle.state[0] > self.W / 2:
                particle.state[0] -= self.W
            elif particle.state[0] < -self.W / 2:
                particle.state[0] += self.W
            if particle.state[1] > self.H/2:
                particle.state[1] -= self.H
            elif particle.state[1] < -self.H/2:
                particle.state[1] += self.H 
            particle.state = np.clip(particle.state, a_min=-self.H/2, a_max=self.H/2)

    def initialize_particle(self, particle, seed=None):
            particle.state = np.random.uniform(low=(-0.5*self.W / 2, -0.5*self.H / 2), high=(0.5*self.W / 2, 0.5*self.H / 2))
            particle.orientation = np.random.uniform(low=0, high=2*np.pi)

    def distance(self, pointA, pointB):
       return torus_distance(pointA, pointB, H=self.H, W=self.W) 
   
    def angle(self, particle, pointB):
       return torus_angle(particle.state, pointB, ref_vec=particle.heading_vector, H=self.H, W=self.W) 


    def generate_rnd_lmarks(self, n_lmarks, min_dist):
        spc_dim = 2
        points = []
        while len(points) < n_lmarks:
            new_candidate = np.random.uniform(low=-self.H / 2, high=self.H / 2, size=spc_dim)
            if len(points) == 0:
                points.append(new_candidate)
            else:
                distances = np.array([np.linalg.norm(pt - new_candidate) for pt in points])
                if all(distances > min_dist):
                    points.append(new_candidate)
        self.landmarks = np.vstack(points)


@comm_space_registry(name='ring1D')
class Ring1dSpace(CommunicationSpace):
    def __init__(self, L=2, tau_st=10,  **kwargs):
        kwargs['b'] = 10
        super(Ring1dSpace, self).__init__(**kwargs)
        self.L = L 
        self.tau_st = tau_st 

    def step_dynamics(self):
        for particle in self.particles.values():
            control = particle.control
            ori = 1 if control[0] > 0.5 else -1 
            speed = (control[1] + 1) / 2
            if speed > 0.5:
                particle.state += (self.dt / self.tau_st) * ori 

            # Apply ring teleportation
            if particle.state[0] > self.L / 2:
                particle.state[0] -= self.L
            elif particle.state[0] < -self.L / 2:
                particle.state[0] += self.L
            particle.state = np.clip(particle.state, a_min=-self.L/2, a_max=self.L/2).flatten()

    def initialize_particle(self, particle, seed=None):
        particle.state = np.array([np.random.uniform(low=-0.5*self.L / 2, high=0.5*self.L)])
        particle.orientation = np.random.uniform(low=0, high=2*np.pi)

    def distance(self, pointA, pointB):
       return ring_distance(pointA, pointB, L=self.L) 
   
    def angle(self, particle, pointB):
       return ring_angle(particle.state, pointB, ref_vec=particle.heading_vector, L=self.L) 

    def generate_rnd_lmarks(self, n_lmarks, min_dist):
        spc_dim = 1
        points = []
        while len(points) < n_lmarks:
            new_candidate = np.random.uniform(low=-self.L / 2, high=self.L / 2, size=spc_dim)
            if len(points) == 0:
                points.append(new_candidate)
            else:
                distances = np.array([np.linalg.norm(pt - new_candidate) for pt in points])
                if all(distances > min_dist):
                    points.append(new_candidate)
        self.landmarks = np.vstack(points)



@comm_space_registry(name='CPPNSpace')
class CPPNSpace(Torus2dSpace):
    def __init__(self, **kwargs):
        super(CPPNSpace, self).__init__(**kwargs)

    def step_dynamics(self):
        for particle in self.particles.values():
            control = particle.control
            delta_x = 0.01 * control[0]
            delta_y = 0.01 * control[1]

            particle.state[0] += delta_x
            particle.state[1] += delta_y

            # Apply torus teleportation
            # if particle.state[0] > self.W / 2:
            #     particle.state[0] -= self.W
            # elif particle.state[0] < -self.W / 2:
            #     particle.state[0] += self.W
            # if particle.state[1] > self.H/2:
            #     particle.state[1] -= self.H
            # elif particle.state[1] < -self.H/2:
            #     particle.state[1] += self.H 
            particle.state = np.clip(particle.state, a_min=-self.H/2, a_max=self.H/2)

    def step(self):
        for particle in self.particles.values():
            stimuli_all = self.perceive(particle)
            general_control = np.zeros(2)
            for neigh in stimuli_all['neigh_states']:
                stimuli = {'neigh_state' : neigh, 'own_state' : particle.state, 'is_lmark' : np.array([0.0]), 'dist_clst_lmark' : stimuli_all['dist_clst_lmark']} 
                particle.step_control(stimuli)
                # __import__('pdb').set_trace()
                general_control += particle.control
            # for lmark in self.landmarks: 
            #     stimuli = {'neigh_state' : lmark, 'own_state' : particle.state, 'is_lmark' : np.array([1.0])} 
            #     particle.step_control(stimuli)
            #     general_control += particle.control

            particle.control = general_control
        self.step_dynamics()
        self.t += 1

    #def perceive(self, particle):
    #    # Aggregate info
    #    #MAYBE PROPERTY
    #    if self.randomize_neighbors:
    #        if self.t == 1 or self.t % 100  == 0:
    #            particle.neighbors = [neigh.virtual_particle for neigh in particle.real_robot.neighbors]
    #            particle.simulate_dynamic_neighborhood(particle.neighbors)     
    #            # particle.neighbors = [neigh.virtual_particle for neigh in particle.real_robot.neighbors]
    #    else:
    #        particle.neighbors = [neigh.virtual_particle for neigh in particle.real_robot.neighbors]
    #    neigh_states = []
    #    neigh_oris = []
    #    for ngh in particle.neighbors:
    #        neigh_states.append(ngh.state.copy())
    #        neigh_oris.append(ngh.orientation)
    #    if len(neigh_states) == 0:
    #        neigh_states = [particle.state.copy()] 
        
    #    return {
    #        'neigh_states' : neigh_states 
    #    }

    def generate_rnd_lmarks(self, n_lmarks, min_dist):
        self.landmarks = np.vstack([[0.7, 0.7], [-0.7, 0.7], [0,-1]])


