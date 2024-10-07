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
        self.dist_clst_neighbor = None 
        self.dist_clst_lmark = None
        self.dist_clst_lmark_av = None 
        self.lmark = None
        self.valid_lmark = None
        self.disabled_lmarks = []
        self.lmark_memory = None
        self.trace_matrix= None
        self.neigh_mask_vec = None
        self.time_settled = 0

    def update_memory(self, occupation_matrix):
        if self.trace_matrix is None:
            self.trace_matrix = occupation_matrix.copy()
        else:
            # update traces with real info
            self.trace_matrix[self.neigh_mask_vec] = occupation_matrix[self.neigh_mask_vec]
            tau = 2000#500# if len(lmarks_status) < 15 else 500
            # Update traces of robots not in neighborhood
            self.trace_matrix[~self.neigh_mask_vec] += (1/tau) * (-self.trace_matrix[~self.neigh_mask_vec])
            # assert any(self.trace_matrix[~self.neigh_mask_vec])
        # if self.lmark_memory is not None:
        #     tau = 300 if len(lmarks_status) < 15 else 500
        #     self.lmark_memory += (1/tau) * (-self.lmark_memory)
        #     self.lmark_memory[self.lmark] = 1
        #     self.lmark_memory[lmarks_status.astype(bool)] = 1
   
    def attach_to_robot(self, real_robot):
        self.real_robot = real_robot
        self.real_robot.virtual_particle = self

    @property
    def neighbors(self):
        return [robot.virtual_particle for robot in self.real_robot.neighbors]

    def set_controller(self, topology):
        self.controller = NeuralNetwork(topology['dt'], time_scale=topology['time_scale'],\
                neuron_model=topology['neuron_model'], synapse_model=topology['synapse_model'])
        self.controller.build_from_dict(topology)
    
    def reset(self):
        # self.neighbors = []
        if self.controller is not None:
            self.controller.reset()
        self.control = None
        self.dist_clst_neighbor = None
        self.dist_clst_lmark = None
        self.dist_clst_lmark_av = None 
        self.lmark = None
        self.valid_lmark = None
        self.disabled_lmarks = []
        self.lmark_memory = None
        self.time_settled = 0
        self.trace_matrix  = None 

    def step_control(self, stimuli):
        # CTRNN
        control = self.controller.step(stimuli)
        self.control = np.array(control['out'])
        
        #manual
        # d_tar = stimuli['dist_clst_lmark_av']
        # phi_tar = stimuli['phi_clst_lmark_av'] / (2*np.pi)
        # # print(phi_tar, d_tar)
        # sp = 0.5 if d_tar[0] > 0.2 else -1
        # self.control = np.r_[phi_tar, sp]

    def simulate_dynamic_neighborhood(self, base_neighbors):
        min_neighs = max(2, len(base_neighbors) // 4)
        # __import__('pdb').set_trace()
        num_neighbors = np.random.choice(range(min_neighs, len(base_neighbors)))
        random_sample = np.random.choice(len(base_neighbors), size=num_neighbors, replace=False)
        self.neighbors = np.array(list(base_neighbors))[random_sample] 

    @property
    def heading_vector(self):
        return np.r_[np.cos(self.orientation), np.sin(self.orientation)]

    @property
    def id(self):
        return self.real_robot.gid

@comm_space_registry(name='VirtualPhysicsCommSpace')
class CommunicationSpace: 
    def __init__(self, threshold=0.2, randomize_neighbors=False, a=1, b=1): 
        self.num_lmarks = 0 
        self.threshold = threshold
        self.randomize_neighbors = randomize_neighbors 
        self.a = a
        self.b = b
        self.dt = 0.1
        self.particles = {}
        self.landmarks = []
        self.lmk_init_method = 'fixed'
        self.lmarks = [{'idx' : i, 'pos' : None, 'pr' : 0} for i in range(self.num_lmarks)]
        self.lmarks_enabled = []
        self.occupation_matrix = None # mask matrix N_robots x N_lmarks that states if robot i is in lmark j
        self.distance_matrix = None # same as above but with distances between robots and lmarks 
        

    def step(self):
        for particle in self.particles.values():
            self.update_matrices(particle)
        stimuli_all = np.repeat({}, len(self.particles)) 
        for particle in self.particles.values():
            # if not particle.real_robot.awaken:
            #     continue
            stimuli_all[particle.id] = self.perceive(particle)
        # for particle in self.particles.values():
            particle.step_control(stimuli_all[particle.id])
        self.step_dynamics()
        self.t += 1
    
    def update_matrices(self, particle):
        st = particle.state
        pidx = particle.id
        for lm_idx, lmk in enumerate(self.landmarks):
            dist_lm = self.distance(lmk, st)
            self.distance_matrix[pidx, lm_idx] = dist_lm 
        self.occupation_matrix[pidx] = 0
        clst_lm = np.argsort(self.distance_matrix[pidx])[0]
        particle.lmark = clst_lm
        self.occupation_matrix[pidx,clst_lm] = 1 


    def perceive(self, particle):
        import time 
        t0 = time.time()
        neigh_states = []
        neigh_oris = []
        pidx = particle.id
        neigh_mask_vec = np.repeat(False, len(self.particles.keys()))
        neigh_mask_vec[pidx] = True
        # try:
        #     particle.neighbors = [neigh.virtual_particle for neigh in particle.real_robot.neighbors]
        # except:
        #     __import__('pdb').set_trace()
        for ngh in particle.neighbors:
            neigh_mask_vec[ngh.id] = True
            neigh_states.append(ngh.state.copy())
            neigh_oris.append(ngh.orientation)
        particle.neigh_mask_vec = neigh_mask_vec
        if len(neigh_states) == 0:
            neigh_states = [particle.state.copy()] 
        
        # Compute closest state and landmark
        clst_state = neigh_states[np.argmin([self.distance(st, particle.state) for st in neigh_states])] 
        sorted_lmarks = np.argsort(self.distance_matrix[pidx])        
        # sorted_lmarks = np.argsort([self.distance(pt, particle.state) for pt in self.landmarks]) 
        clst_lmark_idx = sorted_lmarks[0]
        clst_lmark = self.landmarks[clst_lmark_idx] 
        # particle.time_settled = particle.time_settled + 1 if particle.lmark == clst_lmark_idx else 0
        # particle.lmark = clst_lmark_idx # np.argmin([self.distance(pt, particle.state) for pt in self.landmarks])
        # assert particle.lmark == np.argmin([self.distance(pt, particle.state) for pt in self.landmarks])
        
        occupied_lmarks = np.array([np.sum([neigh.lmark == lm and neigh.time_settled > 100000 
                                    for neigh in particle.neighbors]) > 0 for lm in range(len(particle.landmarks))]).astype(int)
        particle.update_memory(self.occupation_matrix) 
        # t0 = time.time()
        clst_lmark_av = None
        # ord_lmarks = 
        new_sorted = []
        for i in np.unique(particle.lmark_priorities):
            sorted_lm_priorities = np.array(particle.lmark_priorities)[sorted_lmarks]
            new_sorted.append(sorted_lmarks[sorted_lm_priorities == i])
        sorted_lmarks = np.hstack(new_sorted)
        for lm_idx in sorted_lmarks:
            # if lm_idx in particle.disabled_lmarks:
            #     continue
            # if particle.lmark_priorities[lm_idx] == 3:
            #     continue
            lm = self.landmarks[lm_idx]
            if particle.lmark_priorities[lm_idx] == 0: # Priority=0 allows many lmark visitors
                clst_lmark_av = lm.copy()
                break
            dist_lm = self.distance_matrix[pidx, lm_idx]
            # Aux vector storing the distnaces of neighbors and self that are in region of lm_idx
            aux_vec = self.occupation_matrix[neigh_mask_vec, lm_idx] * self.distance_matrix[neigh_mask_vec, lm_idx]
            # Is empty if no one in lmark region or particle is closest among neighbors.
            is_empty = np.sum(aux_vec) == 0 or np.min(aux_vec[aux_vec > 0]) >= dist_lm 
            # if self.occupation_matrix[:, lm_idx].sum() > 2 and is_empty and self.t > 100:
            #     __import__('pdb').set_trace()
            # if is_empty:
            #     is_empty = np.sum(particle.trace_matrix[~neigh_mask_vec, lm_idx] > 0.1) == 0
            if is_empty:
                clst_lmark_av = lm.copy()
                break
        if clst_lmark_av is None: 
            clst_lmark_av = clst_lmark.copy()
        # print('New : ', time.time() - t0)

        # OJO
        clst_lmark = clst_lmark_av.copy()
        #####
        # print(clst_lmark, clst_lmark_av, self.landmarks[lm_idx])
       
        phi_clst_st = self.angle(particle, clst_state)
        if np.isnan(phi_clst_st):
            phi_clst_st = 0
        phi_clst_lmark = self.angle(particle, clst_lmark)
        phi_clst_lmark_av = self.angle(particle, clst_lmark_av)
        dist_clst_st = self.distance(particle.state, clst_state)
        dist_clst_lmark = self.distance(particle.state, clst_lmark)
        dist_clst_lmark_av = self.distance(particle.state, clst_lmark_av)
        particle.dist_clst_neighbor = dist_clst_st
        particle.dist_clst_lmark = dist_clst_lmark
        particle.dist_clst_lmark_av = dist_clst_lmark_av
        # print(f'Particle {particle.id} is going to {lm_idx} with dist={dist_clst_lmark_av}, {dist_clst_lmark} and in {particle.lmark}')
        # Normalize 
        phi_clst_st = np.array([1 / (self.b*phi_clst_st+1)])
        phi_clst_lmark = np.array([1 / (self.b*phi_clst_lmark + 1)])
        phi_clst_lmark_av = np.array([1 / (self.b*phi_clst_lmark_av + 1)])
        dist_clst_st = np.array([1 / (self.a*dist_clst_st + 1)]).flatten()
        dist_clst_lmark = np.array([1 / (self.a*dist_clst_lmark+1)]).flatten()
        dist_clst_lmark_av = np.array([1 / (self.a*dist_clst_lmark_av+1)]).flatten()
        # print('Perceive time ', time.time()-t0)
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
        self.num_lmarks += 1
        if position is not None: # If None it means random
            self.landmarks.append(position)
        else:
            self.lmk_init_method = 'random'

    def step_dynamics(self): pass

    def reset(self, seed=None):
        self.t = 1
        if self.lmk_init_method == 'random':
            self.generate_rnd_lmarks(self.num_lmarks, self.threshold)
        
        for particle in self.particles.values():
            particle.reset()
            self.initialize_particle(particle, seed=seed)
            particle.landmarks = self.landmarks.copy()
            particle.lmark_memory = np.zeros(len(self.landmarks))
        self.occupation_matrix = np.zeros((len(self.particles), len(self.landmarks)))
        self.distance_matrix = np.zeros((len(self.particles), len(self.landmarks)))
        if isinstance(self.landmarks, list):
            self.landmarks = np.array(self.landmarks)

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
        """ Virtual-Physics approach """
        for i, pi in enumerate(self.particles.values()):
            Ftot = np.zeros(2).astype(float)
            for j, pj in enumerate(self.particles.values()):
                if pi.id != pj.id:
                    r = np.linalg.norm(pi.state - pj.state)
                    Fmod = 0.2 / ((r+0.05) ** 2) 
                    Fdir = (pi.state - pj.state) / r 
                    # phi = self.angle(pi, clst_lmark_av)
                    Ftot += Fmod * Fdir 

            for lm in self.landmarks:
                r = np.linalg.norm(pi.state - lm)
                Fmod = 0.1 / ((r + 0.05) ** 2) 
                Fdir = (lm - pi.state) / r 
                # phi = self.angle(pi, clst_lmark_av)
                Ftot += Fmod * Fdir 
            pi.state += (self.dt / self.tau_st) * Ftot 
            # __import__('pdb').set_trace()
            pi.state[0] = np.clip(pi.state[0], a_min=-self.H/2, a_max=self.H/2)
            pi.state[1] = np.clip(pi.state[1], a_min=-self.H/2, a_max=self.H/2)


    def step_dynamics_true(self):
        for particle in self.particles.values():
            # if not particle.real_robot.awaken:
            #     continue
            control = particle.control
            target_orientation = 2*np.pi*control[0]
            speed = (control[1] + 1) / 2
            particle.orientation += (self.dt / self.tau_ori) * (target_orientation - particle.orientation)
            # particle.orientation = 2*np.pi*2*np.pi*control[0]
            particle.orientation = np.clip(particle.orientation, a_min=0, a_max=2*np.pi)
            if speed > 0.5:
                sp = 5 * particle.dist_clst_lmark 
                particle.state += sp * (self.dt / self.tau_st) * particle.heading_vector

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
                distances = np.array([self.distance(pt, new_candidate) for pt in points])
                if all(distances > min_dist):
                    points.append(new_candidate)

        self.landmarks = np.vstack(points)
        order = np.argsort([self.distance(self.landmarks[0], lm) for lm in self.landmarks])
        self.landmarks = self.landmarks[order]
        self.lmarks_enabled = [True for _ in range(len(self.landmarks))]


    def generate_rnd_lmarks2(self, n_lmarks, min_dist):
        H = int(np.floor(np.sqrt(n_lmarks)))
        W = int(np.ceil(np.sqrt(n_lmarks)))
        dx = 0.5 
        dy = 0.5 
        while(H * W != n_lmarks):
            if H*W > n_lmarks:
                W -= 1
            else:
                H += 1
        x = np.linspace(-dx*W/2, dx*W/2, W) 
        y = np.linspace(-dy*H/2, dy*H/2, H) 
        xx, yy = np.meshgrid(x, y)
        points = []
        for x_i, y_i in zip(xx.flatten(), yy.flatten()):
            points.append(np.array([x_i, y_i]))
        # if self.shuffle:
        #     np.random.shuffle(points)
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

@comm_space_registry(name='none')
class VirtualPhysicsCommSpace(CommunicationSpace):
    def __init__(self, H=2, W=2, tau_st=10, **kwargs):
        super(VirtualPhysicsCommSpace, self).__init__(**kwargs)
        self.H = H 
        self.W = W 
        self.tau_st = tau_st 

    def step(self):
        for particle in self.particles.values():
            self.update_matrices(particle)
        # stimuli_all = np.repeat({}, len(self.particles)) 
        # for particle in self.particles.values():
        #     # if not particle.real_robot.awaken:
        #     #     continue
        #     stimuli_all[particle.id] = self.perceive(particle)
        # # for particle in self.particles.values():
        #     particle.step_control(stimuli_all[particle.id])
        self.step_dynamics()
        self.t += 1
        
    def step_dynamics(self):
        """ Virtual-Physics approach """
        for i, pi in enumerate(self.particles.values()):
            Ftot = np.zeros(2).astype(float)
            for j, pj in enumerate(self.particles.values()):
                if pi.id != pj.id:
                    r = np.linalg.norm(pi.state - pj.state)
                    Fmod = 0.2 / ((r+0.05) ** 2) 
                    Fdir = (pi.state - pj.state) / r 
                    # phi = self.angle(pi, clst_lmark_av)
                    Ftot += Fmod * Fdir 

            for lm in self.landmarks:
                r = np.linalg.norm(pi.state - lm)
                Fmod = 0.1 / ((r + 0.05) ** 2) 
                Fdir = (lm - pi.state) / r 
                # phi = self.angle(pi, clst_lmark_av)
                Ftot += Fmod * Fdir 
            pi.state += (self.dt / self.tau_st) * Ftot 
            # __import__('pdb').set_trace()
            pi.state[0] = np.clip(pi.state[0], a_min=-self.H/2, a_max=self.H/2)
            pi.state[1] = np.clip(pi.state[1], a_min=-self.H/2, a_max=self.H/2)

    def initialize_particle(self, particle, seed=None):
            particle.state = np.random.uniform(low=(-0.5*self.W / 2, -0.5*self.H / 2), high=(0.5*self.W / 2, 0.5*self.H / 2))
            particle.orientation = np.random.uniform(low=0, high=2*np.pi)

    def distance(self, pointA, pointB):
        return np.linalg.norm(pointA - pointB)
   
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
                distances = np.array([self.distance(pt, new_candidate) for pt in points])
                if all(distances > min_dist):
                    points.append(new_candidate)

        self.landmarks = np.vstack(points)
        order = np.argsort([self.distance(self.landmarks[0], lm) for lm in self.landmarks])
        self.landmarks = self.landmarks[order]
        self.lmarks_enabled = [True for _ in range(len(self.landmarks))]


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


