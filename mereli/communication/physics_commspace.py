import numpy as np
from mereli.utils.alg_utils import torus_distance, torus_angle, ring_distance, ring_angle
from mereli.register import comm_space_registry

MASS_LMARK = 1
MASS_ROBOT  = 1
FWALL = 0.5

class RobotMolecule:
    """ Class that represents a robot as a particle inside the virtual space """
    def __init__(self):
        self.state = None # Coordiantes or state
        self._mass = MASS_ROBOT # Mass of the particle
        self._interaction = 'R' # R repeller, A attractor
        self.real_robot = None # Instance of the real robot associated to the particle
        self.lmark = None # Current landmark or virtual region to where the particle currently belongs
        self.disabled_lmarks = [] # Ignore, nor relevant
   
    def reset(self):
        self.lmark = None
        
    def update_current_lmark(self):
        """ Update the current landmark based on the current particle's state """
        self.lmark = np.argmin([np.linalg.norm(self.state - lmark.state) for lmark in self.landmarks])

    @property 
    def mass(self):
        return self._mass

    @mass.setter
    def mass(self, new_mass):
        self._mass = new_mass

    @property
    def neighbors(self):
        """Neighbors of the robot in the real world. """
        if self.real_robot is None:
            return [] 
        return [robot.virtual_particle for robot in self.real_robot.neighbors]

    def attach_to_robot(self, real_robot):
        """ Tie robot to particle """
        self.real_robot = real_robot
        self.real_robot.virtual_particle = self
    
    @property
    def id(self):
        return self.real_robot.gid

    @property
    def landmark_states(self):
        """ States of all the landmarks. """
        return np.array([lmk.state for lmk in self.landmarks])

class LandmarkMolecule:
    """ Class that represents a landmark particle. """
    def __init__(self):
        self.state = None
        self._mass = MASS_LMARK 
        self._interaction = 'A' # R repeller, A attractor
        self._real_robot = None

    @property 
    def mass(self):
        return self._mass

    @mass.setter
    def mass(self, new_mass):
        self._mass = new_mass

    @property 
    def interaction(self):
        return self._interaction



@comm_space_registry(name='VirtualPhysicsCommSpace')
class VirtualPhysicsCommSpace:
    """ Class that represents the overall virtual space. """
    def __init__(self, H=2, W=2, tau_st=10):
        self.dt = 0.1 # Time discretization of dynamics
        self.H = H # Height of a 2D space
        self.W = W # Width of a 2D space
        self.tau_st = tau_st 
        self.num_lmarks = 0  
        self.particles = {} # Dictionary that stores the robot particles. 
        self.landmarks = [] # List of landmarks 
        self.lmk_init_method = 'fixed' # Describes how landmarks are initialized
        self.randomize_neighbors = False # Never used
        # self.lmarks = [{'idx' : i, 'pos' : None, 'pr' : 0} for i in range(self.num_lmarks)]

    def step(self):
        self.step_dynamics()
        self.t += 1
        
    def compute_force(self, pi, pj, ftype='R', Fmax=100):
        """ Calculates the force applied by particle pj to particle pi.
        It is based on the masses and the distances according to molecular physics. 
        """
        G = 0.3
        r = np.linalg.norm(pi.state - pj.state) + 1e-3
        Fmod = (G * pi.mass * pj.mass) / (r ** 2) 
        Fmod = min(Fmod, Fmax) # Module of the force vector
        # Phase or direction of the force vector. Depends the nature of the particles (attractor or repellers).
        Fdir = (pi.state - pj.state) / r if ftype == 'R' else  (pj.state - pi.state) / r
        return Fmod * Fdir
        
    def step_dynamics(self):
        # Update landmark info
        origin = np.zeros(2).astype(float)
        """ Virtual-Physics approach 
        Update the virtual state of every agent based on their neighbors.
        """
        for i, pi in enumerate(self.particles.values()):
            r = np.linalg.norm(pi.state - origin)
            Fmod = (0.1 * pi.mass * 1) / (r ** 2 + 0.05) 
            Fmod = min(Fmod, 100)
            Fdir = (pi.state - origin) / (r + 0.05)
            Ftot = 0* Fmod * Fdir

            # Compute forces from other particles
            # Compute neighbors 
            neighbor_particles = [robot.virtual_particle for robot in pi.real_robot.neighbors]
            # Only tesiting and debugging: using all particles full range (UNCOMMENT TO USE)
            # neighbor_particles = self.particles.values()
            for j, pj in enumerate(neighbor_particles):
                if pi.id != pj.id:
                    Fij = self.compute_force(pi, pj, ftype='R')
                    Ftot += Fij 

            # Compute forces that all the landmarks apply to the particle
            for lm in pi.landmarks:
                # r = np.linalg.norm(pi.state - lm.state)
                Fij = self.compute_force(pi, lm, ftype='A')
                Ftot += Fij 
             
            # Rotational dynamics when a wall is close by
            #### TEST CODE ####### 
            ### DISABLED. TO ENABLE: Fwall =FWALL
            eps = 0.8 
            Fwall = 0 # FWALL  
            if pi.state[0] - eps <= -self.W / 2:
                if pi.state[1] - eps < -self.H / 2:
                    # Drive Right 
                    Ftot[0] += Fwall 
                else:
                    # Drive Up 
                    Ftot[1] -= Fwall 
                    
            elif pi.state[0] + eps >= self.W / 2:
                if pi.state[1] + eps >= self.H / 2:
                    # Drive Left 
                    Ftot[0] -= Fwall 
                else:
                    # Drive Down 
                    Ftot[1] += Fwall                 

            if pi.state[1] - eps <= -self.H / 2:
                if pi.state[0] + eps <= self.W / 2 and pi.state[0] - eps <= -self.W / 2:
                    # Drive Down 
                    Ftot[1] += Fwall                   
                else:
                    # Drive Left 
                    Ftot[0] -= Fwall                     
            elif pi.state[1] + eps >= self.H / 2:
                if pi.state[0] + eps >= self.W / 2:
                    # Drive Up 
                    Ftot[1] -= Fwall                     
                else:
                    # Drive Right 
                    Ftot[0] += Fwall                     
            #### END OF TEST CODE ####### 
            #####################

            ### UPDATE STATE DYNAMICS
            pi.state += (self.dt / self.tau_st) * Ftot 

            # Constrain the states to the considering the H and W of the virtual space. 
            pi.state[0] = np.clip(pi.state[0], a_min=-self.H/2, a_max=self.H/2)
            pi.state[1] = np.clip(pi.state[1], a_min=-self.H/2, a_max=self.H/2)
            
            pi.update_current_lmark()

    def initialize_particle(self, particle, seed=None):
        # Intialize the state of particles randomly in [-0.1W/2, 0.1W/2]x[-0.1H/2, 0.1H/2]. 
        scale = .1 #0.1
        particle.state = np.random.uniform(low=(-scale*self.W / 2, -scale*self.H / 2), high=(scale*self.W / 2, scale*self.H / 2))
        # ANother option would be to initialize as zeros. 
        # particle.state = np.zeros(2).astype(float)

    def add_particle(self, robot_name, real_robot):
        particle = RobotMolecule()
        particle.attach_to_robot(real_robot)
        self.particles[robot_name] = particle 
   
    def add_landmark(self, position, mass=1.):
        self.num_lmarks += 1
        lmark = LandmarkMolecule()
        lmark.mass = mass
        if position is not None: # If None it means random
            lmark.state = position
        else:
            self.lmk_init_method = 'random'
        self.landmarks.append(lmark)

    def distance(self, pointA, pointB):
        """ Euclidean distance between 2 points """
        return np.linalg.norm(pointA - pointB)
   
    def generate_rnd_lmarks(self, n_lmarks, min_dist):
        """ Method to generate random positions of landmarks that guarantee a minimum distance among them. """
        spc_dim = 2
        points = []
        print(np.random.random())
        while len(points) < n_lmarks:
            new_candidate = np.random.uniform(low=-self.H / 2, high=self.H / 2, size=spc_dim)
            if len(points) == 0:
                points.append(new_candidate)
            else:
                distances = np.array([self.distance(pt, new_candidate) for pt in points])
                if all(distances > min_dist):
                    points.append(new_candidate)
        # print(points, self.landmarks)
        for st, lm in zip(points, self.landmarks):
            lm.state = st
            # print(st)
            # __import__('pdb').set_trace()
        # order = np.argsort([self.distance(self.landmarks[0], lm) for lm in self.landmarks])
        # self.landmarks = self.landmarks[order]
        # self.lmarks_enabled = [True for _ in range(len(self.landmarks))]

    def reset(self, seed=None):
        """ Reset the whole virtual space """
        self.t = 1
        if self.lmk_init_method == 'random':
            self.generate_rnd_lmarks(self.num_lmarks, 1)
        
        for particle in self.particles.values():
            particle.reset()
            self.initialize_particle(particle, seed=seed)
            particle.landmarks = self.landmarks.copy()
