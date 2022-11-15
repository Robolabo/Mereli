from functools import wraps
import numpy as np
import numpy.linalg as LA
from mereli.register import initializer_registry
from mereli.utils import tanh, compute_angle, isinstance_of_any



def initializer_handler(func):
    """ Decorator for handling initializer responses and adapting them to either 2D or 3D.
    For example, as initializers sample 2D positions, if the engine being used is 3D then it
    adds the third coordinate with a 0.
    """
    @wraps(func)
    def wrapper(initializer, *args, **kwargs):
        init_res = func(initializer, *args, **kwargs)
        return {
            '2D' : {'orientations' : map(lambda x: x[0], init_res)},
            '3D' : {'positions' : map(lambda x: (x[0], x[1], x[2] if len(x) > 2 else 0), init_res),
                    'orientations' : map(lambda x: (0., 0., x[0]), init_res)}
        }.get(initializer.engine, {}).get(initializer.variable, init_res)
    return wrapper

class Initializer:
    """ Base class of the entity initializers. Initializers allow the automatic initialization of either 
    the positions or orientations of world objects as a group. For example it allows to sample positions 
    uniformly within a square without physical overlapping or sample the 2D coordinates of robots in a swarm 
    as a 2D spatial random graph (assuring swarm compactness). 
    If created correctly within the world class, the ``__call__`` method of the initializer is executed every 
    time that the world/environment is reset.
    This base class should not be instantiated directly and every initializer should inherit from it.
    Once instantiated, it is executed using the ``__call__`` method, that returns a list of sampled positions 
    or orientations.
    :param int num_points: number of points to sample in the initialization.
    :param str engine: physics and render engine used (currently it can be either 2D or 3D)
    :param str variable: entity variable to be initialized (either 'positions' or 'orientations')
    """
    def __init__(self, num_points, engine='3D', variable='position'):
        self.num_points = num_points
        self.engine = engine
        self.variable = variable
        assert engine in ['2D', '3D']
        assert variable in ['positions', 'orientations']

    def __call__(self):
        raise NotImplementedError


@initializer_registry(name='same')
class SameValueInitializer(Initializer):
    """
    """
    def __init__(self, *args, val=0, **kwargs):
        super(SameValueInitializer, self).__init__(*args,  **kwargs)
        self.val = val
        self.values = [[val] for _ in range(self.num_points)]

    @initializer_handler
    def __call__(self):
        """ 
        Call method that returns the position or orientation values according to the initialization process.
        
        :returns: ``list`` of numpy arrays containing the initialization (position or orientation).
        """
        return [np.array(val) for val in self.values]



@initializer_registry(name='fixed')
class FixedInitializer(Initializer):
    """ Initializer class that initializes the positions or orientations always at the given fixed 
    values.
    **Reference Name**: ``fixed``.
    :param int num_points: number of points to sample in the initialization.
    :param str engine: physics and render engine used (currently it can be either 2D or 3D)
    :param str variable: entity variable to be initialized (either 'positions' or 'orientations')
    :param list fixed_values: list of the fixed value of the variable to be initialized for each entity in the group.
        The length of the list must equal the ``num_points`` arg.
    """
    def __init__(self, *args, fixed_values=None, **kwargs):
        super(FixedInitializer, self).__init__(*args,  **kwargs)
        self.fixed_values = fixed_values
        if not isinstance_of_any(fixed_values[0], [list, np.ndarray]):
            self.fixed_values = [[val] for val in self.fixed_values]
        assert self.num_points == len(self.fixed_values)

    @initializer_handler
    def __call__(self):
        """ 
        Call method that returns the position or orientation values according to the initialization process.
        
        :returns: ``list`` of numpy arrays containing the initialization (position or orientation).
        """
        return [np.array(val) for val in self.fixed_values]


@initializer_registry(name='fixed_random')
class FixedRandomInitializer(Initializer):
    """ Initializer class that initializes the positions or orientations randomly from a list of 
    fixed possible values. Each defined value has the same probability to be chosen and the sampling 
    can be with or without replacement (it is advised to sample with replacement). 
    **Reference Name**: ``fixed_random``.
    :param int num_points: number of points to sample in the initialization.
    :param str engine: physics and render engine used (currently it can be either 2D or 3D)
    :param str variable: entity variable to be initialized (either 'positions' or 'orientations')
    :param list possible_values: list of the possible values to be sampled with equal prob.
    :param bool replacement: whether to sample with replacement or not.
    """
    def __init__(self, *args, possible_values=None, replacement=True, **kwargs):
        super(FixedRandomInitializer, self).__init__(*args,  **kwargs)
        self.replacement = replacement
        self.possible_values = possible_values

        if not isinstance_of_any(possible_values[0], [list, np.ndarray]):
            self.possible_values = [[val] for val in self.possible_values]
        # assert self.replacement or len(self.possible_values) < self.num_points

    @initializer_handler
    def __call__(self):
        """ 
        Call method that returns the position or orientation values according to the initialization process.
        :returns: ``list`` of numpy arrays containing the initialization (position or orientation).
        """
        sel_indices = np.random.choice(len(self.possible_values), size=self.num_points, replace=self.replacement)
        return [self.possible_values[idx] for idx in sel_indices]

@initializer_registry(name='random_uniform')
class RandomUniformInitializer(Initializer):
    """ Initializer class that initializes the positions randomly within a defined rectangular area (positions) 
    or within a segment (orientations).  
    
    #. **Sampling positions**: If sampling from a rectangle, the area is specified using the arguments ``low`` and ``high``, which the are 
       bottom left and top right rectangle vertices. The coordinates are sampled from the rectangle randomly according to an uniform dist. 
       If sampling positions, potential overlapping between objects is avoided using the ``min_dist`` attribute.
    #. **Sampling orientations**: Samples the orientations from a uniform distribution :math:`\mathcal{U}(low, high)`. Normally, it is used 
       as  :math:`\mathcal{U}(0, 2\pi)`.
    **Reference Name**: ``random_uniform``.
    :param int num_points: number of points to sample in the initialization.
    :param str engine: physics and render engine used (currently it can be either 2D or 3D)
    :param str variable: entity variable to be initialized (either 'positions' or 'orientations')
    :param list low: low 2D or 1D coordinates of the rectangle or segment used to uniformly sample. If sampling orientations then 
        the type of the arg is ``float`` instead of ``list``.
    :param list high: high 2D or 1D coordinates of the rectangle or segment used to uniformly sample. If sampling orientations then 
        the type of the arg is ``float`` instead of ``list``.
    :param int size: dim. of the hyperrectangle used to sample.
    :param float min_dist: minimum distance between any pair of sampled points.
    """
    def __init__(self, *args, low=[-2, -2], high=[2, 2], size=2, min_dist=1., **kwargs):
        super(RandomUniformInitializer, self).__init__(*args,  **kwargs)
        self.low = low
        self.high = high
        self.size = size
        self.min_dist = min_dist
        self.check_overlapping = size > 1

    @initializer_handler
    def __call__(self):
        """ 
        Call method that returns the position or orientation values according to the initialization process.
        :returns: ``list`` of numpy arrays containing the initialization (position or orientation).
        """
        res = []
        if self.check_overlapping:
            while len(res) < self.num_points:
                # new_sample = np.random.uniform(low=self.low, high=self.high, size=self.size)
                if isinstance(self.low, int):
                    new_sample = np.random.uniform(low=self.low, high=self.high, size=self.size).round(3)
                else:
                    new_sample_x = np.round(np.random.uniform(low=self.low[0], high=self.high[0]), 3)
                    new_sample_y = np.round(np.random.uniform(low=self.low[1], high=self.high[1]), 3)
                    new_sample = np.r_[new_sample_x, new_sample_y]
                if len(res) == 0 or all(LA.norm(new_sample - pp) > self.min_dist for pp in res):
                    res.append(new_sample)
        else:
            res = [np.random.uniform(low=self.low, high=self.high, size=self.size).round(3)\
                    for _ in range(self.num_points)]
        return res
        
@initializer_registry(name='random_circle')
class RandomCircleInitializer(Initializer):
    """
    Class for randomly initializing objects positions within a circle area.
    Each point within the area has the same probability (uniform dist.) and
    overlapping objects are avoided. For that, a minimum distance between points 
    is defined.
    It uniformly samples a random angle and radius and computes the cartesian position  
    corresponding to the sampled polar coordinates.
    **Reference Name**: ``random_circle``.
    
    :param int num_points: number of points to sample in the initialization.
    :param str engine: physics and render engine used (currently it can be either 2D or 3D)
    :param str variable: entity variable to be initialized (either 'positions' or 'orientations')
    :param float radius: maximum radius to be sampled.
    :param list center: center of the circle where points are sampled.
    :param float min_dist: minimum distance between any pair of sampled points.
    """
    def __init__(self, *args, center=[0, 0], radius=1., min_dist=0.6, **kwargs):
        super(RandomCircleInitializer, self).__init__(*args,  **kwargs)
        self.center = np.array(center)
        self.radius = radius
        self.min_dist = min_dist
    
    @initializer_handler
    def __call__(self):
        """ 
        Call method that returns the position or orientation values according to the initialization process.
        :returns: ``list`` of numpy arrays containing the initialization (position or orientation).
        """
        res = []
        while len(res) < self.num_points:
            rnd_mod = np.random.uniform(1e-3, self.radius)
            rnd_phase = np.random.uniform(0, 2 * np.pi)
            new_sample = rnd_mod * np.r_[np.cos(rnd_phase), np.sin(rnd_phase)] + self.center
            if len(res) == 0 or all(LA.norm(new_sample - pp) > self.min_dist for pp in res):
                res.append(new_sample)
        return res



@initializer_registry(name='random_circumference')
class RandomCircumference(Initializer):
    """
    Class for randomly initializing objects positions embedded in a circumference.
    It uniformly samples a random angle and computes the cartesian position within 
    a circumference of a given radius and center.
    **Reference Name**: ``random_circumference``.
    
    :param int num_points: number of points to sample in the initialization.
    :param str engine: physics and render engine used (currently it can be either 2D or 3D)
    :param str variable: entity variable to be initialized (either 'positions' or 'orientations')    
    :param float radius: radius of the circumference where points are sampled.
    :param list center: center of the circumference where points are sampled.
    """
    def __init__(self, *args, radius=1.3, center=[0, 0], **kwargs):
        super(RandomCircumference, self).__init__(*args,  **kwargs)
        self.radius = radius
        self.center = np.array(center)
    
    @initializer_handler
    def __call__(self):
        """ 
        Call method that returns the position or orientation values according to the initialization process.
        :returns: ``list`` of numpy arrays containing the initialization (position or orientation).
        """
        theta_rnd = np.random.uniform(low=0, high=2 * np.pi, size=self.num_points)
        return self.radius * np.stack([np.cos(theta_rnd), np.sin(theta_rnd)]).T + self.center



@initializer_registry(name='random_graph')
class RandomGraphInitializer(Initializer):
    """
    Class for randomly initializing objects positions as a random spatial graph.
    **Reference Name**: ``random_graph``.
    
    :param int num_points: number of points to sample in the initialization.
    :param str engine: physics and render engine used (currently it can be either 2D or 3D)
    :param str variable: entity variable to be initialized (either 'positions' or 'orientations')    
    :param float max_rad: maximum distance between two pairs of nodes.
    :param list initial_pos: 2D position of the initial node in the graph.
    """
    def __init__(self, *args, max_rad=3, initial_pos=(0, 0), **kwargs):
        super(RandomGraphInitializer, self).__init__(*args,  **kwargs)
        self.initial_pos = initial_pos
        self.max_rad = max_rad

    @initializer_handler
    def __call__(self):
        """ 
        Call method that returns the position or orientation values according to the initialization process.
        :returns: ``list`` of numpy arrays containing the initialization (position or orientation).
        """
        points = [np.array(self.initial_pos).astype(float)]
        R_max = self.max_rad
        for _ in range(self.num_points-1):
            new_pos = points[-1]
            while any([np.linalg.norm(new_pos - pos) < 0.6 for pos in points]) or np.min([np.linalg.norm(new_pos - pos) for pos in points]) > 1.4:
                delta_X = points[-1] #- 500
                mu = tanh(-(delta_X / R_max) ** 3)
                sigma_x = np.sin(compute_angle(delta_X / R_max)) ** 2 if np.linalg.norm(delta_X) > R_max/2 else 1
                sigma_y = np.sin(compute_angle(delta_X / R_max) + np.pi / 2) ** 2 if np.linalg.norm(delta_X) > R_max/2 else 1
                rho = 0.5 * np.sin(2 * compute_angle(delta_X / R_max)) ** 3
                cov_mat = np.array([[sigma_x, rho * sigma_x * sigma_y], [rho * sigma_x * sigma_y, sigma_y]])
                new_pos = 0.8 * np.random.multivariate_normal(mu, cov_mat, size=1).flatten() + points[-1]
            points.append(new_pos)
        return points
     


@initializer_registry(name='grid')
class GridInitializer(Initializer):
    """
    Class for randomly initializing objects positions within a 2D regular lattice or grid. 
    It is a deterministic initialization.
    **Reference Name**: ``grid``.
    .. todo::
        Class not implemented yet.
    :param int num_points: number of points to sample in the initialization.
    :param str engine: physics and render engine used (currently it can be either 2D or 3D)
    :param str variable: entity variable to be initialized (either 'positions' or 'orientations')    
    """
    def __init__(self, *args, center=[0,0], delta_x=0.2, delta_y=0.2, shuffle=True, **kwargs):
        super(GridInitializer, self).__init__(*args,  **kwargs)
        self.center = center
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.shuffle = shuffle
        
    @initializer_handler
    def __call__(self):
        """ 
        Call method that returns the position or orientation values according to the initialization process.
        :returns: ``list`` of numpy arrays containing the initialization (position or orientation).
        """
        H = int(np.floor(np.sqrt(self.num_points)))
        W = int(np.ceil(np.sqrt(self.num_points)))
        x = np.linspace(self.center[0] - self.delta_x*W/2, self.center[0] + self.delta_x*W/2, W) 
        y = np.linspace(self.center[1] - self.delta_y*H/2, self.center[1] + self.delta_y*W/2, H) 
        xx, yy = np.meshgrid(x, y)
        points = []
        for x_i, y_i in zip(xx.flatten(), yy.flatten()):
            points.append(np.array([x_i, y_i]))
        if self.shuffle:
            np.random.shuffle(points)
        return np.array(points).copy()
