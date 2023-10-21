import numpy as np
from mereli.controllers import RobotController, BasicObstacleAvoider
from mereli.register import controller_registry
from mereli.utils import angle_mean

@controller_registry(name='slam1')
class Slam1Controller(RobotController):
    """
    """
    def __init__(self, *args,  **kwargs):
        super(Slam1Controller, self).__init__(*args, **kwargs)
        self.map = np.empty((0, 2)) 
        self.obstacle_avoider = BasicObstacleAvoider(sensitivity=0.7)
        self.t = 0
    
    def read_to_dist(self, ds_readings):
        rho = self.controller_owner.sensors['distance_sensor'].propagation.rho_att 
        mask = ds_readings > 0.05
        dists = -1 * np.ones(8)
        dists[mask] = -np.log(ds_readings[mask])/rho
        return dists 

    def step(self, state, reward=0.0):
        self.t += 1
        ds_st = state['distance_sensor']
        orientation = state['compass']
        position = state['gps']
        sdirs = self.controller_owner.sensors['distance_sensor'].directions(orientation[-1])
        if ds_st.max() > 0.1:
            estim_dists = self.read_to_dist(ds_st)
            for i in range(8):
                if estim_dists[i] != -1 and ds_st[i] > 0.4:
                    n= 1
                    v_obs = np.r_[np.cos(sdirs[i]), np.sin(sdirs[i])]
                    for j in [i-1, i+1]:
                        k = j % 8
                        if estim_dists[k] != -1 and ds_st[k] > 0.3:
                            n +=1
                            v_obs += ds_st[k] * np.r_[np.cos(sdirs[k]), np.sin(sdirs[k])]
                    posObs = position[:2] + estim_dists[i] * v_obs / n
                    self.map = np.vstack((self.map, posObs))

        if self.t == 2000:
            self.plot_map()
            __import__('pdb').set_trace()
        return self.obstacle_avoider.step(state) 

    def reset(self):
        self.t = 0
        self.obstacle_avoider.reset()

    def plot_map(self):

        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from matplotlib.colors import LogNorm
        import numpy as np
        from scipy.stats import gaussian_kde
        kernel = gaussian_kde(self.map.T)
         
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        xv, yv = np.meshgrid(x, y)
        positions = np.vstack([xv.ravel(), yv.ravel()])
        Z = np.reshape(kernel(positions).T, xv.shape)
        fig, ax = plt.subplots()
        xmin = -2
        ymin = -2
        ymax = 2
        xmax = 2
        ax.imshow(np.rot90(Z), cmap=cm.hot,
                  extent=[xmin, xmax, ymin, ymax])
        # ax.plot(m1, m2, 'k.', markersize=2)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        plt.show()
        __import__('pdb').set_trace()

        

        # import matplotlib.pyplot as plt
        plt.scatter(self.map[:,0], self.map[:,1], color='k')
        plt.xlim(-2,2)
        plt.ylim(-2,2)
        plt.show()

