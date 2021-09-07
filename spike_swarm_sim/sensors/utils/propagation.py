import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class Propagation:
    """ Base class for the mathematical models simulating short range signal 
    propagation. This propagation models are used by the sensor step methods to 
    map Euclidean distances and misalignment angles into signal strengths/readings.
    """
    def __init__(self):
        pass

    def __call__(self, rho, phi, theta=None):
        """ Returns the received signal strength (normalized generally) 
        based on the reception radius and misalignment angle. This method is 
        empty in this base class and must be overwritten by other propagation 
        model classes inheriting from it. 

        :param float rho: distance to the target position (meters).
        :param float phi: misalignment in radians of the received ray.

        :returns: float representing the normalized signal strength of the reception [float].
        """
        raise NotImplementedError
    
    def plot(self, max_rad=5, sensor_name=None):
        """ Illustrates the polar plot of a sector coverage.

        :param float max_rad: maximum coverage range (to set the limits of the plot) 
        :param str sensor_name: name of the sensor (to be set in the title). 
        """
        Rvals = np.linspace(0, max_rad, 200)
        theta_vals = np.radians(np.linspace(0, 360, 360))
        Rgrid, theta_grid = np.meshgrid(Rvals, theta_vals)
        direction = 0
        values = np.zeros([Rgrid.shape[0], Rgrid.shape[1]])
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                theta = theta_grid[i,j]
                phi = np.abs(direction - theta)
                if phi > np.pi: 
                    phi = 2 * np.pi - phi
                values[i, j] = self(Rgrid[i,j], phi) 
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
        line = ax.contourf(theta_grid, Rgrid, values, levels=50,\
                        cmap=plt.get_cmap('Reds'))   
        fig.colorbar(line, ax=ax)
        if sensor_name is not None:
            ax.set_title('Coverage of {} for a sector.'.format(sensor_name))
        else:
            ax.set_title('Sector Coverage.')
        plt.show()
        
    def plot_directivity(self, directions, max_rad=5, sensor_name=None):
        """ Illustrates the polar plot with the directivity pattern of all the sectors at once. 
        It only takes into account the losses due to misalignment, the distance attenuation is 
        not depicted in this plot.

        :param float max_rad: maximum coverage range (to set the limits of the plot) 
        :param str sensor_name: name of the sensor (to be set in the title). 
        """
        theta_vals = np.radians(np.linspace(0, 360, 360))
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
        for direction in directions:
            phi_vals = np.abs(direction - theta_vals)
            phi_vals[phi_vals > np.pi] =  2 * np.pi - phi_vals[phi_vals > np.pi]
            
            # import pdb; pdb.set_trace()
            ax.plot(theta_vals, [self(0.01, th) for th in phi_vals])
            
            # if sensor_name is not None:
            #     ax.set_title('Coverage of {} for a sector.'.format(sensor_name))
            # else:
            #     ax.set_title('Sector Coverage.')
        plt.show()

class ExpDecayPropagation(Propagation):
    r""" Simplified signal propagation using the exponential decaying 
    of the signal of both radius and phi. Both terms are combined as 
    a product. Formally, the model is expressed as,

    .. math::

        f(\rho, \alpha) = \exp\left\{\lambda_{\rho}\,\rho - \lambda_{\alpha}\,\alpha^2\right\}

    where :math:`\rho` is the distance and :math:`\alpha` is the misalignment.


    :param float rho_att: coefficient tuning the attenuation due to the distance (:math:`\lambda_{\rho}`). 
    :param float phi_att: coefficient tuning the attenuation due to the misalignment (:math:`\lambda_{\alpha}`). 
    """
    def __init__(self, rho_att=0.3, phi_att=1):
        self.rho_att = rho_att
        self.phi_att = phi_att

    def __call__(self, rho, phi):
        return np.exp(-self.rho_att * rho) * np.exp(-self.phi_att* phi ** 2)# 1.5 DS, 0.75 LS

class RSSI_Propagation(Propagation):
    """ 
    .. todo::

        The implementation and testing of this class in currently in process. 
    """
    def __init__(self, noise_sigma=0.05):
        self.rssi_0 = -69
        self.n = 2.
        self.noise_sigma = noise_sigma
        self.buffer = [0.0] * 3

    def __call__(self, rho, phi):
        rho = max(rho, 1e-3)
        #* Simulate measured noisy RSSI
        rssi = self.rssi_0 - 10 * self.n * np.log10(rho) + np.random.randn() * self.noise_sigma 
        # Normalize (suppose max dist 10 meters)
        # rssi_max = self.rssi_0 - 10 * self.n
        # rssi_min = self.rssi_0 - 10 * self.n * np.log10(0.1)
        estim_dist = 10 ** ((-rssi + self.rssi_0) / (10*self.n))
        self.buffer.append(estim_dist)
        self.buffer.pop(0)
        estim_dist = np.array([0.1, 0.3, 0.6]).dot(np.array(self.buffer))
        return estim_dist / 10