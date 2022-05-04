import numpy as np
from mereli.register import distribution

@distribution(name="gaussian")
class GaussianDist:
    def __init__(self, mu=0, sigma=.1):
        self.mu = mu
        self.sigma = sigma
    
    def __call__(self):
        return np.random.normal(loc=self.mu, scale=self.sigma)

@distribution(name="rayleigh")
class RayleighDist:
    def __init__(self, sigma=1):
        self.sigma = sigma
    
    def __call__(self):
        return np.random.rayleigh(scale=self.sigma)

@distribution(name="exponential")
class ExponentialDist:
    def __init__(self, sigma=10):
        self.sigma = sigma
    
    def __call__(self):
        return np.random.exponential(scale=self.sigma)

@distribution(name="multinomial")
class MultinomialDist:
    def __init__(self, choices=[]):
        self.choices = choices 
    
    def __call__(self):
        idx = np.random.choice(len(self.choices))
        return self.choices[idx]
