import numpy as np
from mereli.register import normalization


class Normalization:
    def __init__(self, max_val, min_val):
        self.max_val = max_val
        self.min_val = min_val

    def apply(self, value):
        pass

    def revert(self, value):
        pass

@normalization(name="linear")
class LinearNormalization(Normalization):
    def __init__(self, *args, **kwargs):
        super(LinearNormalization, self).__init__(*args, **kwargs)

    def apply(self, value):
        return np.clip((value - self.min_val) / (self.max_val - self.min_val), a_min=0, a_max=1)

    def revert(self, value):
        return value * (self.max_val - self.min_val) + self.min_val

@normalization(name="exponential")
class ExpNormalization(Normalization):
    def __init__(self, *args, **kwargs):
        super(ExpNormalization, self).__init__(*args, **kwargs)

    def apply(self, value):
        return np.clip((np.log10(0.5 * value) - self.min_val) / (self.max_val - self.min_val), a_min=0, a_max=1)

    def revert(self, value):
        return np.clip(2 * 10 ** (value * (self.max_val - self.min_val) + self.min_val), a_min=self.min_val, a_max=self.max_val)