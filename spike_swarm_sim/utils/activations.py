import numpy as np

def heaviside(x):
    r""" Applies a Heaviside activation function.

    .. math::
        :nowrap:

        \[\mathcal{H}(x) = \left\{\begin{array}{ccc}
            1, & \text{ if } & x >=0 \\
            0, & \text{ if } & x < 0 \\
        \end{array}\right.\]

    :param np.ndarray x: input vector to which the activation is applied.

    :returns: numpy array with the element-wise transformed vector.
    """
    return x * float(x >= 0.0)

def relu(x):
    """  Applies a ReLU () activation function to the input vector.

    .. math::
        :nowrap:

        \[\mathrm{relu}(x) = \left\{\begin{array}{ccc}
            x, & \text{ if } & x >=0 \\
            0, & \text{ if } & x < 0 \\
        \end{array}\right.\]
    
    :param np.ndarray x: input vector to which the activation is applied.

    :returns: numpy array with the element-wise transformed vector.
    """
    x[x < 0] = 0.0
    return x

def sigmoid(x):
    """ Applies a sigmoid activation function to the input vector.

    .. math::
        :nowrap:

        \[\sigma(x) = \dfrac{1}{1 + e^{-x}}\]

    :param np.ndarray x: input vector to which the activation is applied.

    :returns: numpy array with the element-wise transformed vector.
    """
    return np.divide(1, 1 + np.exp(-x))

def tanh(x):
    """ Applies a hyperbolic tangent activation function to the input vector.

    .. math::
        :nowrap:

        \[\mathrm{tanh}(x) = \dfrac{e^{2x} - 1}{e^{2x} + 1}\]

    :param np.ndarray x: input vector to which the activation is applied.

    :returns: numpy array with the element-wise transformed vector.
    """
    return (np.exp(2*x) - 1) / (np.exp(2*x) + 1)

def softmax(x, tau=1):
    r""" Applies a softmax activation function to the input vector.
    
    .. math::
        :nowrap:

        \[\mathrm{softmax}(x)_j = \dfrac{e^{\mathbf{x}_j/\tau}}{\displaystyle \sum_{ i} e^{x_i / \tau} }\]

    :param np.ndarray x: input vector to which the activation is applied.

    :returns: numpy array with the element-wise transformed vector.
    """
    return np.exp(x/tau) / np.sum(np.exp(x/tau))