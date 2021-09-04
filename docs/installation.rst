Installation
=============

Clone this repository:

.. code-block::

    >>> git clone https://github.com/Robolabo/SpikeSwarmSim.git
    >>> cd SpikeSwarmSim


Download the simulator requirements:

.. code-block::

    >>> pip3 install -r requirements.txt

Additionally, if a working library MPI is installed in your system:

.. code-block::

    >>> pip3 install mpi4py==3.0.3


.. note:: 

    The simulator can be cleanly executed without MPI installed. However, MPI parallelization functionalities 
    (for example parallel evaluation in genetic algorithms) cannot be harnessed.

In order to execute the examples or to use the simulator outside the SpikeSwarmSim directory, run the following 
command:

.. code-block::

    >>> python setup.py install