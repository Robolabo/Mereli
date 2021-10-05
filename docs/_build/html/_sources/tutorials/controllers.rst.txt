.. _tutorial_controllers:

Controllers
===============



Overview
---------

Controllers are programs that define the behavior of robots. Controllers received sensory information, manipulate it and produce 
actions as a result. The actions are thereafter fed to actuators in order to interact with the environment. A controller can be any 
kind of program, ranging from simple rule based decions to neural controllers (control through artificial neural networks). In this 
simulator, any controller inherits directly or indirectly from the class py:class:`mereli.controllers.Controller`. 
Additionally, controllers applied to robots must inherit from py:class:`mereli.controllers.RobotController`. The reason for having 
these two base classes is because in this simulator, the term controller transcends the control of robots. For instance, a moving light that 
describes an orbit is also considered a controller. 

The robot controllers that are currently implemented in the simulator are the following:

+---------------------------+----------------------------------+-------------------------------------------------------------------------+
|   **Reference Name**      |   **Python Class**               | **Description**                                                         |
+---------------------------+----------------------------------+-------------------------------------------------------------------------+
|``random_walk``            |``RandomMovementController``      | Move the robot randomly.                                                | 
+---------------------------+----------------------------------+-------------------------------------------------------------------------+
|``basic_obstable_avoider`` | ``BasicObstacleAvoider``         | Controller for detecting obstacles and avoiding collision.              | 
+---------------------------+----------------------------------+-------------------------------------------------------------------------+
| ``braitenberg2b``         | ``Braitenberg2B``                | Braitenberg Vehicle 2B.                                                 | 
+---------------------------+----------------------------------+-------------------------------------------------------------------------+
|  ``braitenberg2a``        | ``Braitenberg2A``                | Braitenberg Vehicle 2A.                                                 |
+---------------------------+----------------------------------+-------------------------------------------------------------------------+
| ``neural_controller``     | ``NeuralController``             | | Controller with a neural network for processing sensed stimuli and    |
|                           |                                  | | computing actions as a result.                                        |
+---------------------------+----------------------------------+-------------------------------------------------------------------------+



Creating Controllers
---------------------

In order to instantiate a controller within a simulation, 




Custom  Controllers
---------------------






