.. _tutorial_controllers:

Controllers
===============



Overview
---------

Controllers are programs that define the behavior of robots. Controllers received sensory information, manipulate it and produce 
actions as a result. The actions are thereafter fed to actuators in order to interact with the environment. A controller can be any 
kind of program, ranging from simple rule based decions to neural controllers (control through artificial neural networks). 


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


