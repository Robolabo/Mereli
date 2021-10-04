.. _tutorial_actuators:

Actuators
==========

Overview
---------

+---------------------------+----------------------------------+-------------------------------------------------------------------------+
|   **Reference Name**      |   **Python Class**               | **Description**                                                         |
+---------------------------+----------------------------------+-------------------------------------------------------------------------+
|``joint_velocity_actuator``|    ``JointVelocityActuator``     | Actuator for controlling the velocity of the joints of the robots.      | 
+---------------------------+----------------------------------+-------------------------------------------------------------------------+
|``joint_position_actuator``|   ``JointPositionActuator``      | Actuator for controlling the angle position of the joints of the robots.| 
+---------------------------+----------------------------------+-------------------------------------------------------------------------+
|  ``led_actuator``         |  ``LedActuator``                 | Actuator that controls the LEDs of a robot.                             | 
+---------------------------+----------------------------------+-------------------------------------------------------------------------+
|  ``IR_transmitter``       | ``CommunicationTransmitter``     | | Communication transmitter based on IR technology to enable            |
|                           |                                  | | inter-robot interactions.                                             |
+---------------------------+----------------------------------+-------------------------------------------------------------------------+
| ``grasp_actuator``        | ``GraspActuator``                | | High level actuator for simplying the action of grasping and dropping |
|                           |                                  | | small objects scattered in the arena.                                 |
+---------------------------+----------------------------------+-------------------------------------------------------------------------+
|``RF_transmitter``         | ``RF_Transmitter``               | RF based communication transmitter. **Currently in process**.           |
+---------------------------+----------------------------------+-------------------------------------------------------------------------+

Creating Actuators
-------------------

Even though we have already mentioned how to create and register actuators within a simulation in previous examples, 
lets introduce it formally and in detail. As sensor, actuators are enabled in the robot controller, so that 
the subset of actuators executed in a simulator is customizable. You can enable actuators in a previously 
created controller ``ctlr`` as follows:

**Example 3.1**::

>>> ctlr.add_actuator("joint_velocity_actuator",  {"joint_ids" : [0, 1], "max_velocity" : 8})
>>> ctlr.add_actuator("led_actuator",  {})

It works exactly as its ``Controller.add_sensor`` counterpart, yet accepting actuator reference names and the proper 
actuator arguments or parameters.

Alternatively, multiple sensors can be activated at once using the ``Controller.add_actuator_from_dict``. 
For example:

**Example 3.2**::

>>> actuator_dict = {
>>>    "joint_velocity_actuator" : {"joint_ids" : [0, 1], "max_velocity" : 8},
>>>    "led_actuator"    : {}
>>> }
>>> ctlr.add_actuators_from_dict(actuator_dict)

Enabled actuators must be created before the simulation is started.  The actual creation/instantiation 
of the actuators is accomplished when calling the robot class constructor. It is also possible to create the actuators by calling the ``Robot.reset`` 
if they have not been previously created for whatever reason (so that the actuators are not instantiated twice). 
The following code examples clarifies this issue:

**Example 3.3**::

>>> # Activates the joint velocity actuator.
>>> ctlr.add_actuator("joint_velocity_actuator", {"joint_ids" : [0, 1], "max_velocity" : 8}) 
>>> # Creates an actual joint velocity actuator object.
>>> robot = Epuck([0,0,0], 0, controller=ctlr) 
>>> # Does not create the actuator again because it was instantiated in the
>>> # previous line.
>>> robot.reset()
>>>
>>> # Activates the LED actuator.
>>> robot.controller.add_actuator("led_actuator", {}) 
>>> # Creates an actual LED actuator object.
>>> robot.reset()


Actuator instances are stored inside the robot class owning them, as an attribute of the class called ``Robot.actuators``. This variable is a 
python ``dict`` mapping actuator reference names to actual python actuator class instantiations. For example, the following code 
block shows the content of ``robot.actuators`` after the executing of the Example 3.3:


**Example 3.4**::

>>> print(robot.actuators)
>>>    {
>>>        'led_actuator': <mereli.actuators.led_actuator.LedActuator object at 0x000001F3E7997408>, 
>>>        'joint_velocity_actuator': <mereli.actuators.joint_actuator.JointVelocityActuator object at 0x000001F3E79972C8>}
>>>    }



Thereafter, once created, the actuator actions are firslty planned inside the ``Robot.step`` method by calling ``Robot.plan_actions``. Planning 
an action essentially means registering it in order to "maybe" be executed later by the world class. Once all the robot ``step`` methods have 
been executed (and all the actions are planned), it is the world the one responsible for calling the ``Actuator.step`` method through its own 
``actuate`` method. This division of action planning and actual executing will be revisited in the future (an maybe modified).

Joint Velocity Actuator
------------------------

The joint velocity actuator controls the velocity of the joints of a robot. 
Given a reference velocity (action), the actuator drives the velocity of the joint towards 
the desired target value. All its functionality is collected inside the ``JointVelocityActuator.step``
method. The velocity control is accomplished using the pybullet ``setJointMotorControl2`` function
(with ``controlMode=p.VELOCITY_CONTROL``), which internally implements all the control systems for us.  
The action argument provided to the ``step`` method is a vector settling the desired reference velocity of 
each joint to be controlled. The action references are contrained within [-1, 1], meaning that an 
action of 1 means maximum velocity in one sense and -1 represents maximum velocity in the opposite 
sense. An action of 0 means no rotation at all. 
As an example, in order to control a two wheeled mobile robot (e.g. the e-puck), one would set the 
``joint_id`` argument of the class to ``[0, 1]``, specifying that there are two joints to be controlled 
(and their ids are 0 and 1). 

Joint Position Actuator
------------------------

The joint position actuator controls the angle position of the joints of a robot. 
Given a reference angle position (action), the actuator drives the position of the joint towards 
the desired target value.All its functionality is collected inside the ``JointPositionActuator.step``
method. The velocity control is accomplished using the pybullet ``setJointMotorControl2`` function
(with ``controlMode=p.POSITION_CONTROL``), which internally implements all the control systems for us.  
The action argument provided to the ``step`` method is a vector settling the desired reference velocity of 
each joint to be controlled. The action argument is a vector settling the desired reference positions of 
each joint. The action references are contrained within [-1, 1], which are mapped 
into [-pi, pi] inside the ``step`` method.

LED Actuator
-------------




Grasp and Drop Actuator
------------------------
