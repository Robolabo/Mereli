.. _tutorial_communication:

Communication
===============

Communication systems allow the direct interaction and cooperation among robots. 
Using communication systems the robots can share information about the perceived states, 
the performed actions or even the robot intentions. The exchanged data can be in the 
form of an encoded message or the context information about the environment. Examples 
of environment context are the reception signal strength or the relative orientation (sector)
from where the message was received. 

The main parts of a communication system are the following:

#. **Receiver**: the communication receiver is a robot sensor responsible of perceiving the messages 
   transmitted by other robots in the surroundings. Some receivers can be sectorized sensors, so that 
   the robot can be aware of the orientation from the message was perceived. The most relevant example 
   of a sectorized receiver is the the IR receiver, which has a separate IR photodiode in each sector.
#. **Transmitter**: 
#. **Communication system**:

All of these parts must be included in order to correctly use the inter-robot communication. As 
we will see below, the receiver is implemented and enabled as a sensor and the actuator is 
coded and activated as an actuator.  

Currently, only IR based communication receivers, transmitters and systems are implemented. Nonetheless, 
other communication technologies, such as RF or sound, can be included and used in the future (following 
the same directives).

Receiver
---------


Transmitter
------------


Communication Systems
---------------------