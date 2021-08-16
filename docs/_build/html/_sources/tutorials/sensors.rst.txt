.. _tutorial_sensors:

Sensors
=============================





+---------------------------+----------------------------------+-------------------------------------------------------------------------+
|   **Reference Name**      |   **Python Class**               | **Description**                                                         |
+---------------------------+----------------------------------+-------------------------------------------------------------------------+
|   ``distance_sensor``     |    ``DistanceSensor``            |  Sensor for estimating the distance to nearby solid objects.            | 
+---------------------------+----------------------------------+-------------------------------------------------------------------------+
|    ``light_sensor``       |   ``LightSensor``                |  Sensor for estimating the light intensity.                             | 
+---------------------------+----------------------------------+-------------------------------------------------------------------------+
|  ``IR_receiver``          | | ``IRCommunicationReceiver``    | IR based communication receiver.                                        | 
|                           | | ``BufferedIRCommRX``           |                                                                         |
+---------------------------+----------------------------------+-------------------------------------------------------------------------+
|  ``ground_sensor``        | ``GroundSensor``                 | Sensor that detects the presence of a ground area underneath the robot. |
+---------------------------+----------------------------------+-------------------------------------------------------------------------+
| ``color_sensor``          | ``ColorSensor``                  | High level sensor for detecting nearby objects of a precise color.      |
+---------------------------+----------------------------------+-------------------------------------------------------------------------+
|``joint_position_sensor``  | ``JointPositionSensor``          | Sensor for measuring the position state of the joints of the robot.     |
+---------------------------+----------------------------------+-------------------------------------------------------------------------+
|``joint_velocity_sensor``  | ``JointVelocitySensor``          | Sensor for measuring the velocity state of the joints of the robot.     |
+---------------------------+----------------------------------+-------------------------------------------------------------------------+
| ``collision_sensor``      | ``CollisionSensor``              | Sensor for detecting if the robot is colliding with a solid object.     |
+---------------------------+----------------------------------+-------------------------------------------------------------------------+
| ``own_position_sensor``   | ``OwnPositionSensor``            | High level sensor for obtaining the absolute position of the robot.     |
+---------------------------+----------------------------------+-------------------------------------------------------------------------+
|``own_orientation_sensor`` | ``OwnOrientationSensor``         | High level sensor for obtaining the absolute orientation of the robot.  |
+---------------------------+----------------------------------+-------------------------------------------------------------------------+
|``neighborhood_pos_sensor``| ``NeighborhoodPositionSensor``   | High level sensor for obtaining the absolute position of the neighbors. |
+---------------------------+----------------------------------+-------------------------------------------------------------------------+

.. warning::
    The ``own_position_sensor``, ``own_orientation_sensor`` and ``neighborhood_pos_sensor`` are sensors that break the locality and 
    partial observability principals by reading high level absolute information such as the robot coordinates or absolute heading orientation. 

Creating Sensors
-------------------

Sensors have to be enabled by the robot controller in order to be used afterwards. In other words, robots have a pool of sensors they can use and 
the controllers specify the subset of them that will be harnessed in each experiment. For instance, a single robot light pursuit experiment 
only uses the light sensor while an obstacle avoidance task only employs the distance sensor. 
The following extract of code shows how to add sensors one by one to an existing controller object. In the example, only 
the distance sensor. 

Example::

>>> ctlr = BasicObstacleAvoider()
>>> ctlr.add_sensor("distance_sensor", {"n_sectors" : 4, "range" : 1})
>>> ctlr.add_actuator("joint_velocity_actuator",  {"joint_ids" : [0, 1], "max_velocity" : 13})
>>> ent = Epuck3D(pos, ori, controller=ctlr)

The ``add_sensor`` method receives both the reference name of the sensor (do not confuse with the class name, 
see table above). Notice that the ``add_sensor`` method of the controller only requests the indicated sensor. It is 
the creation of the robot where the actual sensors are created (see ``Robot`` constructor). Alternatively, the sensors can 
be also created and activated through the ``reset`` method of the robot.

Alternatively, sensors can be added using ``add_sensors_from_dict``. Using this method, the user can add all the robot 
sensors at once. The following code example shows how to add sensors using this method:

Example::

>>> ctlr = BasicObstacleAvoider()
>>> sensors_dict = {
>>>     "distance_sensor" : {"n_sectors" : 4, "range" : 1},
>>>     "light_sensor" : {"n_sectors" : 4, "range" : 5}
>>> }
>>> ctlr.add_sensors_from_dict(sensors_dict)
>>> ctlr.add_actuator("joint_velocity_actuator",  {"joint_ids" : [0, 1], "max_velocity" : 13})
>>> ent = Epuck3D(pos, ori, controller=ctlr)

Thereafter, once created, the sensors measurements are read by means of the ``step`` method that must be 
implemetented in each sensor class. However, this method must not be called directly because it is the method 
``perceive`` of the class ``Robot`` the one responsible of direcltly reading the sensor units. The method 
``perceive`` is similarly called by the ``Robot.step`` method at each simulation cycle. The different readings 
of the enabled sensors are gathered to compose the current partially observable state (which is a ``dict`` mapping 
sensor reference names to sensor reading vectors).   


Directional Sensors
-------------------  

Directional sensors are a type of sensors that have independent sensing units for diferent sectors or coverage areas. 
The clearest example is the one of epucks or other similar mobile robots. In these cases multiple sensing units are placed 
at strategic points of the robot perimeter. Therefore, aside from having an independent measurement of what is happening on 
each sector, the robot can know the orientation from where an event was perceived. 
For example, the ``light_sensor`` can be sectorized with, say, 4 sectors: the most sensitive direction of each 
sector would then be :math:`[\theta_r,\, \theta_r + \pi/2,\, \theta_r + \pi,\, \theta_r + 3\pi/4]`, where  
:math:`\theta_r` is the robot heading orientation. Consequenlty, if the light is detected only in the third sector 
(the one pointing at :math:`\theta_r + \pi`),  then the robot can both that there is a light source in the surroundings 
and that it is placed at its back. 

FIGURA, comentar ejemplo.


The refence names of the directional sensors that are currently implemented are: ``distance_sensor``, ``light_sensor``, ``IR_receiver`` and 
``color_sensor``. 
All the directional sensors inherit from the base class ``DirectionalSensor``. This class implements the ``step`` method that 
is generally common to all directional sensors (it does not have to be overwritten unless needed). 
In contrast, any sensor inheriting from  ``DirectionalSensor`` must fill the methods ``target_filter`` and 
``step_direction``, particularizing its behavior. Essentially, the ``target_filter`` method filters out those entities 
(among all the entities in the surroundings) that are candidate to be perceived by the sensor. The ``step_direction`` 
method implements the actual sensor reading in each sector and for each perceived entitity. 
All the sectors have the same sensing procedure. 
For a detailed explanation of the method arguments, see :py:meth:`spike_swarm_sim.sensors.base_sensor.DirectionalSensor.step_direction` .
In the implemented directional sensors, such as ``DistanceSensor`` and ``LightSensor``, the main steps of 
``step_direction`` are the following:

1. If the input argument ``direction_reading`` is ``None`` (no object was perceived before), the initialize 
   the reading.
2. Filter out those objects that are very distant or in which the misalignment (``phi``) is lower than the sensor apeture.
3. Compute the signal strength based on the Euclidean distance to the entity (``rho``) and the misalignment (``phi``) and 
   using the corresponding propagation model. 
4. Cast an invisible ray between the physicsl sensor position and the perceived entity position (this may be changed in the 
   future to the closest point of the perceived entity). This casted ray verifies the presence of obstacles.
5. If no obstacle was detected then update the direction reading. The update can be selecting the maximum value 
   between the previous measurement and the new one of as an additive field (among other options).

There is a last explanation before going on to the next topic. When creating the URDF files of 3D entities, the directional 
sensors have to be defined and attached to established links. For that, we have designed a non-standardized xml tag structure,
as it can be observed in the following example:

Example::

>>> <sensor name="distance_sensor">
>>>     <sector index="0">
>>>         <parent link="IR0"/>
>>>         <origin xyz="0 0 0" rpy="0 0 0"/>
>>>     </sector>
>>>     <sector index="1">
>>>         <parent link="IR1"/>
>>>        <origin xyz="0 0 0" rpy="0 0 0"/>
>>>    </sector>
>>>    <sector index="2">
>>>        <parent link="IR2"/>
>>>        <origin xyz="0 0 0" rpy="0 0 0"/>
>>>    </sector>
>>>    <sector index="3">
>>>         <parent link="IR3"/>
>>>         <origin xyz="0 0 0" rpy="0 0 0"/>
>>>     </sector>
>>> </sensor>

The tag sensor has an attribute that must hold the name of the sensor's reference name. Thereafter, 
there is one inner tag per each sector of the sensor. In the example we show the ``distance_sensor`` 
with 4 sectors. Each sector has the attribute index, indicating the index within [0, n_sectors-1] of the 
sector. The tag parent inside sectors denotes the physical link to which the sensor is attached. 
A link can have multiple sensors as it can observed in the source code of the epuck.urdf. 
The current use of this task is to access via the physics engine API to the precise positions of the 
physical 3D models (links) of the sensors. This positions are used for casting rays and detecting obstacles. 
This information can be accessed through the ``get_sensor_position`` method of the physics engine.
Due to the fact that this part of the URDF files is not standardized, we process it manually at the 
``add_physics`` method of the physics engine and using the ``parse_sensors`` method. 

.. todo::
    The tag origin is not currently used, but its future use is to indicate the orientation (via quaternion 
    or Euler angle) of the maximum sensor sensitivity. For the moment, only equiareal directional sectors of 
    mobile robots are implemented.

.. todo::
    The sensor tags already mentioned are currently only processed and parsed for directional sensors. Nonetheless, 
    the idea is to define any sensor of the robots and use the URDF to state the sensing equipment of robots. 
    In this way, robot controllers could only make use of the sensors specified in the URDF file.

.. todo:: 
    The definition of sensors within the model files is only implemented in the 3D URD files. Due to its reduced 
    relevance, it is currently in process in the case of the 2D JSON model files.
