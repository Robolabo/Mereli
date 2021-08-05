.. _tutorials:

*****************************************
Tutorials
*****************************************

Worlds
==============================



Entities
==============================

Entities are any kind of world objects that are created in the world or environment space. Jointly with the physics and render engine, they 
define the variety experiments to be carried out. Every entity defined in the simulator inherits from the class :py:class:`spike_swarm_sim.objects.world_object.WorldObject`. 
This base class cannot be directly instantiated and defines a set of abstract properties of entities. Even thought their names are descriptive, the 
following table defines the currently implemented entity properties:

+-----------------------+---------------------------------------------------------------------+
|   **Property**        | **Description**                                                     |
+-----------------------+---------------------------------------------------------------------+
|   tangible            |   Whether the object has collisions or not.                         |
+-----------------------+---------------------------------------------------------------------+
|   static              |   Whether the object is static or can move.                         |
+-----------------------+---------------------------------------------------------------------+
|   luminous            |   Whether the object emits light or not.                            |
+-----------------------+---------------------------------------------------------------------+
|   controllable        |   Whether the object has a controller or not.                       |
+-----------------------+---------------------------------------------------------------------+
|   trainable           | Whether the object controller can be trained (not really used yet). |
+-----------------------+---------------------------------------------------------------------+

Additionally, on top of ``WorldObject``, ``WorldObject2D`` and ``WorldObject3D`` are built. These classes inherit from ``WorldObject`` 
and particularize the entities for 2D and 3D simulations. For the moment, 3D objects inherit from ``WorldObject3D`` and 2D entities inherit from ``WorldObject2D``. 
Nonetheless', this implementation is temporal and, in future versions, it is expected that any entity will be decoupled from the type of physics and graphics it employs in 
a simulation. In this way, ``WorldObject2D`` and ``WorldObject3D`` would be removed and only ``WorldObject`` would act as base class. 

.. todo::
    Several objects (robot, light_source, etc) have two classes devoted to 2D and 3D physics and graphics. This implementation 
    is temporal until the 2D/3D standardization is fulfilled. The aim is that there is a single class for each entity and it is 
    the physics engine the responsible for loading and executing the correct object model, physics and graphics.

The entities that are currently implemented in the simulator are gathered in the following table:

+-----------------------+-------------------------+----------------------------------------------------------+---------------------------+
|   Entity Name         |   Python Class          |     Description                                          |      Image                |
+-----------------------+-------------------------+----------------------------------------------------------+---------------------------+
|   robot               |   Robot                 |                                                          | .. image:: figs/robot.png |
|                       |                         |                                                          |    :width: 50%            |
+-----------------------+-------------------------+----------------------------------------------------------+---------------------------+
|    light_source       |  LightSource            |                                                          | .. image:: figs/light.png |
|                       |                         |                                                          |    :width: 50%            |
+-----------------------+-------------------------+----------------------------------------------------------+---------------------------+
|    wall               |  Wall                   |                                                          |                           |
|                       |                         |                                                          |                           |
+-----------------------+-------------------------+----------------------------------------------------------+---------------------------+
|    ground_area        |  Ground Area            |                                                          |                           |
|                       |                         |                                                          |                           |
+-----------------------+-------------------------+----------------------------------------------------------+---------------------------+
|    cube               |  Cube                   |                                                          |  .. image:: figs/cube.png |
|                       |                         |                                                          |     :width: 50%           |
+-----------------------+-------------------------+----------------------------------------------------------+---------------------------+
|    ball               |  Ball                   |                                                          |                           |
|                       |                         |                                                          |                           |
+-----------------------+-------------------------+----------------------------------------------------------+---------------------------+
|    epuck              |    Epuck3D              |                                                          |                           |
|                       |                         |                                                          |                           |
+-----------------------+-------------------------+----------------------------------------------------------+---------------------------+
                 
.. note::
    The class ``Robot`` and ``Robot3D`` are base classes for any simulated robot. Although these classes can be directly instantiated 
    through the config. files, it is advisable to create a separate class inheriting from them for the precise robot model (epuck, minitaur, hexapod, etc.).
    Nonetheless, if instantiated directly,  ``Robot`` and ``Robot3D`` will load the epuck model by default.

Another important aspect about entities is that they are arranged in groups. Groups are useful for several reasons. Firstly, they ease the execution of 
many operations as a group instead of applying it object by object. The clearest example is the initialization of the position of the entities, allowing 
the initialization as group, forming an spatial graph, a grid or avoiding object overlapping. Initializers will be treated later in this tutorial. Secondly, 
they allow the instantiation of homogeneous groups of entities. By homogeneous we refer to the situation in which every entity has the same characteristics, 
equipment and capabilities. For example, a group would be a swarm of 10 mobile robots with the same controller, sensors and actuators.  

Entity creation and modelling
------------------------------

There are three different ways to instantiate entities in an experiment: using directly the API, using a configuration ``dict`` and 
using a configuration file. Even though all of them are equivalent, the third option is much more used in practice for designing and 
carrying out robotics experiments. The first and second options are more suitable when building other software on top of this simulator or 
using the library inside another program. Hereafter, all of the mentioned options are explained. 
Firstly, lets address the first option: instantiating entities using the API. Even though this option is not as useful as the others, it is 
important to know it in order to understand the simulator design. There are two main steps in the entity instantiation, which are the entity 
creation (calling the class constructor) and its addition and registry within the world. 
The following example shows the process of creating two robots and a blue cube:  

Example::

>>> # Create controller and define sensors
>>> ctlr1 = BasicObstacleAvoider()
>>> ctlr2 = BasicObstacleAvoider()
>>> for ctrl in [ctlr1, ctlr2]:
>>>     ctlr.add_sensors_from_dict({"distance_sensor3D" : {"n_sectors" : 4, "range" : 1}})
>>>     ctlr.add_actuators_from_dict({"joint_velocity_actuator" : {"joint_ids" : [0, 1], "max_velocity" : 13}})
>>> # Create objects instances
>>> robot1 = Epuck3D(np.array([0,0]), 0, controller=ctlr1)
>>> robot2 = Epuck3D(np.array([1,0]), 3.14, controller=ctlr2)
>>> cube = Cube(np.array([2,2]), 0, color='blue', mass=1, side_len=0.2)
>>> # Add objects to world
>>> world.register_entity('swarm_member_1', robot1, group='swarm')
>>> world.register_entity('swarm_member_2', robot2, group='swarm')
>>> world.register_entity('cube_1', cube, group='cubes')

The first steps imply creating the robot controller, which is a basic obstacle avoidance controller, and the activation of the sensors 
and actuators. For the moment, lets ignore this steps as they are treated later in this tutorial. Secondly, the robots and the cube are 
created using the corresponding Python class (imports not shown in the example). The robots are directly initialized at (0,0) and (1,0) and 
with orientations 0 and :math:`\pi` radians. The cube is created at (2,2). 
The last step is to register the entities in the world. This is accomplished using the ``register_entity`` method of the ``World`` class. 
The arguments are the entity name, the python instance and the group to which it belongs. Notice that the two robots belong to the same group.

The second way of creating entities is using ``dict`` objects to gather all the configuration details. Thereafter, once this dictionary has been 
created, the method ``build_from_dict`` of the class ``World`` is used. The configuration details and an explanation of the possible fields can 
be consulted at the `Configuration Files <configuration_files.html>`__ section. At this point, lets recreate the previous example with two robots 
and a blue cube:

Example::

    >>> world_cfg = {
    >>>     "engine" : "3D",
    >>>     "height": 10,
    >>>     "width":  10,
    >>>     "objects" : {
    >>>         "swarm" : {
    >>>             "type" : "epuck",
    >>>             "num_instances" : 2,
    >>>             "controller" : "basic_obstable_avoider",
    >>>             "sensors" : {
    >>>                 "distance_sensor3D" : {"n_sectors" : 4, "range" : 1}
    >>>            },  
    >>>            "actuators" : {
    >>>                "joint_velocity_actuator" : {"joint_ids" : [0, 1], "max_velocity" : 13}
    >>>            },
    >>>            "initializers" : {
    >>>                "positions" : {"name" : "fixed", "params" : {"fixed_values": [[0, 0], [1, 0]]}}
    >>>                "orientations" : {"name" : "fixed", "params" : {"fixed_values": [0, 3.14]}}
    >>>            },
    >>>         },
    >>>            "cubes" : {
    >>>               "type" : "cube", 
    >>>                "initializers" : {
    >>>                    "positions" : {"name" : "random_uniform",  "params" : {"low" : [-1, -1], "high" : [1, 2], "size" : 2}}
    >>>                },
    >>>                "params" : {"mass" : 1, "side_len" : 0.2, "color" : "blue"}
    >>>            }
    >>>     }
    >>> }
    >>> world.build_from_dict(world_cfg)

The only configuration distinction is that we have already introduced the entity initializers. More precisely, we have used the 
:py:class:`spike_swarm_sim.utils.initializers.FixedInitializer`, that deterministically assigns the given fixed values to the 
positions and orientations of the objects in the group.

The third option is very similar to using a ``dict`` gathering the configuration. Specifically, it is about using configuration files gathering 
the configuration. Moreover, the configuration files not only encompass the world configuration but also the neural topology and optimization algorithm (if any). 
For the moment, the format of the configuration files is ``json`` and the configuration files are stored in the ``spike_swarm_sim/config`` directory. 
The meaning of the JSON field is explained at the `Configuration Files <configuration_files.html>`__ section. 
When using configuration files, the only way of executing the simulator is via the main.py file. More precisely, provided that the configuration file name is ``config_example.json``, 
the command line execution would be:

Example::

>>> python main.py -f config_example -R 

where ``-R`` means that the simulation is run in visual mode and ``-f`` is the config. file name. For the description of the rest of command line arguments see XXXXXXXX. 

Finally, another important aspect when creating entities is their model definition. Each entity class has an attribute called ``model_file`` that states the name of the 
file defining the 2D or 3D model of the object. The model must be previously designed, coded and stored in ``spike_swarm_sim/objects/models``. In the case of the 3D 
entities using the ``pybullet`` based engine (the only 3D engine currently available), the 3D model must be defined as an `URDF <http://wiki.ros.org/urdf>`_ file. 
These aside from from  defining the geometry, inertia, masses, and so on, it also establishes the links and joints of robots. 
See `epuck.urdf <_static/epuck.urdf>`_ for an example of creating a simplified epuck through an URDF file. 
On the contrary, for creating 2D entities for ``pymunk`` based physics engine, we created a simple yet easily extensible JSON module syntax. An example of a JSON 
file modelling a simplified 2D epuck is shown in `epuck.json <_static/epuck.json>`_ .

.. todo::
    Explain the details the syntax of the JSON model files of 2D entities. To be done when 2D/3D standardization is finished.

Initializers
-------------

Now, lets return to the entity initializers. Initializer are python classes that noticeably ease the selection of entity initial states of 
the positions and the orientations. For example it allows to sample positions uniformly within a square without physical overlapping or 
sample the 2D coordinates of robots in a swarm as a 2D spatial random graph (assuring swarm compactness). 
Initializers work at the group level, meaning that they initialize the positions and orientations considering whole groups of entities. 
This remarkably useful for avoiding initial overlapping or creating compact swarms of robots. The mapping between group names and initializers 
is an attribute of the class ``World`` that holds all the groups and entities. This attribute is called ``initializers`` of type ``dict``.   
An pseudocode example of this attribute is exposed below:

Example::

    >>> world.initializers = {
    >>>     'swarm' : {
    >>>         'positions' : Initializer1,
    >>>         'orientations' : Initializer2,
    >>>     },
    >>>     'cubes' : {
    >>>         'positions' : Initializer3,
    >>>     }        
    >>> }

Note that ``Initializer1``, ``Initializer2`` and  ``Initializer3`` are not actual classes. In practice, they should be instances of any of the 
available Initializer classes.

The initializer classes that are currently implemented within the simulator are:

+-----------------------+---------------------------+---------------------------------------------------------------------------------------------+
| **Reference Name**    |   **Python Class**        |     **Description**                                                                         |
+-----------------------+---------------------------+---------------------------------------------------------------------------------------------+
|    fixed              |  FixedInitializer         | Initializes the positions or orientations always at the given fixed values.                 |
+-----------------------+---------------------------+---------------------------------------------------------------------------------------------+
|    fixed_random       |  FixedRandomInitializer   | Randomly initializes the positions or orientations from a list of fixed possible values.    |
+-----------------------+---------------------------+---------------------------------------------------------------------------------------------+
|    random_uniform     |  RandomUniformInitializer | Randomly initializes objects positions within a rectangle area (positions) or a segment     |
|                       |                           | orientations().                                                                             |
+-----------------------+---------------------------+---------------------------------------------------------------------------------------------+
|    random_circle      |  RandomCircleInitializer  | Randomly initializes objects positions within a circle area.                                |
+-----------------------+---------------------------+---------------------------------------------------------------------------------------------+
| random_circumference  |  RandomCircumference      | Randomly initializes objects positions embedded in a circumference.                         |
+-----------------------+---------------------------+---------------------------------------------------------------------------------------------+
|    random_graph       |  RandomGraphInitializer   | Randomly initializes objects positions as a random spatial graph.                           |
+-----------------------+---------------------------+---------------------------------------------------------------------------------------------+
|                       |                           | Deterministically initializes objects positions within a 2D regular lattice or grid.        |
|                       |                           |                                                                                             |
|    grid               |  GridInitializer          | .. todo::                                                                                   |
|                       |                           |    Not implemented yet                                                                      |
|                       |                           |                                                                                             |
|                       |                           |                                                                                             |
+-----------------------+---------------------------+---------------------------------------------------------------------------------------------+

Finally, lets learn how initializers are used. In case of using configuration files or ``dict`` objects, the initializers are used as in the  last 
example of the previous section, introducing for each object the initializer reference name and the parameters (see the API for an explanation of the 
class arguments). On the contrary, in order to understand its use with the API, lets observe the following example:

Example::

>>> n_robots = 5 # Number of robots in the group 'swarm'.
>>> # Create and register initializers 
>>> ini_ori = RandomUniformInitializer(n_robots, low=0, high=6.28, size=1, engine='3D', variable='orientations')
>>> ini_pos = RandomUniformInitializer(n_robots, low=[-3,-3], high=[3,3], size=2, engine='3D',  variable='positions')
>>> world.set_initializer('swarm', ini_pos, initializer_ori=ini_ori)
>>> for i, (pos, ori) in enumerate(zip(ini_pos(), ini_ori())):
>>>     ctlr = BasicObstacleAvoider()
>>>     ctlr.add_sensors_from_dict({"distance_sensor3D" : {"n_sectors" : 4, "range" : 1}})
>>>     ctlr.add_actuators_from_dict({"joint_velocity_actuator" : {"joint_ids" : [0, 1], "max_velocity" : 13}})
>>>     ent = Robot3D(pos, ori, controller=ctlr)
>>>     world.register_entity('swarm_' + str(i), ent, group='swarm')


In the examples, we have created a group called ``'swarm'`` with 5 robots and with the following initialization:

#. **Positions**:  are sampled randomly from the hypercube :math:`\,[-3,\,3]^2`.
#. **Orientations**:  are sampled randomly from an uniform distribution :math:`\,\mathcal{U}(0, 2\pi)`.

Additionally, notice that, once created, the initializers are registered in the world and mapped to the corresponding group 
using the method :py:meth:`spike_swarm_sim.world.World.set_initializer`.




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



Communication receiver
-----------------------


Actuators
==============================


Controllers
==============================


Artificial Neural Networks
==============================




Configuration Files
==============================


Overview
--------------



World Configuration
-------------------

* *engine* (str)
    Physics and render engine to be used in the simulation. Currently, it can be either 2D and 3D.

* *height* (float)
    Height in metres of the environment arena.

* *width* (float)
    Width in metres of the environment arena.

* *objects* (dict)
    Python ``dict``gathering all the entites/objects instantiated in the arena. The entities are gathered in groups that 
    share the same characteristics. Therefore, each key-value pair maps the name of the group of entities to the configuration 
    of the group members. The most important configuration fields of each entity group are the type of the entities of the 
    group, the number of instances and the initializers. Moreover, in the case of robot entities, the controller, the enabled sensors and 
    the employed actuators are also stated here. 

    Example::
    
    >>> "robotA" : {
    >>>     "type" : "robot",
    >>>     "num_instances" : 2,
    >>>     "controller" : "neural_controller",
    >>>     "sensors" : {
    >>>         "yellow_light_sensor" : {"n_sectors" : 4, "range" : 5},
    >>>         "IR_receiver" : {"n_sectors" : 4, "range" : 2, "msg_length" : 1, "selection_scheme" : "cyclic"},
    >>>         "distance_sensor3D" : {"n_sectors" : 4, "range" : 3}
    >>>     },
    >>>     "actuators" : {
    >>>         "joint_velocity_actuator" : {"joint_ids" : [0, 1], "max_velocity" : 13},
    >>>         "IR_transmitter" : {"quantize": false, "range" : 15, "msg_length":1, "K" : 2}
    >>>     },
    >>>     "initializers" : {
    >>>         "positions" : {"name" : "random_uniform",  "params" : {"low" : [-1, -1], "high" : [1, 1], "size" : 2}},
    >>>         "orientations" : {"name" : "random_uniform",  "params" : {"low":0, "high" : 6.28, "size" : 1}}
    >>>     },
    >>>     "perturbations" : {
    >>>         "stimuli_inhibition" : {"affected_robots": [1], "stimuli" : "IR_receiver", "replace_value" : 0}
    >>>     },
    >>>     "params" : {"trainable" : true}
    >>>   },
    >>>  "yellow_light" : {
    >>>      "type" : "light_source",
    >>>      "num_instances": 1,
    >>>      "controller" : null,
    >>>      "initializers" : {
    >>>          "positions" : {"name" : "fixed", "params" : {"fixed_values": [[0, 3]]}}
    >>>      },
    >>>      "params" : {"range" : 20, "color" : "yellow"}
    >>>  }

    +----------------------+-----------------------------+---------------------+-----------+----------------------------+
    | Sensor Reference     |  Sensor Class               | Parameter           |  Default  |   Description              |          
    +----------------------+-----------------------------+---------------------+-----------+----------------------------+
    |                      |                             |  n_sectors          |    4      |                            |
    | distance_sensor      |   DistanceSensor            +---------------------+-----------+----------------------------+
    |                      |                             |   range             |    2      |                            |
    +----------------------+-----------------------------+---------------------+-----------+----------------------------+
    |                      |                             |  n_sectors          |    4      |                            |
    | | IR_receiver        | | IRCommunicationReceiver   +---------------------+-----------+----------------------------+
    |                      | | BufferedIRCommRX          |   range             |    2      |                            |
    |                      |                             +---------------------+-----------+----------------------------+
    |                      |                             |   msg_length        |    1      |                            |
    |                      |                             +---------------------+-----------+----------------------------+
    |                      |                             |   max_hops          |    10     |                            |
    |                      |                             +---------------------+-----------+----------------------------+
    |                      |                             |   selection_scheme  |  "cyclic" |                            |
    +----------------------+-----------------------------+---------------------+-----------+----------------------------+
    |                      |                             |  n_sectors          |    4      |                            |
    |   light_sensor       |   LightSensor               +---------------------+-----------+----------------------------+
    |                      |                             |   range             |    2      |                            |
    +----------------------+-----------------------------+---------------------+-----------+----------------------------+
    | ground_sensor        |    GroundSensor             |                     |           |                            |
    +----------------------+-----------------------------+---------------------+-----------+----------------------------+
    | color_sensor         |     ColorSensor             |                     |           |                            |
    +----------------------+-----------------------------+---------------------+-----------+----------------------------+

    
Topology Configuration
-----------------------

* **dt** (float)
    Euler time step used to iteratively solve the differential equations of the neurons.

* *time_scale* (float)

* *neuron_model* (str)
    ==============  ================   ======================================
    Reference Name  Python class        Description
    ==============  ================   ======================================
    rate_model      RateModel           aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    adex            AdExModel           aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    izhikevich      IzhikevichModel     aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    lif             LIFModel            aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    exp_lif         ExpLIFModel         aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    morris_lecar    MorrisLecarModel    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    ==============  ================   ======================================

    .. note::
        ``MorrisLecarModel`` and ``LIFModel`` classes do exist in the simulator but cannot be used for the 
        moment.

* *synapse_model*  (str)
    ==============  ================   ======================================
    Reference Name  Python class        Description
    ==============  ================   ======================================
    static          RateModel           aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    adex            AdExModel           aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    ==============  ================   ======================================

* *stimuli* (dict)

    Example::
    
    >>> "stimuli": {
    >>>     "I1" : {"n" : 4, "sensor" : "yellow_light_sensor"},
    >>>     "I2" : {"n" : 4, "sensor" : "red_light_sensor"},
    >>>     "I3" : {"n" : 4, "sensor" : "distance_sensor3D"},
    >>>     "I4" : {"n" : 1, "sensor" : "task_sensor"},
    >>>     "I5" : {"n" : 4, "sensor" : "IR_receiver:msg"}
    >>> }

* *ensembles* (dict)
    Python ``dict`` defining the neuron ensembles or layers of the architecture. For the moment, 
    all the ensembles' neurons are based on the same neuron model established in the ``neuron_model`` 
    field. Nonetheless, the neurons of each ensemble can have different neuron parameters (e.g. one
    can define an ensemble of fast rate model neurons with a small :math:`\tau` and another ensemble composed 
    by slow neurons with a large :math:`\tau`). The ``dict`` maps ensemble names to ensemble parameters. There 
    is only one compulsory parameter named as ``n`` that represents the number of neurons in the ensemble. 
    The other parameters are gathered within the values of a subdict with key ``"params"``. 
    The following example shows the ensemble configuration of an architecture with rate models and four  ensembles 
    (``H1``, ``H2``, ``OUT_COMM`` and ``OUT_MOT``):

    Example::

    >>> "neuron_model" : "rate_model",
    >>> "ensembles": {
    >>>     "H1" : {"n" : 10, "params" : {"tau" : 2, "bias" : -1, "gain" : 0.75}},
    >>>     "H2" : {"n" : 5, "params" : {"tau" : 1, "bias" : -0.5}},
    >>>     "OUT_COMM" : {"n" : 1, "params" : {"tau": 0.5}},
    >>>     "OUT_MOT" : {"n" : 2, "params" : {"tau": 0.5, "activation" : "tanh"}}
    >>> }

    It can be observed in the example, the parameters of the ``"params"`` field  are optional and 
    it they are not defined, the default value is used. Moreover, the parameters shown in the example 
    are specifically used only in the case of the ``rate_model``. For other neuron models, the parameters 
    are different according to the neuron requirements. The following table gathers the possible parameters 
    of each of the available neuron models: 

    +--------------+---------------+----------------+-----------+----------------------------+
    | Neuron Model | Parameter     |  Math Notation | Default   |   Description              |          
    +--------------+---------------+----------------+-----------+----------------------------+
    |              |   tau         |  :math:`\tau`  |   1.0     | Neuron's time constant     |
    |  rate_model  +---------------+----------------+-----------+----------------------------+
    |              |   bias        |  :math:`\beta` |   0.0     |  Neuron bias or offset     |
    |              +---------------+----------------+-----------+----------------------------+
    |              |   gain        |  :math:`g`     |   1.0     | Neuron gain                |
    |              +---------------+----------------+-----------+----------------------------+
    |              | activation    |  :math:`f`     | "sigmoid" | Neuron activation function |
    +--------------+---------------+----------------+-----------+----------------------------+
    |              |   tau_w       |  :math:`\tau_w`|   1.0     | Neuron's time constant     |
    |  adex        +---------------+----------------+-----------+----------------------------+
    |              |   tau_m       |  :math:`\tau_m`|   0.0     |  Neuron bias or offset     |
    |              +---------------+----------------+-----------+----------------------------+
    |              |   V_rest      |  :math:`g`     |   1.0     | Neuron gain                |
    |              +---------------+----------------+-----------+----------------------------+
    |              |   V_reset     |  :math:`f`     | "sigmoid" | Neuron activation function |
    |              +---------------+----------------+-----------+----------------------------+
    |              |   A           |  :math:`A`     | "sigmoid" | Neuron activation function |
    |              +---------------+----------------+-----------+----------------------------+
    |              |   B           |  :math:`B`     | "sigmoid" | Neuron activation function |
    |              +---------------+----------------+-----------+----------------------------+
    |              |   theta_rest  |  :math:`A`     | "sigmoid" | Neuron activation function |
    |              +---------------+----------------+-----------+----------------------------+
    |              |   R           |  :math:`R`     | "sigmoid" | Neuron activation function |
    |              +---------------+----------------+-----------+----------------------------+
    |              |refractoriness |  :math:`A`     | "sigmoid" | Neuron activation function |
    +--------------+---------------+----------------+-----------+----------------------------+
    |              |   A           |  :math:`a`     |   0.02    | Neuron's time constant     |
    |  izhikevich  +---------------+----------------+-----------+----------------------------+
    |              |   B           |  :math:`b`     |   0.2     |  Neuron bias or offset     |
    |              +---------------+----------------+-----------+----------------------------+
    |              |   C           |  :math:`c`     |   -65     | Neuron gain                |
    |              +---------------+----------------+-----------+----------------------------+
    |              |   D           |  :math:`d`     |    8.0    | Neuron activation function |
    +--------------+---------------+----------------+-----------+----------------------------+
    |              |   tau         |  :math:`\tau`  |   20      | Neuron's time constant     |
    |  exp_lif     +---------------+----------------+-----------+----------------------------+
    |              |   R           |  :math:`R`     |   1       |  Neuron bias or offset     |
    |              +---------------+----------------+-----------+----------------------------+
    |              |   v_rest      |  :math:`c`     |  -65.0    | Neuron gain                |
    |              +---------------+----------------+-----------+----------------------------+
    |              |   time_refrac |  :math:`d`     |   10.     | Neuron activation function |
    |              +---------------+----------------+-----------+----------------------------+
    |              |   thresh      |  :math:`d`     |  -40.0    | Neuron activation function |
    +--------------+---------------+----------------+-----------+----------------------------+


* **synapses** (dict)

    Python ``dict`` defining the groups of synapses connecting the neurons and input nodes in defined in 
    ``stimuli`` and ``ensembles`` fields. Each key-value of the synapses dictionary defines the connections 
    between two ensembles (either of neurons or input nodes) and maps the synapse name to the synapse configuration. 
    The synapse configuration encompasses several parameters that structurally define the connection. Among them, 
    the ``"pre"`` and ``"post"`` parameters are compulsory and define the pre-synaptic and post-synaptic ensembles. 
    These ensemble names have to be defined either in the ``stimuli`` or ``ensembles`` fields. Moreover, notice that 
    each synapse entry can define a set of connections instead of a single one in the cases in which the number of 
    neurons or nodes in the ensembles is greater than 1. The following extract of code extends the example exposed 
    in the ``ensembles`` and ``stimuli`` explanation, showing how to connect the ensembles:

    Example::

    >>> "synapses" :  {
    >>>    "I1-H1" : {"pre":"I1", "post":"H1", "trainable":true, "p":1.0},
    >>>    "I2-H1" : {"pre":"I2", "post":"H1", "trainable":true, "p":1.0},
    >>>    "I3-H1" : {"pre":"I3", "post":"H1", "trainable":true, "p":1.0},
    >>>    "I4-H1" : {"pre":"I4", "post":"H1", "trainable":true, "p":1.0},
    >>>    "I5-H1" : {"pre":"I5", "post":"H1", "trainable":true, "p":1.0},
    >>>    "H1-H2" : {"pre":"H1","post":"H2", "trainable":true, "p":1.0},
    >>>    "H2-OUT_COMM" : {"pre":"H2","post":"OUT_COMM", "trainable":true, "p":1.0},
    >>>    "H2-OUT_MOT" : {"pre":"H2","post":"OUT_MOT", "trainable":true, "p":1.0},
    >>> }

    Additionally, the following table gathers the possible parameters to be defined in a synapse entry. 
    Notice that the ``neuroTX`` parameter is only valid when using spiking neural networks 
    (spiking neurons + dynamic synapses). It is worth mentioning the function of ``p``, which represents 
    the connection probability. Its meaning is that each synapse joining two ensembles will be created 
    with a probability ``p``. This parameter is specially important when building large unstructured 
    neural sparse meshes in which the connectivity is, say, 25%. In regular ANNs it is normally fixed 
    to 1.0 (fully connected).

    +---------------+----------------+------------------------------------------------------------+
    | **Parameter** | **Default**    |   **Description**                                          |          
    +---------------+----------------+------------------------------------------------------------+
    |   pre         |   Compulsory   | Name of the pre-synaptic neuron ensemble.                  |
    +---------------+----------------+------------------------------------------------------------+
    |   post        |   Compulsory   | Name of the post-synaptic neuron ensemble.                 |
    +---------------+----------------+------------------------------------------------------------+
    |   p           |     1.0        | Connection probability among ``pre`` and ``post`` neurons. |
    +---------------+----------------+------------------------------------------------------------+
    |   neuroTX     |  "AMPA+NDMA"   | Type of synapse neurotransmitter. Only valid if            |
    |               |                | dynamic_synapse is used. Possible values: "AMPA+NDMA",     |
    |               |                | "AMPA", "GABA" or "NDMA".                                  |
    +---------------+----------------+------------------------------------------------------------+
    | trainable     |    True        | Flag indicating if the weights of the synapses             |
    |               |                | can be optimized. Currently not used.                      |
    +---------------+----------------+------------------------------------------------------------+


* *outputs* (dict)
    The outputs field states which of the previously defined ensembles are output layers of the architecture.
    It also maps these output layers, responsible of generating actions, to the corresponding actuators.  
    The ``dict`` maps output names (do not confuse with ensemble names) with the output configuration. 
    The output configuration states the ensemble name (has to be defined) and the actuator name (has to be an 
    implemented actuator). The following example defines ``OUT_COMM`` and ``OUT_MOT`` neuron ensembles as outputs:

    Example::

    >>> "outputs" : {
    >>>    "outA" : {"ensemble" : "OUT_MOT", "actuator" : "joint_velocity_actuator", "enc": "real"},
    >>>    "outB" : {"ensemble" : "OUT_COMM", "actuator" : "IR_transmitter", "enc": "real"}
    >>> }

    +----------------------------+-----------------------------+----------------------------------------------------------------+
    | **Actuator Name**          | **Actuator Class**          |   **Description**                                              |          
    +----------------------------+-----------------------------+----------------------------------------------------------------+
    |   joint_velocity_actuator  |   JointVelocityActuator     | Actuator for the velocity control of joints.                   |
    +----------------------------+-----------------------------+----------------------------------------------------------------+
    |   joint_position_actuator  |   JointPositionActuator     | Actuator for the position control of joints.                   |
    +----------------------------+-----------------------------+----------------------------------------------------------------+
    |   IR_transmitter     |   CommunicationTransmitter  | IR-based Communication transmitter.                            |
    +----------------------------+-----------------------------+----------------------------------------------------------------+
    |   led_actuator             |   LedActuator3D/LedActuator | Actuator for controlling the LEDs of a robot.                  |
    +----------------------------+-----------------------------+----------------------------------------------------------------+
    |   grasp_actuator           |   GraspActuator             | | High level simplified actuator for grasping and dropping     |
    |                            |                             | | small lightweight objects (TODO: for the moment only cubes). |
    +----------------------------+-----------------------------+----------------------------------------------------------------+
 
    .. note::
        Just like the sensor classes, some actuator classes have a 2D and 3D implementation. This feature is temporal 
        until 2D and 3D classes are standardized.

    .. note::
        The ``IR_transmitter`` actuator name is provisional and it will eventually changed to a more descriptive name.

* *encoding* (dict)

* *decoding* (dict)
    
    Example::

    >>> "decoding" : {
    >>>     "outA" : {"scheme" : "IdentityDecoding", "params" : {"is_cat" : false}},
    >>>     "outB" : {"scheme" : "IdentityDecoding", "params" : {"is_cat" : false}}
    >>> }

* *learning_rule* (dict)


Algorithm Configuration
------------------------


Examples
--------------