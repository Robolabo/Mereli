.. _tutorial_entities:


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
