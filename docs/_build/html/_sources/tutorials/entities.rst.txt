.. _tutorial_entities:


Entities
==============================


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


