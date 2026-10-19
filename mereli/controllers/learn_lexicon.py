""" Mereli controller of the lexicon experiments of Sendra-Arranz & Gutiérrez, "Cultural
evolution of perceptually grounded compositional lexicons in swarm robotics systems",
Applied Soft Computing 202 (2026) 115879.

The language-game logic lives in the standalone package ``swarm_language`` (no Mereli
dependency). This controller only connects it to Mereli: it reads the ``object_sensor``,
collects the messages of the robot's neighbors (robots closer than 1 m, see
``World.update_neighbor_matrix``) and moves the robot with the basic obstacle avoider.
"""
import logging
import weakref
from swarm_language import LanguageAgent, swarm_metrics
from swarm_language.metrics import REPORT_HEADER, report_line
from mereli.controllers import RobotController, BasicObstacleAvoider
from mereli.register import controller_registry


@controller_registry(name='build_lexicon_B')
class BuildLexiconControllerB(RobotController):
    """ Robot exploring the arena while playing language games with its neighbors.

    Required sensors: ``object_sensor`` and ``distance_sensor`` (obstacle avoidance).
    Required actuators: ``joint_velocity_actuator``.

    All parameters are optional, and the defaults reproduce the published experiments.
    Example of controller entry in a JSON config::

        "controller" : {
            "name" : "build_lexicon_B",
            "params" : {
                "speak_prob" : 0.9,
                "faulty_color_robots" : [22, 23, 24],
                "lexicon" : {"association_threshold" : 0.4},
                "discrimination" : {"threshold" : 0.4}
            }
        }

    :param float speak_prob: probability of broadcasting a word at each control step.
    :param list faulty_color_robots: ids (``robot.id``) of the robots with a faulty color
        sensor (experiment R2).
    :param list faulty_shape_robots: ids of the robots with faulty shape perception.
    :param bool perspective_noise: perspective-dependent shape perception (experiment R1).
    :param dict lexicon: parameters of :class:`swarm_language.Lexicon`.
    :param dict discrimination: parameters of :class:`swarm_language.PatternDetector`.
    :param int report_every: every this many simulation steps, the first robot of the group
        prints the lexicon emergence status of the swarm (LSS, CSS and lexicon sizes). Set it
        to 0 to disable. Printing does not affect the simulation.
    """
    # All the controllers of this type, used to compute swarm-level metrics.
    _instances = weakref.WeakSet()

    def __init__(self, controller_owner=None, speak_prob=0.9, faulty_color_robots=(),
                 faulty_shape_robots=(), perspective_noise=False, lexicon=None, discrimination=None,
                 report_every=1000):
        super(BuildLexiconControllerB, self).__init__(controller_owner=controller_owner)
        self.obstacle_avoider = BasicObstacleAvoider()
        self.faulty_color_robots = list(faulty_color_robots)
        self.faulty_shape_robots = list(faulty_shape_robots)
        self.agent = LanguageAgent(speak_prob=speak_prob, perspective_noise=perspective_noise,
                                   lexicon_params=lexicon, discrimination_params=discrimination)
        self.report_every = report_every
        self._last_report = 0
        BuildLexiconControllerB._instances.add(self)

    # Attributes accessed by the data logger, animated plots and neighboring robots.
    @property
    def lexicon(self):
        return self.agent.lexicon

    @property
    def pattern_detector(self):
        return self.agent.pattern_detector

    @property
    def sharing(self):
        """ Message currently broadcast by the robot (None if silent). """
        return self.agent.message

    def step(self, *args, **kwargs):
        self.agent.distorted_color = self.robot.id in self.faulty_color_robots
        self.agent.distorted_shape = self.robot.id in self.faulty_shape_robots
        messages = [neigh.controller.sharing for neigh in self.robot.neighbors]
        self.agent.step(self.get_sensor_reading('object_sensor'), messages)
        self.obstacle_avoider.step(*args, **kwargs)
        if self.report_every and self.robot.gid == 0:
            self.report()

    def report(self):
        """ Prints the lexicon emergence status of the swarm (the robots of the same group in the
        same world) once every ``report_every`` steps. """
        block = self.t // self.report_every
        if block <= self._last_report:
            return
        if self._last_report == 0:
            print(REPORT_HEADER)
        self._last_report = block
        swarm = [ctrl.lexicon for ctrl in list(BuildLexiconControllerB._instances)
                 if ctrl.controller_owner is not None and ctrl.robot.group == self.robot.group
                 and ctrl.robot.physics_client is self.robot.physics_client]
        try:
            sim_time = self.t * self.robot.physics_client.dt
            print(report_line(self.t, swarm_metrics(swarm), sim_time), flush=True)
        except Exception as e:  # reporting must never stop a simulation
            logging.warning('Lexicon report failed: %r', e)

    def reset(self):
        self.obstacle_avoider.controller_owner = self.controller_owner
        self.obstacle_avoider.reset()
        self._last_report = 0
