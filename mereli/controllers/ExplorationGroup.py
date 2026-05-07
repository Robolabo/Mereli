import os

import numpy as np

from mereli.controllers import RobotController
from mereli.register import controller_registry, controllers


@controller_registry(name="stop")
class StopController(RobotController):
    """Skill basica: detener el robot mientras espera una orden."""

    def __init__(self, *args, **kwargs):
        super(StopController, self).__init__(*args, **kwargs)
        self.flag = True

    def step(self, state, reward=0.0):
        self.get_actuator("joint_velocity_actuator").action = np.array([0.0, 0.0])


@controller_registry(name="orient_red_light")
class OrientRedLightController(RobotController):
    """Skill basica: rotar en el sitio hasta mirar hacia una luz roja."""

    def __init__(self, *args, angular_speed=0.2, alignment_tolerance=0.05, **kwargs):
        super(OrientRedLightController, self).__init__(*args, **kwargs)
        self.angular_speed = angular_speed
        self.alignment_tolerance = alignment_tolerance
        self.flag = True
        self.centered = False

    def step(self, state, reward=0.0):
        ls_read = self.get_sensor_reading("red_light_sensor")
        action = np.array([0.0, 0.0])
        light_centered = False

        if np.max(ls_read) == 0.0: # Si no ve luz roja, gira en el sitio para buscarla.
            action = self.angular_speed * np.array([1.0, -1.0])
        else: # Si ve luz roja, calcula si esta centrada o si tiene que girar a la izquierda o derecha para centrarla.
            light_left = np.sum(ls_read[[7, 6, 5, 4]])
            light_right = np.sum(ls_read[[0, 1, 2, 3]])
            front_light = np.sum(ls_read[[0, 7]])

            if front_light >= max(light_left, light_right): # Si la luz frontal es la mas intensa, consideramos que esta centrada aunque haya algo de ruido en los laterales.
                action = np.array([0.0, 0.0])
                light_centered = True
            elif abs(light_right - light_left) <= self.alignment_tolerance: # Si la diferencia entre luz izquierda y derecha es pequeña, consideramos que esta centrada aunque no haya mucha luz frontal.
                action = np.array([0.0, 0.0])
                light_centered = True
            elif light_right > light_left: # Si la luz es mas intensa a la derecha, gira a la derecha para centrarla.
                action = self.angular_speed * np.array([-1.0, 1.0])
            else:
                action = self.angular_speed * np.array([1.0, -1.0])

        if light_centered and not self.centered: # Solo imprimimos el mensaje la primera vez que detectamos que la luz esta centrada, para no spamear la consola.
            print(f"[orient_red_light] Luz roja centrada en step {self.controller_owner.t}")
        self.centered = light_centered

        self.get_actuator("joint_velocity_actuator").action = action


@controller_registry(name="approach_red_light")
class ApproachRedLightController(RobotController):
    """Skill basica: avanzar recto hasta quedar cerca de una luz roja."""

    def __init__(self, *args, forward_speed=0.5, near_threshold=0.9, **kwargs):
        super(ApproachRedLightController, self).__init__(*args, **kwargs)
        self.forward_speed = forward_speed
        self.near_threshold = near_threshold
        self.flag = True
        self.light_found = False

    def step(self, state, reward=0.0):
        ls_read = self.get_sensor_reading("red_light_sensor")

        if np.max(ls_read) >= self.near_threshold: # Si la luz roja es lo suficientemente intensa, consideramos que estamos cerca y paramos.
            action = np.array([0.0, 0.0])
            if not self.light_found:
                print(f"[approach_red_light] Luz encontrada en step {self.controller_owner.t}")
            self.light_found = True
        else: # Si no estamos cerca, avanzamos recto para acercarnos a la luz.
            action = self.forward_speed * np.array([1.0, 1.0])
            self.light_found = False

        self.get_actuator("joint_velocity_actuator").action = action


@controller_registry(name="exploration_group")
class ExplorationGroupController(RobotController):
    """Controller de subsumpcion para la futura arquitectura jerarquica.

    La capa de supervivencia se evalua siempre mediante ``basic_obstacle_avoider``.
    Debajo corre una tarea secundaria configurable, que solo puede controlar las
    ruedas cuando no hay obstaculos cerca.
    """

    def __init__(self, *args, survival_task="basic_obstacle_avoider",
        secondary_task="navigate", survival_params=None, secondary_params=None, **kwargs):
        super(ExplorationGroupController, self).__init__(*args, **kwargs)

        # Permitimos configurar las dos capas desde el JSON 
        survival_params = survival_params or {}
        secondary_params = secondary_params or {}

        # Guardamos los nombres para poder registrar que capa esta mandando en
        # cada instante y para que, en el futuro, el LLM local pueda cambiar la
        # tarea secundaria sin tocar la capa de supervivencia.
        self.survival_task = survival_task
        self.secondary_task = secondary_task
        self.current_task = secondary_task  # CUIDADO: mas adelante que no ponga avoid aqui y el llm se ralle

        # Cada rutina escribe provisionalmente su accion en el actuador. Antes de
        # ejecutar la siguiente, copiamos esa accion aqui para que coordinate()
        # pueda escoger la que corresponda por prioridad.
        self.activations = {
            "survival": np.zeros(2),
            "secondary": np.zeros(2),
        }

        # CURRENT_EXP_FOLDER y despues genera trayectoria.png leyendo este CSV.
        self.output_dir = os.environ.get("CURRENT_EXP_FOLDER", "outputs")
        self.log_name = os.path.join(self.output_dir, "recorrido_robot.csv")

        if not os.path.exists(self.log_name):
            with open(self.log_name, "w") as f:
                f.write("step,x,y,tarea\n")

        # Instanciamos los controllers reales desde el registro global de Mereli.
        # En esta primera version la capa secundaria es navigate, pero mas tarde
        # podra ser la skill elegida por el LLM individual.
        self.survival_controller = controllers[survival_task](**survival_params)
        self.secondary_controller = controllers[secondary_task](**secondary_params)

    def step(self, state, reward=0.0):
        # 1. Ejecutar siempre la capa inconsciente de seguridad.
        self.survival_controller.step(state)
        self.activations["survival"] = np.array(
            self.get_actuator("joint_velocity_actuator").action
        )

        # 2. Ejecutar tambien la tarea secundaria. Su accion solo se aplicara si
        # coordinate() comprueba que no hay obstaculos activando la supervivencia.
        self.secondary_controller.step(state)
        self.activations["secondary"] = np.array(
            self.get_actuator("joint_velocity_actuator").action
        )

        action = self.coordinate()
        self.log_robot_state()
        return action

    def log_robot_state(self):
        # Guardamos solo lo minimo para que main.py pueda pintar la trayectoria.
        # La columna tarea permite ver cuando manda avoid obstacle y cuando manda
        # la skill secundaria.
        t = self.controller_owner.t
        x, y = self.controller_owner.position[0], self.controller_owner.position[1]

        if t % 10 == 0:
            with open(self.log_name, "a") as f:
                f.write(f"{t},{x:.3f},{y:.3f},{self.current_task}\n")

    def coordinate(self):
        # Prioridad absoluta: si basic_obstacle_avoider levanta flag, inhibe la
        # tarea secundaria y toma el control de las ruedas.
        if self.survival_controller.flag:
            self.current_task = self.survival_task
            action_wheels = self.activations["survival"]
        else:
            self.current_task = self.secondary_task
            action_wheels = self.activations["secondary"]

        # Dejamos la accion final escrita tanto en el actuador como en el return,
        # siguiendo el estilo de otros controllers de Mereli.
        self.get_actuator("joint_velocity_actuator").action = np.array(action_wheels)
        return {"joint_velocity_actuator": self.get_actuator("joint_velocity_actuator").action}

    def reset(self):
        # Las rutinas hijas se crean antes de que el robot exista por completo.
        # En reset les conectamos el owner real para que puedan leer sensores,
        # actuadores, posicion y tiempo de simulacion.
        self.survival_controller.controller_owner = self.controller_owner
        self.secondary_controller.controller_owner = self.controller_owner
        self.survival_controller.reset()
        self.secondary_controller.reset()
