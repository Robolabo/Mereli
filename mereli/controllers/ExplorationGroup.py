import os
import json

import numpy as np

from mereli.controllers import RobotController
from mereli.register import controller_registry, controllers
from mereli.utils import compute_angle, angle_diff


def angle_difference(a, b):
    """Diferencia angular minima entre dos angulos en radianes."""
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def get_unexplored_red_light_reading(controller):
    """Lectura roja filtrada por luces ya exploradas, si el controller padre la ofrece."""
    main_controller = getattr(getattr(controller, "controller_owner", None), "controller", None)
    if main_controller is not controller and hasattr(main_controller, "filtered_red_light_reading"):
        return main_controller.filtered_red_light_reading()[0]
    return controller.get_sensor_reading("red_light_sensor")


@controller_registry(name="stop")
class StopController(RobotController):
    """Skill basica: detener el robot mientras espera una orden."""

    def __init__(self, *args, **kwargs):
        """Inicializa la skill de parada permanente."""
        super(StopController, self).__init__(*args, **kwargs)
        self.flag = True

    def step(self, state, reward=0.0):
        """Escribe velocidad cero en las ruedas."""
        self.get_actuator("joint_velocity_actuator").action = np.array([0.0, 0.0])


@controller_registry(name="navigate")
class NavigateController(RobotController):
    """Skill basica: explorar hasta ver una luz roja nueva."""

    def __init__(self, *args, red_seen_threshold=0.05, **kwargs):
        """Guarda el umbral de deteccion roja que termina la navegacion."""
        super(NavigateController, self).__init__(*args, **kwargs)
        self.flag = True
        self.red_seen_threshold = red_seen_threshold
        self.reset()

    def reset(self):
        """Reinicia el estado de finalizacion de la skill."""
        self.done = False
        self.done_success = False
        self.done_reason = ""

    def step(self, state, reward=0.0):
        """Avanza mientras no haya una luz roja nueva por encima del umbral."""
        ls_read = get_unexplored_red_light_reading(self)
        max_red = float(np.max(ls_read))

        if max_red >= self.red_seen_threshold:
            self.done = True
            self.done_success = True
            self.done_reason = (
                f"luz roja nueva detectada durante navegacion "
                f"(max_red={max_red:.3f} >= {self.red_seen_threshold:.3f})."
            )
            self.get_actuator("joint_velocity_actuator").action = np.array([0.0, 0.0])
            return

        self.done = False
        self.done_success = False
        self.done_reason = ""
        self.get_actuator("joint_velocity_actuator").action = np.ones(2)


@controller_registry(name="orient_red_light")
class OrientRedLightController(RobotController):
    """Skill basica: rotar en el sitio hasta mirar hacia una luz roja."""

    def __init__(
        self,
        *args,
        angular_speed=0.2,
        fine_angular_speed=0.08,
        front_balance_tolerance=0.08,
        **kwargs
    ):
        """Configura las velocidades de giro y la tolerancia de centrado frontal."""
        super(OrientRedLightController, self).__init__(*args, **kwargs)
        self.angular_speed = angular_speed
        self.fine_angular_speed = fine_angular_speed
        self.front_balance_tolerance = front_balance_tolerance
        self.flag = True
        self.centered = False
        self.done = False

    def step(self, state, reward=0.0):
        """Gira el robot hasta equilibrar la lectura roja en los sensores frontales."""
        ls_read = get_unexplored_red_light_reading(self)
        action = np.array([0.0, 0.0])
        light_centered = False

        if np.max(ls_read) == 0.0: # Si no ve luz roja, gira en el sitio para buscarla.
            action = self.angular_speed * np.array([1.0, -1.0])
        else: # Si ve luz roja, calcula si esta centrada o si tiene que girar a la izquierda o derecha para centrarla.
            light_left = np.sum(ls_read[[7, 6, 5, 4]])
            light_right = np.sum(ls_read[[0, 1, 2, 3]])
            front_right = float(ls_read[0])
            front_left = float(ls_read[7])
            front_max = max(front_right, front_left)
            front_diff = front_right - front_left

            if (
                front_right > 0.0
                and front_left > 0.0
                and abs(front_diff) <= self.front_balance_tolerance * max(front_max, 1e-9)
            ):
                action = np.array([0.0, 0.0])
                light_centered = True
            elif front_right > 0.0 and front_left > 0.0:
                if front_right > front_left:
                    action = self.fine_angular_speed * np.array([-1.0, 1.0])
                else:
                    action = self.fine_angular_speed * np.array([1.0, -1.0])
            elif light_right > light_left: # Si la luz es mas intensa a la derecha, gira a la derecha para centrarla.
                action = self.angular_speed * np.array([-1.0, 1.0])
            else:
                action = self.angular_speed * np.array([1.0, -1.0])

        if light_centered and not self.centered: # Solo imprimimos el mensaje la primera vez que detectamos que la luz esta centrada, para no spamear la consola.
            print(f"[orient_red_light] Luz roja centrada en step {self.controller_owner.t}")
        self.centered = light_centered
        self.done = light_centered

        self.get_actuator("joint_velocity_actuator").action = action


@controller_registry(name="approach_red_light")
class ApproachRedLightController(RobotController):
    """Skill basica: avanzar recto hasta quedar cerca de una luz roja."""

    def __init__(self, *args, forward_speed=0.5, near_threshold=0.92, **kwargs):
        """Configura la velocidad de avance y el umbral de cercania a la luz."""
        super(ApproachRedLightController, self).__init__(*args, **kwargs)
        self.forward_speed = forward_speed
        self.near_threshold = near_threshold
        self.flag = True
        self.reset()

    def reset(self):
        """Reinicia el seguimiento del maximo de intensidad roja observado."""
        self.light_found = False
        self.done = False
        self.best_red = 0.0
        self.done_reason = ""
        self.done_success = False

    def step(self, state, reward=0.0):
        """Avanza hacia la luz roja hasta alcanzar el umbral o detectar que la ha pasado."""
        ls_read = get_unexplored_red_light_reading(self)
        max_red = float(np.max(ls_read))
        if max_red > self.best_red:
            self.best_red = max_red
        if self.controller_owner.t % 10 == 0:
            print(
                f"[approach_red_light] max_red={max_red:.3f}, "
                f"best_red={self.best_red:.3f}, "
                f"threshold={self.near_threshold:.3f}"
            )

        if max_red >= self.near_threshold: # Si la luz roja es lo suficientemente intensa, consideramos que estamos cerca y paramos.
            action = np.array([0.0, 0.0])
            if not self.light_found:
                print(f"[approach_red_light] Luz encontrada en step {self.controller_owner.t}")
            self.light_found = True
            self.done = True
            self.done_success = True
            self.done_reason = f"max_red={max_red:.3f} alcanzo threshold={self.near_threshold:.3f}."
        elif self.best_red > 0.5 and max_red < self.best_red - 0.03:
            action = np.array([0.0, 0.0])
            print(
                f"[approach_red_light] Maximo de luz pasado: "
                f"best_red={self.best_red:.3f}, max_red={max_red:.3f}. Deteniendo."
            )
            self.light_found = True
            self.done = True
            self.done_success = False
            self.done_reason = (
                f"max_red subio hasta {self.best_red:.3f} y despues bajo a {max_red:.3f}; "
                "probablemente paso cerca de la luz sin alcanzar el umbral."
            )
        else: # Si no estamos cerca, avanzamos recto para acercarnos a la luz.
            action = self.forward_speed * np.array([1.0, 1.0])
            self.light_found = False
            self.done = False
            self.done_reason = ""
            self.done_success = False

        self.get_actuator("joint_velocity_actuator").action = action


@controller_registry(name="load_blue_battery")
class LoadBlueBatteryController(RobotController):
    """Skill basica: esperar quieto hasta cargar la bateria azul."""

    def __init__(self, *args, charge_threshold=0.97, **kwargs):
        """Guarda el umbral de carga azul que completa la skill."""
        super(LoadBlueBatteryController, self).__init__(*args, **kwargs)
        self.charge_threshold = charge_threshold
        self.flag = True
        self.charged = False
        self.done = False

    def step(self, state, reward=0.0):
        """Mantiene el robot parado hasta que el sensor azul supera el umbral."""
        bat_lv = self.get_sensor_reading("blue_battery_sensor")[0]
        self.get_actuator("joint_velocity_actuator").action = np.array([0.0, 0.0])

        if bat_lv >= self.charge_threshold:
            if not self.charged:
                print("Bateria cargada")
            self.charged = True
            self.done = True
        else:
            self.charged = False
            self.done = False


@controller_registry(name="go_to_coordenadas")
class GoToCoordenadasController(RobotController):
    """Skill basica: ir por GPS a unas coordenadas objetivo conocidas."""

    def __init__(
        self,
        *args,
        target_coords=None,
        arrival_threshold=0.05,
        alpha=5.0,
        offset=0.6,
        rot_speed=0.5,
        **kwargs
    ):
        """Configura el objetivo y los parametros de convergencia a coordenadas."""
        super(GoToCoordenadasController, self).__init__(*args, **kwargs)
        self.target_coords = None if target_coords is None else np.array(target_coords[:2], dtype=float)
        self.arrival_threshold = arrival_threshold
        self.alpha = alpha
        self.offset = offset
        self.rot_speed = rot_speed
        self.flag = True
        self.reset()

    def reset(self):
        """Reinicia el estado de finalizacion de la navegacion a coordenadas."""
        self.done = False
        self.done_success = False
        self.done_reason = ""

    def step(self, state, reward=0.0):
        """Controla las ruedas para orientar y avanzar hasta las coordenadas objetivo."""
        target_coords = self.target_coords
        if target_coords is None:
            self.done = True
            self.done_success = False
            self.done_reason = "no hay coordenadas objetivo disponibles."
            self.get_actuator("joint_velocity_actuator").action = np.array([0.0, 0.0])
            return

        curr_pos = np.array(self.controller_owner.position[:2], dtype=float)
        dist_tar = float(np.linalg.norm(target_coords - curr_pos))
        if dist_tar <= self.arrival_threshold:
            self.done = True
            self.done_success = True
            self.done_reason = (
                f"objetivo alcanzado en ({target_coords[0]:.3f}, {target_coords[1]:.3f}) "
                f"con distancia {dist_tar:.3f}."
            )
            self.get_actuator("joint_velocity_actuator").action = np.array([0.0, 0.0])
            return

        desired_dir = (target_coords - curr_pos) / dist_tar
        robot_ori = self.controller_owner.orientation[-1]
        heading_ori = np.r_[np.cos(robot_ori), np.sin(robot_ori)]
        a1 = compute_angle(desired_dir)
        a2 = compute_angle(heading_ori)
        A = 1 / (1 + np.exp(-self.alpha * (dist_tar - self.offset)))
        angle = angle_diff(a1, a2)

        if angle <= 0.5:
            action = A * np.array([1.0, 1.0])
        elif np.abs(angle - np.pi) <= 0.3:
            action = np.array([-1.0, -1.0])
        elif a1 > a2:
            if a1 - a2 > np.pi:
                action = self.rot_speed * np.array([-1.0, 1.0])
            else:
                action = self.rot_speed * np.array([1.0, -1.0])
        else:
            if a2 - a1 > np.pi:
                action = self.rot_speed * np.array([1.0, -1.0])
            else:
                action = self.rot_speed * np.array([-1.0, 1.0])

        self.done = False
        self.done_success = False
        self.done_reason = ""
        self.get_actuator("joint_velocity_actuator").action = action


@controller_registry(name="annotate_red_light_position")
class AnnotateRedLightPositionController(RobotController):
    """Skill basica: anotar desde donde se percibe una luz roja cercana."""

    def __init__(self, *args, detection_threshold=0.75, **kwargs):
        """Configura el umbral de intensidad necesario para registrar una luz."""
        super(AnnotateRedLightPositionController, self).__init__(*args, **kwargs)
        self.detection_threshold = detection_threshold
        self.flag = True
        self.reset()

    def reset(self):
        """Limpia la anotacion pendiente y permite registrar una nueva luz."""
        self.light_position_printed = False
        self.annotation_printed = False
        self.done = False
        self.done_success = False
        self.done_reason = ""
        self.annotation_data = None

    def step(self, state, reward=0.0):
        """Si la luz roja filtrada es fuerte, guarda la observacion de esa luz."""
        self.get_actuator("joint_velocity_actuator").action = np.array([0.0, 0.0])
        self.print_red_light_position_once()

        ls_read = get_unexplored_red_light_reading(self)
        intensity = float(np.max(ls_read))
        if intensity < self.detection_threshold or self.annotation_printed:
            return

        sector = int(np.argmax(ls_read))
        robot_pos = self.controller_owner.position
        sensor = self.controller_owner.sensors["light_sensor"]
        global_angle = sensor.directions(self.controller_owner.orientation[-1])[sector]
        global_angle_deg = np.degrees(global_angle)
        light_id, _ = self.nearest_red_light_position(robot_pos, global_angle)

        print(
            "[annotate_red_light_position] "
            f"Posicion vision=({robot_pos[0]:.3f}, {robot_pos[1]:.3f}), "
            f"intensidad={intensity:.3f}, "
            f"sector={sector}, "
            f"angulo_global={global_angle:.3f} rad ({global_angle_deg:.1f} deg)"
        )
        self.annotation_data = {
            "step": int(self.controller_owner.t),
            "light_id": str(light_id) if light_id is not None else None,
            "light_position": self.round_position(robot_pos),
            "robot_position": self.round_position(robot_pos),
            "intensity": round(intensity, 3),
            "sector": sector,
            "global_angle_rad": round(float(global_angle), 3),
            "global_angle_deg": round(float(global_angle_deg), 1),
        }
        self.annotation_printed = True
        self.done = True
        self.done_success = True
        self.done_reason = (
            f"luz registrada en tabla de exploradas: {self.annotation_data}"
        )

    def nearest_red_light_position(self, robot_pos, target_angle=None):
        """Busca la luz roja real mas cercana para asociarla a la lectura observada."""
        nearest_id = None
        nearest_pos = None
        nearest_score = np.inf
        robot_xy = np.array(robot_pos[:2], dtype=float)
        for light_id, light_data in self.controller_owner.physics_client.luminous_objects.items():
            if light_data.get("color") != "red":
                continue
            light_pos = self.controller_owner.physics_client.get_body_position(light_id, 0)
            light_vector = np.array(light_pos[:2], dtype=float) - robot_xy
            distance = float(np.linalg.norm(light_vector))
            if target_angle is None:
                score = distance
            else:
                light_angle = float(np.arctan2(light_vector[1], light_vector[0]))
                score = abs(angle_difference(light_angle, target_angle)) + 0.01 * distance
            if score < nearest_score:
                nearest_id = light_id
                nearest_pos = light_pos
                nearest_score = score
        return nearest_id, nearest_pos

    def round_position(self, pos):
        """Redondea una posicion a dos coordenadas serializables."""
        if pos is None:
            return None
        return [round(float(pos[0]), 3), round(float(pos[1]), 3)]

    def print_red_light_position_once(self):
        """Imprime una unica vez la posicion real de debug de la primera luz roja."""
        if self.light_position_printed:
            return

        for light_id, light_data in self.controller_owner.physics_client.luminous_objects.items():
            if light_data.get("color") == "red":
                light_pos = self.controller_owner.physics_client.get_body_position(light_id, 0)
                print(
                    "[annotate_red_light_position] "
                    f"Posicion inicial luz roja=({light_pos[0]:.3f}, {light_pos[1]:.3f}, {light_pos[2]:.3f})"
                )
                self.light_position_printed = True
                return


@controller_registry(name="exploration_group")
class ExplorationGroupController(RobotController):
    """Controller de subsumpcion para la futura arquitectura jerarquica.

    La capa de supervivencia se evalua siempre mediante ``basic_obstacle_avoider``.
    Debajo corre una tarea secundaria configurable, que solo puede controlar las
    ruedas cuando no hay obstaculos cerca.
    """

    def __init__(
        self,
        *args,
        survival_task="basic_obstacle_avoider",
        secondary_task="navigate",
        survival_params=None,
        secondary_params=None,
        use_llm=False,
        macro_task="Ve a cargar la bateria azul",
        llm_model="gpt-oss:20b",
        llm_base_url="http://127.0.0.1:11434",
        decision_interval=10,
        **kwargs
    ):
        """Crea el controller jerarquico y prepara las skills disponibles."""
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
        self.use_llm = use_llm
        self.macro_task = macro_task
        self.decision_interval = decision_interval # Compatibilidad con JSON antiguos; el modo LLM actual solo decide al terminar una skill.
        self.found_red_lights = {}
        self.known_light_angle_tolerance = 0.45
        self.allowed_actions = [
            "stop",
            "navigate",
            "orient_red_light",
            "approach_red_light",
            "load_blue_battery",
            "go_to_coordenadas",
            "annotate_red_light_position",
        ]

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
        self.survival_controller = controllers[survival_task](**survival_params)
        self.routines = {}
        for action in self.allowed_actions:
            rt_params = secondary_params if action == secondary_task else {}
            self.routines[action] = controllers[action](**rt_params)

        if secondary_task not in self.routines:
            self.routines[secondary_task] = controllers[secondary_task](**secondary_params)

        self.secondary_controller = self.routines[secondary_task]

        if self.use_llm:
            self.setup_llm(llm_model, llm_base_url)

    def setup_llm(self, llm_model, llm_base_url):
        """Inicializa el cliente local del LLM y el prompt tactico fijo."""
        self.llm_thought = "Inicializando controlador tactico."
        self.llm_action = "stop"
        self.waiting_for_llm = True
        self.pending_observation = "Inicio. No hay subtarea completada todavia."
        self.memory_history = []
        self.completed_task_sequence = []

        # Estas reglas usan los nombres reales de skills que ya existen en este
        # fichero. La macro_task concreta se pasara en el contexto dinamico.
        self.llm_rules = """
        Eres el Controlador Tactico de un robot e-puck.
        Tu objetivo es cumplir la MACRO-TAREA asignada por el Cerebro Central.
        Para lograrlo, debes elegir paso a paso que habilidad ejecutar.

        ACCIONES PERMITIDAS:
        - "stop": Detiene los motores. Usala para quedarse quieto o esperar.
        - "navigate": Recorre el mapa sin rumbo. Se detiene sola cuando detecta una luz roja NO explorada.
        - "orient_red_light": Gira para centrar una luz roja detectada.
        - "approach_red_light": Avanza recto hacia una luz roja ya centrada.
        - "load_blue_battery": Se queda quieto esperando hasta que la bateria azul llegue al umbral.
        - "go_to_coordenadas": Va por GPS a las coordenadas que indiques en target_coords.
        - "annotate_red_light_position": Anota la posicion de la luz roja para registrarla y darla por encontrada.

        SIGNIFICADO DE LA OBSERVACION:
        - found_red_lights: tabla de luces rojas ya exploradas por este robot.
        - red_light_sensor_unexplored: lectura roja filtrada por el robot. Los sectores que apuntan a luces ya registradas se ponen a 0.
        - max_unexplored_red_light_sensor: maximo de red_light_sensor_unexplored. Es el valor principal para decidir si queda una luz roja nueva visible.
        - Si max_unexplored_red_light_sensor > 0, el robot percibe una senal roja que todavia considera no explorada.
        - Una luz roja solo queda explorada despues de que annotate_red_light_position termine con exito.

        POLITICA DE DECISION:
        - Usa max_unexplored_red_light_sensor y red_light_sensor_unexplored para decidir si hay una luz roja nueva hacia la que ir.
        - Si max_unexplored_red_light_sensor == 0, no hay luz roja nueva visible. Puedes continuar recorriendo el mapa. 
        - Si max_unexplored_red_light_sensor > 0 y no acabas de completar "orient_red_light", significa que hay una luz roja nueva visible pero no centrada.
        - Si en la memoria aparece que "orient_red_light" termino con exito, significa que la luz roja esta completamente centrada. 
        - Si en la memoria aparece que "approach_red_light" termino con exito porque alcanzo el umbral de cercania, significa que el robot ya esta muy cerca de una luz roja nueva y debes registrarla.
        - Si en la memoria aparece que "approach_red_light" fue detenida sin exito porque max_red bajo despues de un pico, la aproximacion falló y debes volver orientarte hacia la luz antes de intentar acercarte otra vez.
        - Si quieres ir a un punto concreto del mapa, elige "go_to_coordenadas" e incluye "target_coords": [x, y].
        - Si "go_to_coordenadas" termina con exito, el robot ya esta sobre la posicion objetivo.
        - Si "annotate_red_light_position" termina con exito, esa luz queda marcada como explorada en found_red_lights. Debes continuar buscando otras luces nuevas.
        - Usa la secuencia de ultimas tareas completadas como memoria de progreso. No repitas una subtarea ya completada salvo que una observacion posterior diga explicitamente que su condicion se perdio.

        Responde UNICAMENTE con este JSON:
        {
        "thought": "Tu razonamiento logico basado en la observacion y el historial.",
        "action": "Una de las acciones permitidas.",
        "target_coords": [0.0, 0.0]
        }
        Usa "target_coords" solo cuando action sea "go_to_coordenadas"; en el resto de acciones puedes omitirlo.
        """
        try:
            from langchain_ollama import ChatOllama
            from langchain_core.messages import HumanMessage, SystemMessage

            self.HumanMessage = HumanMessage
            self.llm_system_message = SystemMessage(content=self.llm_rules)
            self.llm = ChatOllama(
                model=llm_model,
                temperature=0,
                base_url=llm_base_url,
                keep_alive="5m"
            )
            print("[LLM LOCAL]: Cliente sincronico iniciado.")
        except Exception as exc:
            self.llm = None
            self.HumanMessage = None
            self.llm_system_message = None
            print(f"[LLM LOCAL ERROR FATAL]: No se pudo inicializar el LLM: {exc}")

    def filtered_red_light_reading(self, raw_reading=None):
        """Pone a cero sectores que apuntan a luces rojas ya registradas."""
        raw_reading = self.get_sensor_reading("red_light_sensor") if raw_reading is None else raw_reading
        filtered = np.array(raw_reading, dtype=float).copy()
        ignored_lights = []
        if not self.found_red_lights or filtered.size == 0:
            return filtered, ignored_lights

        robot_pos = np.array(self.controller_owner.position[:2], dtype=float)
        robot_heading = self.controller_owner.orientation[-1]
        sensor = self.controller_owner.sensors["light_sensor"]
        sensor_angles = np.array(sensor.directions(robot_heading), dtype=float)

        for light_key, light_info in self.found_red_lights.items():
            light_pos = None
            light_id = light_info.get("light_id")
            if light_id is not None:
                try:
                    light_pos = self.controller_owner.physics_client.get_body_position(int(light_id), 0)
                except (TypeError, ValueError, KeyError):
                    light_pos = None
            if light_pos is None:
                light_pos = light_info.get("light_position")
            if light_pos is None:
                continue

            light_vector = np.array(light_pos[:2], dtype=float) - robot_pos
            distance = float(np.linalg.norm(light_vector))
            if distance < 1e-9:
                continue

            light_angle = float(np.arctan2(light_vector[1], light_vector[0]))
            angle_diffs = np.abs(np.array([angle_difference(light_angle, angle) for angle in sensor_angles]))
            sectors = np.where(angle_diffs <= self.known_light_angle_tolerance)[0].astype(int).tolist()
            if not sectors:
                sectors = [int(np.argmin(angle_diffs))]

            for sector in sectors:
                filtered[sector] = 0.0

            ignored_lights.append({
                "id": light_key,
                "light_position": light_info.get("light_position"),
                "current_angle_rad": round(light_angle, 3),
                "distance": round(distance, 3),
                "ignored_sectors": sectors,
            })

        return filtered, ignored_lights

    def register_found_red_light(self, annotation_data):
        """Inserta en memoria una luz anotada por la skill de anotacion."""
        if not annotation_data:
            return False

        annotation_data = dict(annotation_data)
        robot_id = str(getattr(self.controller_owner, "name", None) or id(self.controller_owner))
        annotation_data["robot_id"] = robot_id

        light_id = annotation_data.get("light_id")
        light_position = annotation_data.get("light_position")
        if light_id is not None:
            light_key = f"{robot_id}_red_light_{light_id}"
        elif light_position is None:
            light_key = f"{robot_id}_red_light_unknown_{len(self.found_red_lights)}"
        else:
            light_key = f"{robot_id}_red_light_{light_position[0]:.2f}_{light_position[1]:.2f}"

        if light_key in self.found_red_lights:
            print(f"[found_red_lights] linea duplicada ignorada: {light_key}")
            return False

        self.found_red_lights[light_key] = annotation_data
        print(f"[found_red_lights] {light_key}: {annotation_data}")
        return True

    def build_observation(self, state):
        """Construye una observacion compacta y serializable para el LLM."""
        # Resumen compacto para el LLM. Evitamos pasar objetos complejos y nos
        # quedamos con valores serializables.
        obs = {
            "t": int(self.controller_owner.t),
            "pos": [
                round(float(self.controller_owner.position[0]), 3),
                round(float(self.controller_owner.position[1]), 3),
            ],
            "current_task": self.current_task,
            "last_observation": self.pending_observation,
        }
        raw_red_light_reading = self.get_sensor_reading("red_light_sensor")
        unexplored_red_light_reading, ignored_lights = self.filtered_red_light_reading(raw_red_light_reading)
        obs["found_red_lights"] = self.found_red_lights
        obs["red_light_sensor_unexplored"] = np.round(unexplored_red_light_reading.astype(float), 3).tolist()
        obs["max_unexplored_red_light_sensor"] = round(float(np.max(unexplored_red_light_reading)), 3) # valor clave que el LLM debe usar para decidir si hay una luz roja nueva que explorar.
        obs["red_light_sensor"] = obs["red_light_sensor_unexplored"]
        obs["max_red_light_sensor"] = obs["max_unexplored_red_light_sensor"]
        for sensor_name in [
            "distance_sensor",
            "blue_battery_sensor",
        ]:
            if sensor_name in state:
                value = state[sensor_name]
                if isinstance(value, np.ndarray):
                    obs[sensor_name] = np.round(value.astype(float), 3).tolist()
                    obs[f"max_{sensor_name}"] = round(float(np.max(value)), 3)
                else:
                    obs[sensor_name] = value
        world = getattr(self.controller_owner, "world", None)
        if world is not None:
            for obj in world.hierarchy.values():
                if getattr(obj, "color", None) == "blue" and hasattr(obj, "position"):
                    obs["blue_light_position"] = [
                        round(float(obj.position[0]), 3),
                        round(float(obj.position[1]), 3),
                    ]
                    break
        return obs

    def request_llm_decision(self, state):
        """Consulta al LLM la siguiente skill a ejecutar usando estado y memoria."""
        if self.llm is None:
            print("[LLM LOCAL ERROR]: LLM no inicializado. Usando stop.")
            self.apply_llm_action(
                "LLM no inicializado; se mantiene detenido.",
                "stop",
                int(self.controller_owner.t),
            )
            return

        t = int(self.controller_owner.t)
        memoria_historial = "\n".join(self.memory_history[-10:]) or "Sin acciones previas."
        ultimas_tareas = self.completed_task_summary()
        observacion = self.build_observation(state)
        contexto = f"""
        ESTADO ACTUAL:
        - Macro-tarea asignada: {self.macro_task}
        - Observacion actual de los sensores: {observacion}
        - Secuencia de las ultimas 10 subtareas completadas:
        {ultimas_tareas}
        - Diario de memoria con thought/action/observation:
        {memoria_historial}
        """

        try:
            response = self.llm.invoke([
                self.llm_system_message,
                self.HumanMessage(content=contexto),
            ])
            raw_content = response.content.strip()
            if "```json" in raw_content:
                raw_content = raw_content.split("```json")[1].split("```")[0].strip()

            data = json.loads(raw_content)
            thought = data.get("thought", "")
            action = data.get("action", "stop")
            target_coords = data.get("target_coords")
        except Exception as exc:
            print(f"[LLM LOCAL ERROR]: {exc}")
            thought = f"Error consultando el LLM: {exc}"
            action = "stop"
            target_coords = None

        self.apply_llm_action(thought, action, t, target_coords)

    def completed_task_summary(self):
        """Devuelve un resumen textual de las ultimas subtareas terminadas."""
        recent_tasks = self.completed_task_sequence[-10:]
        if not recent_tasks:
            return "Sin subtareas completadas todavia."
        return "\n".join(
            f"{idx}. step={item['step']} | task={item['task']} | result={item['result']}"
            for idx, item in enumerate(recent_tasks, start=1)
        )

    def apply_llm_action(self, thought, action, obs_step, target_coords=None):
        """Aplica la accion elegida por el LLM y activa la skill correspondiente."""
        if action not in self.allowed_actions:
            print(f"[LLM LOCAL] Accion no permitida '{action}'. Usando stop.")
            action = "stop"

        previous_task = self.secondary_task
        self.llm_thought = thought
        self.llm_action = action
        self.secondary_task = action
        self.secondary_controller = self.routines[action]
        if previous_task != action and hasattr(self.secondary_controller, "reset"):
            self.secondary_controller.reset()
        elif hasattr(self.secondary_controller, "done"):
            self.secondary_controller.done = False
        if action == "go_to_coordenadas":
            if target_coords is None:
                self.secondary_controller.target_coords = None
            else:
                try:
                    parsed_target = np.array(target_coords[:2], dtype=float)
                    if parsed_target.shape != (2,):
                        raise ValueError
                    self.secondary_controller.target_coords = parsed_target
                except (TypeError, ValueError):
                    print(f"[LLM LOCAL] Coordenadas invalidas para go_to_coordenadas: {target_coords}")
                    self.secondary_controller.target_coords = None
        self.waiting_for_llm = False

        action_step = int(self.controller_owner.t)
        elapsed_steps = action_step - int(obs_step)
        print(f"\n[LLM LOCAL | obs_step {obs_step} | action_step {action_step} | delay {elapsed_steps}] thought: {thought}")
        print(f"[LLM LOCAL | action_step {action_step}] action: {action}\n")

    def notify_task_done(self, skill_name):
        """Registra el resultado de una skill terminada y prepara otra decision."""
        # Cuando una skill declara done=True, se registra observation, se para el
        # robot y se deja preparado el siguiente ciclo de decision.
        done_reason = getattr(self.secondary_controller, "done_reason", "")
        done_success = getattr(self.secondary_controller, "done_success", True)
        if skill_name == "annotate_red_light_position" and done_success:
            self.register_found_red_light(getattr(self.secondary_controller, "annotation_data", None))
        if done_success:
            observation = f"Subtarea '{skill_name}' completada con exito."
        else:
            observation = f"Subtarea '{skill_name}' detenida sin exito."
        if done_reason:
            observation = f"{observation} Motivo: {done_reason}"
        self.pending_observation = observation
        self.completed_task_sequence.append({
            "step": int(self.controller_owner.t),
            "task": skill_name,
            "result": observation,
        })
        self.memory_history.append(
            f"thought={self.llm_thought} | action={skill_name} | observation={observation}"
        )
        print(f"[LLM LOCAL] observation: {observation}")
        self.secondary_task = "stop"
        self.secondary_controller = self.routines["stop"]
        self.waiting_for_llm = True

    def step(self, state, reward=0.0):
        """Ejecuta un ciclo de control con o sin LLM segun la configuracion."""
        # 1. Ejecutar siempre la capa inconsciente de seguridad.
        self.survival_controller.step(state)
        self.activations["survival"] = np.array(
            self.get_actuator("joint_velocity_actuator").action
        )

        if self.use_llm:
            return self.step_with_llm(state)

        # Modo manual: ejecuta la tarea secundaria fijada en el JSON.
        self.secondary_controller.step(state)
        self.activations["secondary"] = np.array(
            self.get_actuator("joint_velocity_actuator").action
        )

        action = self.coordinate()
        self.log_robot_state()
        return action

    def step_with_llm(self, state):
        """Ejecuta la capa secundaria en modo LLM y detecta fin de skill."""
        # El robot solo consulta al LLM cuando necesita una nueva mision:
        # al inicio o justo despues de que una skill declare done=True.
        # La llamada es sincronica; no hay decisiones retrasadas con sensores viejos.
        if self.waiting_for_llm:
            self.request_llm_decision(state)

        self.secondary_controller.step(state)
        self.activations["secondary"] = np.array(
            self.get_actuator("joint_velocity_actuator").action
        )

        #Es decir: después de ejecutar la skill secundaria, mira si esa skill tiene done=True. Si lo tiene, llama a notify_task_done(...).
        active_skill = self.secondary_task
        if getattr(self.secondary_controller, "done", False) and active_skill != "stop":
            self.notify_task_done(active_skill)
            self.activations["secondary"] = np.zeros(2)

        action = self.coordinate()
        self.log_robot_state()
        return action

    def log_robot_state(self):
        """Guarda cada diez pasos la posicion y la tarea activa en el CSV."""
        # Guardamos solo lo minimo para que main.py pueda pintar la trayectoria.
        # La columna tarea permite ver cuando manda avoid obstacle y cuando manda
        # la skill secundaria.
        t = self.controller_owner.t
        x, y = self.controller_owner.position[0], self.controller_owner.position[1]

        if t % 10 == 0:
            with open(self.log_name, "a") as f:
                f.write(f"{t},{x:.3f},{y:.3f},{self.current_task}\n")

    def coordinate(self):
        """Elige entre supervivencia y skill secundaria segun la prioridad."""
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
        """Reconecta owners reales y reinicia todas las skills hijas."""
        # Las rutinas hijas se crean antes de que el robot exista por completo.
        # En reset les conectamos el owner real para que puedan leer sensores,
        # actuadores, posicion y tiempo de simulacion.
        self.found_red_lights = {}
        self.survival_controller.controller_owner = self.controller_owner
        for skill_controller in self.routines.values():
            skill_controller.controller_owner = self.controller_owner
            skill_controller.reset()
        self.survival_controller.reset()
