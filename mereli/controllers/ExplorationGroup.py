import os
import json
import time
from multiprocessing import Process, Manager

import numpy as np

from mereli.controllers import RobotController
from mereli.register import controller_registry, controllers


def exploration_llm_loop(shared_data, system_rules_content, base_url, model_name):
    """Proceso local del LLM individual.

    Sigue el patron de astorekeeperLLM2: el controller principal escribe el
    contexto en una pizarra compartida y este proceso responde con un JSON
    que contiene thought + action.
    """
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage, SystemMessage

        # El LLM vive en un proceso aparte para no bloquear cada step fisico del
        # simulador mientras razona.
        llm = ChatOllama(
            model=model_name,
            temperature=0,
            base_url=base_url,
            keep_alive="5m"
        )
        sys_msg = SystemMessage(content=system_rules_content)
        print("[LLM LOCAL]: Proceso iniciado.")
    except Exception as exc:
        print(f"[LLM LOCAL ERROR FATAL]: No se pudo inicializar el LLM: {exc}")
        return

    last_processed_sensor_ts = 0
    while True:
        # sensor_ts funciona como reloj/logical timestamp: si cambia, hay una
        # nueva observacion que el LLM debe procesar.
        current_sensor_ts = shared_data.get("sensor_ts", 0)
        processing = shared_data.get("processing", False)

        if not processing and current_sensor_ts > last_processed_sensor_ts:
            shared_data["processing"] = True
            contexto = shared_data.get("contexto", "")
            try:
                response = llm.invoke([sys_msg, HumanMessage(content=contexto)])
                raw_content = response.content.strip()

                # Algunos modelos devuelven JSON dentro de un bloque markdown.
                # Lo limpiamos para poder llamar a json.loads().
                if "```json" in raw_content:
                    raw_content = raw_content.split("```json")[1].split("```")[0].strip()

                data = json.loads(raw_content)
                thought = data.get("thought", "")
                action = data.get("action", "stop")

                shared_data["thought"] = thought
                shared_data["action"] = action
                shared_data["action_ts"] = current_sensor_ts
                last_processed_sensor_ts = current_sensor_ts
                print(f"[LLM LOCAL]: decision registrada en t={shared_data.get('last_t')}")
            except Exception as exc:
                print(f"[LLM LOCAL ERROR]: {exc}")
            finally:
                shared_data["processing"] = False

        # Evita que el proceso hijo consuma CPU girando en vacio.
        time.sleep(0.05)


@controller_registry(name="stop")
class StopController(RobotController):
    """Skill basica: detener el robot mientras espera una orden."""

    def __init__(self, *args, **kwargs):
        super(StopController, self).__init__(*args, **kwargs)
        self.flag = True

    def step(self, state, reward=0.0):
        self.get_actuator("joint_velocity_actuator").action = np.array([0.0, 0.0])


@controller_registry(name="navigate")
class NavigateController(RobotController):
    """Skill basica: explorar hasta recibir cualquier senal de luz roja."""

    def __init__(self, *args, **kwargs):
        super(NavigateController, self).__init__(*args, **kwargs)
        self.flag = True
        self.done = False

    def step(self, state, reward=0.0):
        ls_read = self.get_sensor_reading("red_light_sensor")
        red_light_detected = np.max(ls_read) > 0.0

        if red_light_detected:
            action = np.array([0.0, 0.0])
            if not self.done:
                print(f"[navigate] Luz roja detectada en step {self.controller_owner.t}")
            self.done = True
        else:
            action = np.ones(2)
            self.done = False

        self.get_actuator("joint_velocity_actuator").action = action


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
        super(OrientRedLightController, self).__init__(*args, **kwargs)
        self.angular_speed = angular_speed
        self.fine_angular_speed = fine_angular_speed
        self.front_balance_tolerance = front_balance_tolerance
        self.flag = True
        self.centered = False
        self.done = False

    def step(self, state, reward=0.0):
        ls_read = self.get_sensor_reading("red_light_sensor")
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

    def __init__(self, *args, forward_speed=0.5, near_threshold=0.9, **kwargs):
        super(ApproachRedLightController, self).__init__(*args, **kwargs)
        self.forward_speed = forward_speed
        self.near_threshold = near_threshold
        self.flag = True
        self.reset()

    def reset(self):
        self.light_found = False
        self.done = False
        self.best_red = 0.0
        self.done_reason = ""

    def step(self, state, reward=0.0):
        ls_read = self.get_sensor_reading("red_light_sensor")
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
            self.done_reason = f"max_red={max_red:.3f} alcanzo threshold={self.near_threshold:.3f}."
        elif self.best_red > 0.5 and max_red < self.best_red - 0.03:
            action = np.array([0.0, 0.0])
            print(
                f"[approach_red_light] Maximo de luz pasado: "
                f"best_red={self.best_red:.3f}, max_red={max_red:.3f}. Deteniendo."
            )
            self.light_found = True
            self.done = True
            self.done_reason = (
                f"max_red subio hasta {self.best_red:.3f} y despues bajo a {max_red:.3f}; "
                "probablemente paso cerca de la luz sin alcanzar el umbral."
            )
        else: # Si no estamos cerca, avanzamos recto para acercarnos a la luz.
            action = self.forward_speed * np.array([1.0, 1.0])
            self.light_found = False
            self.done = False
            self.done_reason = ""

        self.get_actuator("joint_velocity_actuator").action = action


@controller_registry(name="load_blue_battery")
class LoadBlueBatteryController(RobotController):
    """Skill basica: esperar quieto hasta cargar la bateria azul."""

    def __init__(self, *args, charge_threshold=0.97, **kwargs):
        super(LoadBlueBatteryController, self).__init__(*args, **kwargs)
        self.charge_threshold = charge_threshold
        self.flag = True
        self.charged = False
        self.done = False

    def step(self, state, reward=0.0):
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


@controller_registry(name="annotate_red_light_position")
class AnnotateRedLightPositionController(RobotController):
    """Skill basica: anotar desde donde se percibe una luz roja cercana."""

    def __init__(self, *args, detection_threshold=0.75, **kwargs):
        super(AnnotateRedLightPositionController, self).__init__(*args, **kwargs)
        self.detection_threshold = detection_threshold
        self.flag = True
        self.light_position_printed = False
        self.annotation_printed = False
        self.done = False

    def step(self, state, reward=0.0):
        self.get_actuator("joint_velocity_actuator").action = np.array([0.0, 0.0])
        self.print_red_light_position_once()

        ls_read = self.get_sensor_reading("red_light_sensor")
        intensity = float(np.max(ls_read))
        if intensity < self.detection_threshold or self.annotation_printed:
            return

        sector = int(np.argmax(ls_read))
        robot_pos = self.controller_owner.position
        sensor = self.controller_owner.sensors["light_sensor"]
        global_angle = sensor.directions(self.controller_owner.orientation[-1])[sector]
        global_angle_deg = np.degrees(global_angle)

        print(
            "[annotate_red_light_position] "
            f"Posicion vision=({robot_pos[0]:.3f}, {robot_pos[1]:.3f}), "
            f"intensidad={intensity:.3f}, "
            f"sector={sector}, "
            f"angulo_global={global_angle:.3f} rad ({global_angle_deg:.1f} deg)"
        )
        self.annotation_printed = True
        self.done = True

    def print_red_light_position_once(self):
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
        macro_task="Explorar el mapa y encontrar luces rojas.",
        llm_model="gpt-oss:20b",
        llm_base_url="http://127.0.0.1:11434",
        decision_interval=10,
        **kwargs
    ):
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
        self.decision_interval = decision_interval #cada cuantos steps se le pide al LLM una nueva decision, para que pueda reaccionar a cambios en sensores durante skills largas como navigate.
        self.allowed_actions = [
            "stop",
            "navigate",
            "orient_red_light",
            "approach_red_light",
            "load_blue_battery",
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
        # Pizarra compartida con el proceso del LLM. El controller escribe
        # observaciones/contexto; el LLM escribe thought/action.
        self.manager = Manager()
        self.shared_data = self.manager.dict()
        self.shared_data["thought"] = "Inicializando controlador tactico."
        self.shared_data["action"] = "stop"
        self.shared_data["contexto"] = ""
        self.shared_data["last_t"] = 0
        self.shared_data["sensor_ts"] = 0
        self.shared_data["action_ts"] = 0
        self.shared_data["processing"] = False

        self.llm_thought = self.shared_data["thought"]
        self.llm_action = "stop"
        self.last_action_ts = 0
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
        - "navigate": Avanza/explora el mapa en busqueda de una luz roja. Termina cuando red_light_sensor recibe cualquier valor mayor que 0.
        - "orient_red_light": Gira para centrar una luz roja detectada.
        - "approach_red_light": Avanza recto hacia una luz roja ya centrada.
        - "load_blue_battery": Se queda quieto esperando hasta que la bateria azul llegue al umbral.
        - "annotate_red_light_position": Anota la posicion desde la que ve una luz roja cercana para registrarla y darla por encontrada.

        POLITICA DE DECISION:
        - Si en la memoria aparece que "orient_red_light" termino con exito, significa que la luz roja esta completamente centrada. En ese caso NO vuelvas a usar "orient_red_light" en la siguiente accion.
        - Si en la memoria aparece que "approach_red_light" termino con exito, significa que el robot ya esta cerca de la luz. 
        - Usa la secuencia de ultimas tareas completadas como memoria de progreso. No repitas una subtarea ya completada salvo que una observacion posterior diga explicitamente que su condicion se perdio.

        Responde UNICAMENTE con este JSON:
        {
        "thought": "Tu razonamiento logico basado en la observacion y el historial.",
        "action": "Una de las acciones permitidas."
        }
        """

        # Arrancamos el LLM local. La integracion completa aun requiere que
        # step() escriba contexto y lea la action devuelta.
        self.brain_process = Process(
            target=exploration_llm_loop,
            args=(self.shared_data, self.llm_rules, llm_base_url, llm_model),
            daemon=True
        )
        self.brain_process.start()

    def build_observation(self, state):
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
        red_light_reading = self.get_sensor_reading("red_light_sensor")
        obs["red_light_sensor"] = np.round(red_light_reading.astype(float), 3).tolist()
        obs["max_red_light_sensor"] = round(float(np.max(red_light_reading)), 3)
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
        return obs

    """ sirve para pedirle una nueva decisión al LLM.
    No llama directamente al modelo. Lo que hace es escribir en la “pizarra compartida”
    (self.shared_data) el contexto actual del robot. Luego el proceso separado del LLM
    (exploration_llm_loop) detecta que hay una observación nueva y responde con una acción"""

    def request_llm_decision(self, state):
        # Escribe el estado actual en la pizarra. El proceso del LLM detecta el
        # cambio en sensor_ts y responde con thought + action.
        if self.shared_data.get("processing", False):
            return

        t = self.controller_owner.t
        memoria_historial = "\n".join(self.memory_history[-10:]) or "Sin acciones previas."
        ultimas_tareas = self.completed_task_summary()
        observacion = self.build_observation(state)
        self.shared_data["contexto"] = f"""
        ESTADO ACTUAL:
        - Macro-tarea asignada: {self.macro_task}
        - Observacion actual de los sensores: {observacion}
        - Secuencia de las ultimas 10 subtareas completadas:
        {ultimas_tareas}
        - Diario de memoria con thought/action/observation:
        {memoria_historial}
        """
        self.shared_data["last_t"] = t
        self.shared_data["sensor_ts"] = t

    def completed_task_summary(self):
        recent_tasks = self.completed_task_sequence[-10:]
        if not recent_tasks:
            return "Sin subtareas completadas todavia."
        return "\n".join(
            f"{idx}. step={item['step']} | task={item['task']} | result={item['result']}"
            for idx, item in enumerate(recent_tasks, start=1)
        )

    def read_llm_action(self):
        # Lee una decision nueva si el proceso del LLM ya ha contestado.
        action_ts = self.shared_data.get("action_ts", 0)
        if action_ts == self.last_action_ts:
            return

        self.last_action_ts = action_ts
        thought = self.shared_data.get("thought", "")
        action = self.shared_data.get("action", "stop")
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
        self.waiting_for_llm = False

        print(f"\n[LLM LOCAL | step {action_ts}] thought: {thought}")
        print(f"[LLM LOCAL] action: {action}\n")

    def notify_task_done(self, skill_name):
        # Cuando una skill declara done=True, se registra observation, se para el
        # robot y se deja preparado el siguiente ciclo de decision.
        observation = f"Subtarea '{skill_name}' completada con exito."
        done_reason = getattr(self.secondary_controller, "done_reason", "")
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
        # Si estamos esperando una orden, pedimos contexto al LLM. Tambien se
        # vuelve a pedir periodicamente, para que pueda reaccionar a sensores
        # nuevos durante skills largas como navigate.
        t = self.controller_owner.t
        if self.waiting_for_llm or t % self.decision_interval == 0:
            self.request_llm_decision(state)
        self.read_llm_action()

        # Mientras el LLM piensa, el robot queda quieto salvo que obstacle avoid
        # tome control por seguridad.
        if self.waiting_for_llm:
            self.secondary_task = "stop"
            self.secondary_controller = self.routines["stop"]

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
        for skill_controller in self.routines.values():
            skill_controller.controller_owner = self.controller_owner
            skill_controller.reset()
        self.survival_controller.reset()
