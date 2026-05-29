import numpy as np
from mereli.controllers import RobotController
from mereli.register import controller_registry, controllers
from mereli.utils import compute_angle

import datetime
import os
import csv
import json 
import sys
from multiprocessing import Process, Manager

#LLM API imports
import json
import time
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


@controller_registry(name="navigate")
class NavigateController(RobotController):
    def __init__(self, *args,  **kwargs):
        super(NavigateController, self).__init__(*args, **kwargs)
        self.flag = True 

    def step(self, state, reward=0):
        self.get_actuator('joint_velocity_actuator').action = np.ones(2) 


def follow_light_search_strict(controller, sensor_name, turn_speed=0.2, forward_speed=0.5, stop_on_strong=True):
    ls_read = controller.get_sensor_reading(sensor_name)
    max_light = np.max(ls_read)

    if stop_on_strong and max_light > 0.9:
        return np.zeros(2)

    if max_light <= 0.02:
        return np.array([forward_speed, forward_speed])

    light_left = np.sum(ls_read[[7, 6, 5, 4]])
    light_right = np.sum(ls_read[[0, 1, 2, 3]])

    if ls_read[0] > 0 and ls_read[7] > 0:
        return np.array([forward_speed, forward_speed])
    if light_right > light_left:
        return turn_speed * np.array([-1, 1])
    return turn_speed * np.array([1, -1])


@controller_registry(name="load_blue_battery")
class LoadBlueBatteryController(RobotController):
    def __init__(self, *args,  wait_full_load=True, **kwargs):
        super(LoadBlueBatteryController, self).__init__(*args, **kwargs)
        self.flag = False

    def step(self, state, reward=0, force_mission=False):
        action = np.array([0.0, 0.0])
        if self.flag:
            action = follow_light_search_strict(self, 'blue_light_sensor')
        self.get_actuator('joint_velocity_actuator').action = action

@controller_registry(name="load_red_battery")
class LoadRedBatteryController(RobotController):
    def __init__(self, *args,  wait_full_load=True, **kwargs):
        super(LoadRedBatteryController, self).__init__(*args, **kwargs)
        self.flag = False

    def step(self, state, reward=0, force_mission=False):
        action = np.array([0.0, 0.0])
        if self.flag:
            action = follow_light_search_strict(self, 'red_light_sensor')
        self.get_actuator('joint_velocity_actuator').action = action

def llm_brain_loop(shared_data, system_rules_content, base_url):
    """
    PROCESO INDEPENDIENTE: El Cerebro.
    Este bucle corre en un núcleo de CPU distinto al del robot.
    """
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage, SystemMessage
    import json
    import time

    print(f"🧠 [CEREBRO]: Iniciando proceso hijo. Conectando...")

    try:
        # Inicialización del modelo dentro del proceso hijo
        llm = ChatOllama(
            model="gpt-oss:20b",
            temperature=0,
            base_url= base_url,
            keep_alive="5m"
        )
    
        sys_msg = SystemMessage(content=system_rules_content)

        print("🧠 [CEREBRO]: Proceso de IA iniciado y listo.")
    except Exception as e:
        print(f"❌ [CEREBRO ERROR FATAL]: No se pudo inicializar ChatOllama: {e}")
        return

    last_processed_sensor_ts = 0
    while True:
        current_sensor_ts = shared_data.get('sensor_ts', 0)
        processing = shared_data.get('processing', False)

        if not processing and current_sensor_ts > last_processed_sensor_ts:
            shared_data['processing'] = True
            contexto = shared_data.get('contexto', '')
            request_wall_time = time.perf_counter()
            try:
                response = llm.invoke([sys_msg, HumanMessage(content=contexto)])
                raw_content = response.content.strip()

                # Limpieza de JSON
                if "```json" in raw_content:
                    raw_content = raw_content.split("```json")[1].split("```")[0].strip()
                
                data = json.loads(raw_content)

                # 3. Escribir la decisión en la pizarra para que el robot la lea
                decision = data.get('decision', 'simple_forage')
                memoria = data.get('memoria_interna', '')
                razonamiento = data.get('razonamiento', '')

                response_wall_time = time.perf_counter()
                shared_data['decision'] = decision
                shared_data['memoria_interna'] = memoria
                shared_data['razonamiento'] = razonamiento
                shared_data['decision_ts'] = current_sensor_ts
                shared_data['llm_request_wall_time'] = request_wall_time
                shared_data['llm_response_wall_time'] = response_wall_time
                shared_data['llm_latency_s'] = response_wall_time - request_wall_time
                last_processed_sensor_ts = current_sensor_ts

                print(f"🧠 [CEREBRO]: t={shared_data.get('last_t')} | Decisión registrada\n")
            except Exception as e:
                print(f"❌ [CEREBRO ERROR]: {e}")
            finally:
                shared_data['processing'] = False

        # Evitar consumo excesivo de CPU en el bucle de espera
        time.sleep(0.05)

@controller_registry(name='astorekeeperLLM2')
class AStoreKeeperLLM2Controller(RobotController):
    """
    """
    def __init__(self, *args, routines={'navigate' : 0}, **kwargs):
        super(AStoreKeeperLLM2Controller, self).__init__(*args, **kwargs)
        self.routines = {}
        self.priorities = {}
        self.activations = {}

        # --- CONFIGURACIÓN DE MULTIPROCESSING (BLACKBOARD) ---
        self.manager = Manager()
        self.shared_data = self.manager.dict()
        
        # Estado inicial de la pizarra compartido con el proceso IA
        self.shared_data['decision'] = 'simple_forage'
        self.shared_data['memoria_interna'] = 'Inicio de misión.'
        self.shared_data['razonamiento'] = 'Inicializando...'
        self.shared_data['contexto'] = ''
        self.shared_data['last_t'] = 0
        self.shared_data['sensor_ts'] = 0
        self.shared_data['decision_ts'] = 0
        self.shared_data['processing'] = False
        self.last_decision_ts = 0

        self.rules = """
        Eres el cerebro de un robot e-puck.
        Tu objetivo es elegir la rutina técnica correcta basada en los sensores y el historial.
        Sabiendo que el robot tiene dos baterías (azul y roja), con valores de 0.0 a 1.0, y puede realizar cuatro rutinas:
        "simple_forage", en la encuentra objetos y los deposita en una zona especifica,
        "turn_yellow_lights_OFF", que apaga las tres luces amarillas, y
        "load_blue_battery" y "load_red_battery",  que recargan las respectivas baterías.

        ESTRUCTURA DE RESPUESTA (JSON):
        {
        "razonamiento": "Tu análisis de los sensores y por qué eliges la acción.",
        "memoria_interna": "Tu diario. Aquí guarda contadores y decisiones (ej: 'He ordenado forage 2 veces. Todavía no he completado la misión de apagar luces').",
        "decision": "Nombre de la rutina [simple_forage, load_red_battery, load_blue_battery, turn_yellow_lights_OFF]"
        }

        REGLAS DE MEMORIA:
        1. Lee siempre tu 'Memoria anterior' para saber qué estabas haciendo.
        2. Actualiza tu 'memoria_interna' en cada respuesta.

        JERARQUÍA DE DECISIÓN (Sigue este orden):
        1. PERSISTENCIA DE CARGA: Si tu 'Rutina actual' es una de carga (load) y la batería NO ha llegado a 0.7, DEBES seguir respondiendo esa misma rutina de carga.
        2. EMERGENCIA: Si una batería baja de 0.3, manda cargarla. PRIORIZA carga (Red > Blue).
        3. MISIÓN LUCES: Cuando en tu 'memoria_interna' anotes que has hecho forage 3 veces, cambia a 'turn_yellow_lights_OFF'. PERO: Si 'luces_amarillas_APAGADAS_actualmente' ya es 3, escribe en memoria "Misión completada" y cambia a 'simple_forage'.
        4. FORAGE: En cualquier otro caso, manda 'simple_forage'

        ### EJEMPLO DE COMPORTAMIENTO (One-shot):
        Usuario: "Sensores: {'bat_azul': 0.80, 'bat_roja': 0.25, 'luces_amarillas_APAGADAS_actualmente': 3}.
        Respuesta: {
                    "razonamiento": "La batería roja está al 0.25, lo cual es crítico.",
                    "memoria_interna": "He ordenado hacer forage 4 veces. Ya completé la mision de apagar luces amarillas. Interrumpo para cargar roja.",
                    "decision": "load_red_battery"
                    }
        """

        # Lanzar el proceso del cerebro
        self.brain_process = Process(
            target=llm_brain_loop, 
            args=(self.shared_data, self.rules, "http://127.0.0.1:11434"),
            daemon=True
        )
        self.brain_process.start()

        local_routines = {
            "navigate": NavigateController,
            "load_blue_battery": LoadBlueBatteryController,
            "load_red_battery": LoadRedBatteryController,
        }
        for rt, pr in routines.items(): 
            priority = pr if isinstance(pr, int) else pr['priority']
            rt_params = pr.get('params', {}) if isinstance(pr, dict) else {}
            self.priorities[rt] = priority
            routine_cls = local_routines.get(rt, controllers[rt])
            self.routines[rt] = routine_cls(**rt_params)
            self.activations[rt] = np.zeros(2)

        #Cambiar tarea con cambio de orden
        self.external_routine = "simple_forage" # la orden del JSON

        # --- CARPETAS POR FECHA ---
        self.output_dir = os.environ.get("CURRENT_EXP_FOLDER", "outputs")
        self.log_name = os.path.join(self.output_dir, "recorrido_robot.csv")
        self.metrics_log_name = os.path.join(self.output_dir, "llm_tiempos.csv")

        if not os.path.exists(self.log_name):
            with open(self.log_name, "w") as f:
                f.write("step,x,y,bat_azul,bat_roja,num_luces,decision,decision_ts,tarea\n")

        print(f"📁 Guardando experimento en: {self.output_dir}")

    def log_llm_timing(self, request_step, apply_step, latency_s, decision):
        blind_steps = int(apply_step) - int(request_step)
        async_ratio = blind_steps / latency_s if latency_s > 0 else 0.0
        robot_name = getattr(self.controller_owner, "name", "robot_0")
        file_exists = os.path.exists(self.metrics_log_name)
        with open(self.metrics_log_name, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists or os.path.getsize(self.metrics_log_name) == 0:
                writer.writerow([
                    "scope", "robot", "request_step", "apply_step",
                    "blind_steps", "latency_s", "async_ratio", "decision"
                ])
            writer.writerow([
                "local", robot_name, int(request_step), int(apply_step),
                blind_steps, f"{latency_s:.6f}", f"{async_ratio:.6f}", decision
            ])


    def step(self, state, reward=0.0):
        
        # Obtener datos actuales
        t = self.controller_owner.t 
        # La posición es un array [x, y, z]
        x, y = self.controller_owner.position[0], self.controller_owner.position[1]
        # El sensor de batería azul devuelve un array, cogemos el primer valor
        val_azul = state['blue_battery_sensor'][0] 
        v_roja = state['red_battery_sensor'][0]
        #Contador de luces apagadas para la rutina turn_yellow_lights_OFF
        n_luces = self.routines['turn_yellow_lights_OFF'].lights_off

        # 1. ACTUALIZAR SENSORES EN LA PIZARRA CONSTANTEMENTE
        robot_state = {
            "t": t,
            "bat_azul": round(float(val_azul), 2),
            "bat_roja": round(float(v_roja), 2),
            "luces_apagadas": n_luces
        }
        self.shared_data['contexto'] = f"SENSORS: {robot_state}. MEM: {self.shared_data['memoria_interna']}"
        self.shared_data['last_t'] = t
        self.shared_data['sensor_ts'] = t

        # 2. LEER LA ÚLTIMA DECISIÓN (Comportamiento por defecto)
        nueva_orden = self.shared_data['decision']
        razon = self.shared_data.get('razonamiento', '')
        memoria = self.shared_data.get('memoria_interna', '')
        decision_ts = self.shared_data.get('decision_ts', 0)

        self.external_memory = memoria
        self.llm_reasoning = razon
        self.llm_decision = nueva_orden

        if decision_ts != self.last_decision_ts:
            self.last_decision_ts = decision_ts
            print(f"\n🧠 [PENSAMIENTO | step {decision_ts}]: {self.llm_reasoning}")
            print(f"📖 [MEMORIA]: {self.external_memory}")
            print(f"🎯 [ACCIÓN]: {self.llm_decision}\n")
            latency_s = float(self.shared_data.get('llm_latency_s', 0.0))
            self.log_llm_timing(decision_ts, t, latency_s, self.llm_decision)
            if nueva_orden in self.routines:
                self.external_routine = nueva_orden

        # 3. Ejeutar rutinas
        for k, routine in self.routines.items():
            routine.flag = (self.external_routine == k)
            routine.step(state)
            self.activations[k] = self.get_actuator('joint_velocity_actuator').action 

        # 4. Lógica de continuidad (Evitar el cierre de simulación)
        rt_obj = self.routines.get(self.external_routine)
        if rt_obj and not rt_obj.flag:
            if self.external_routine != "simple_forage":
                print(f"✅ Tarea {self.external_routine} completada. Volviendo a simple_forage...")
                self.external_routine = "simple_forage"
                self.routines["simple_forage"].flag = True

        action = self.coordinate()

        # 5. Guardar en el CSV cada 10 pasos. decision es la orden LLM;
        # tarea es la capa que finalmente controla las ruedas.
        if t % 10 == 0:
            with open(self.log_name, "a") as f:
                f.write(f"{t},{x:.3f},{y:.3f},{val_azul:.3f},{v_roja:.3f},{n_luces},{self.llm_decision},{decision_ts},{self.current_routine}\n")

        return action
    
    def coordinate(self):

        """Combina la orden del LLM con una capa de seguridad por subsuncion."""

        obstacle_avoider = self.routines.get("basic_obstacle_avoider")
        if obstacle_avoider is not None and obstacle_avoider.flag:
            self.current_routine = "basic_obstacle_avoider"
            action_wheels = self.activations["basic_obstacle_avoider"]
            self.get_actuator('joint_velocity_actuator').action = np.array(action_wheels)
            return {'joint_velocity_actuator' : self.get_actuator('joint_velocity_actuator').action}

        # Prioridad absoluta a la orden externa
        if self.external_routine and self.external_routine in self.routines:
            self.current_routine = self.external_routine
            action_wheels = self.activations[self.external_routine]
        else:
            # Comportamiento secuencial si no hay orden
            names = [*self.priorities.keys()]
            names.sort(key=self.priorities.get)
            self.current_routine = "none"
            action_wheels = np.zeros(2)
            for k in names:   
                if self.routines[k].flag:
                    self.current_routine = k
                    action_wheels = self.activations[k]
                    break
        
        self.get_actuator('joint_velocity_actuator').action = np.array(action_wheels)
        return {'joint_velocity_actuator' : self.get_actuator('joint_velocity_actuator').action}

    def reset(self):
        for rt in self.routines.values():
            rt.controller_owner = self.controller_owner
            rt.reset()

@controller_registry(name="simple_forage")
class SimpleForageController(RobotController):
    def __init__(self, *args,  wait_full_load=True, **kwargs):
        super(SimpleForageController, self).__init__(*args, **kwargs)
        self.flag = False
        #incluimos logica para evitar obstaculos
        self.sensitivity = 0.4
        self.carrying = False
        

    def step(self, state, reward=0):
        # Obtener el tiempo actual de la simulación
        t = self.controller_owner.t
        #Sensores
        st_ds = self.get_sensor_reading('distance_sensor') # Proximidad
        gs_read = self.get_sensor_reading('ground_sensor')
        ls_read = self.get_sensor_reading('red_light_sensor') # Luz roja
        
        if gs_read == 1.0 and not self.carrying: #zona gris
            self.carrying = True
            print(f"📦 [STEP {t}] ¡OBJETO RECOGIDO! Buscando zona de depósito...")

        if gs_read == 0.0 and self.carrying: #zona negra
            self.carrying = False
            print(f"🗑️ [STEP {t}] ¡OBJETO DEPOSITADO! (Zona negra pisada). Volviendo a patrullar.")
            # Pequeña maniobra de escape para alejarse de la luz

        if np.max(st_ds) > self.sensitivity:  # Obstáculo detectado, lógica de evasión
            if any(st_ds[[0,1]] > self.sensitivity):
                # print('Turn Left')
                action = np.array([1., -1])
            elif any(st_ds[[6,7]] > self.sensitivity):
                # print('Turn Right')
                action = np.array([-1, 1.])
            else:
            # print('GO straight over')
                action = np.array([1.,1.])

        elif self.carrying: # Garbage collected
            action = follow_light_search_strict(
                self,
                'red_light_sensor',
                turn_speed=0.1,
                forward_speed=0.7,
                stop_on_strong=False,
            )
            if ls_read[0] > 0 and ls_read[7] > 0:
                print("Luz roja detectada, pero centrada. Avanzando hacia ella.")
        else:
            action = np.array([0.7, 0.7]) #navigate
        self.get_actuator('joint_velocity_actuator').action = action


        
@controller_registry(name='subsumption_garbage')
class SubsumptionGarbageController(RobotController):
    """
    """
    def __init__(self, *args,  **kwargs):
        super(SubsumptionGarbageController, self).__init__(*args, **kwargs)
        self.routines = ['forage', 'nav', 'load', 'avoid']
        self.activations = {'forage' : np.zeros(2), 'nav' : np.array([1., 1.]), 'load' : np.zeros(2), 'avoid' : np.zeros(2)} 
        self.flags = {k : False for k in self.routines} 
        self.bat_threshold = .5

    def step(self, state, reward=0.0):
        self.avoid_obstacles(state)
        self.load_battery(state)
        self.navigate(state)
        self.forage(state)
        return self.coordinate()


    def coordinate(self):
        if self.flags['avoid']:
            action_wheels = self.activations['avoid']
        elif self.flags['load']:
            action_wheels = self.activations['load']
        elif self.flags['forage']:
            action_wheels = self.activations['forage']
        else: 
            action_wheels = self.activations['nav']
        return {'joint_velocity_actuator' : np.array(action_wheels)}

    def navigate(self, state):
        pass

    def forage(self, state):
        mgs_read = state['memory_ground_sensor']
        self.flags['forage'] = False 
        if mgs_read == 1: # Garbage collected
            ls_read = state['red_light_sensor']
            if ls_read[0] * ls_read[7] == 0:
                self.flags['forage'] = True
                light_left= np.sum(ls_read[[7,6,5,4]])
                light_right = np.sum(ls_read[[0,1,2,3]])
                if light_right > light_left:
                    self.activations['forage'] = np.array([-1, 1]) 
                else: 
                    self.activations['forage'] = np.array([1, -1]) 


@controller_registry(name="turn_yellow_lights_OFF")
class TurnYellowLightsOFFController(RobotController):
    def __init__(self, *args,  **kwargs):
        super(TurnYellowLightsOFFController, self).__init__(*args, **kwargs)
        self.flag = False
        self.lights_off = 0
        self.target_lights = 3
        self.light_threshold = 0.1  #distance to consider the light is reached

        self.recorded_positions = []

        
    def step(self, state, reward=0):

        if self.lights_off >= self.target_lights:
            if self.flag:
                print(f"Objetivo alcanzado: {self.lights_off} luces amarillas apagadas. Deteniendo rutina.")
                self.flag = False #desactivar rutina al apagar todas las luces amarillas
            action_wheels = np.array([0., 0.])
            action_light = 0.0


        ls_read = self.get_sensor_reading('yellow_light_sensor')
        max_light = np.max(ls_read)  #Luz más cercana

        action_wheels = np.array([1., 1.]) # Moverse hacia adelante (Exploración)
        action_light = 0.0  # Por defecto: No intentar apagar

        # Si la luz es muy intensa (estamos muy cerca), paramos para asegurar el apagado y registramos.
        if max_light > 0.9 and self.flag: 
            
            action_wheels = np.array([0., 0.]) 
            action_light = 1.0 

            # Registro
            current_pos = self.get_sensor_reading('gps') 
            is_new_light = True
            for pos in self.recorded_positions:
                if np.linalg.norm(current_pos - pos) < 0.5:  #chechk if the light position is new
                    is_new_light = False
                    break
            
            if is_new_light:
                self.recorded_positions.append(current_pos)
                self.lights_off += 1
                print(f"Luz amarilla APAGADA/REGISTRADA. Contador: {self.lights_off}. Posición: {current_pos}")
            else:
                action_wheels = 1.0 * np.array([1., 1.]) 
                action_light = 0.0
                print("Luz amarilla ya registrada previamente. No se incrementa el contador.")
        
        # Orientación y Búsqueda
        elif max_light > self.light_threshold: 

            # Si no estamos en proximidad máxima, alineamos y avanzamos.
            if ls_read[0] * ls_read[7] == 0:
                # Lógica de Giro: Luz descentrada (Sectores 0 y 7 no leen a la vez)
                light_left = np.sum(ls_read[[7, 6, 5, 4]])
                light_right = np.sum(ls_read[[0, 1, 2, 3]])
                
                # Velocidad de giro (usa un valor más alto que 0.1, por ejemplo 0.5)
                turn_speed = 0.2
                
                if light_right > light_left:
                    action_wheels = turn_speed * np.array([-1, 1])  # Girar izquierda
                else: 
                    action_wheels = turn_speed * np.array([1, -1])  # Girar derecha
            else:# Luz centrada
                action_wheels = 0.7 * np.array([1, 1]) # ¡AVANZAR HACIA ELLA!
            pass
        
        # 3. Aplicar acciones
        self.get_actuator('switch_light').action = action_light
        self.get_actuator('joint_velocity_actuator').action = action_wheels
"""
        # Si la rutina llega al final sin un 'return' y sin encontrar luz, debería seguir con la exploración por defecto ([1., 1.])
        if self.lights_off < self.target_lights and max_light < self.light_threshold:
            self.get_actuator('joint_velocity_actuator').action = np.array([1., 1.]) # Explorar si no hay luz
            self.flag = True

            
"""

@controller_registry(name="turn_yellow_lights_ON")
class TurnYellowLightsONController(RobotController):
    def __init__(self, *args, **kwargs):
        super(TurnYellowLightsONController, self).__init__(*args, **kwargs)
        self.flag = False
        self.targets = [] # posiciones de las luces a encender
        self.proximity_threshold = 0.2
        self.current_target_idx = 0


    def set_targets(self, positions):
        self.targets = positions
        self.current_target_idx = 0
        print(f"Recibidas {len(self.targets)} posiciones para encender.")

    def step(self, state, reward=0):
        
        if self.current_target_idx >= len(self.targets):
            self.flag = False
            return
        

        self.flag = True
        current_pos = self.get_sensor_reading('gps')
        current_theta = self.get_sensor_reading('compass') * 2 * np.pi # Corregir escala 
        
        target_pos = self.targets[self.current_target_idx]

        # Calcular vector hacia el objetivo
        diff = target_pos - current_pos
        dist = np.linalg.norm(diff)

        # Lógica de navegación simple hacia coordenada
        action_wheels = np.array([1., 1.])
        action_light = 0.0

        if dist < self.proximity_threshold:
            # Hemos llegado: Encender luz y pasar a la siguiente
            action_wheels = np.zeros(2) # Parar
            action_light = 1.0 # Encender (asumiendo 1.0 es ON)
            print(f"Luz {self.current_target_idx + 1} ENCENDIDA.")
            # Pasar al siguiente objetivo
            self.current_target_idx += 1

            if self.current_target_idx >= len(self.targets):
                print("Misión de encendido completada.")
                self.flag = False       
        else:
            target_angle = np.arctan2(diff[1], diff[0])
            alpha = target_angle - current_theta
            alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
            if abs(alpha) < 0.2: 
                action_wheels = np.array([1.0, 1.0]) * 0.5
            else:
                if alpha > 0:
                    action_wheels = np.array([-0.5, 0.5]) * 0.5 # Girar Izquierda
                else:
                    action_wheels = np.array([0.5, -0.5]) * 0.5
            action_light = 0.0

        self.get_actuator('joint_velocity_actuator').action = action_wheels
        self.get_actuator('switch_light').action = action_light
            