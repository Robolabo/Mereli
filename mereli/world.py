import logging
import time
import csv
import copy
from collections import deque
import numpy as np
import os
    



from mereli.objects import  Robot, Wall, Map, GroundArea
from mereli.physics_engines.pybullet_engine import PybulletEngine
from mereli.register import (controllers, world_objects, initializers, dones, rewards, 
    env_perturbations, communication_systems, world_registry, done_registry)
from mereli.tasks.task import TaskManager
from mereli.utils import (increase_time, mov_average_timeit, isinstance_of_any)
from mereli.globals import global_states
from mereli.objectives import done
from mereli.tasks import TaskManager
from mereli.communication import CommunicationSpace
from mereli.data_logging import DataLogger
from mereli.animated_graph import AnimatedLayout
try:
    from mereli.dashboard.connection import DashboardConnection
except:
    pass

# LLM imports
import json
import time
from multiprocessing import Process, Manager
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


def llm_brain_loop(shared_data, system_rules_content, base_url, num_robots):
    """
    PROCESO INDEPENDIENTE: El Cerebro Central.
    Este bucle corre en un núcleo de CPU distinto al del robot.
    """
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage, SystemMessage
    import json
    import time
    import re

    print(f"🧠 [CEREBRO CENTRAL]: Iniciando proceso hijo. Conectando...")

    try:
        # Inicialización del modelo dentro del proceso hijo
        llm = ChatOllama(
            model="gpt-oss:20b",
            temperature=0,
            base_url= base_url,
            keep_alive="5m"
        )
    
        sys_msg = SystemMessage(content=system_rules_content)
        print("🧠 [CEREBRO CENTRAL]: Proceso de IA iniciado y listo.")
    
    except Exception as e:
        print(f"❌ [CEREBRO CENTRAL ERROR FATAL]: No se pudo inicializar ChatOllama: {e}")
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

                # 3. Escribir las decisiones en la pizarra para que los robots las lean
                default_decision = shared_data.get('default_decision', 'simple_forage')
                decisions = data.get('decisions', [default_decision] * num_robots)
                memoria = data.get('memoria_interna', '')
                razonamiento = data.get('razonamiento', '')
                luces_trianguladas = data.get('luces_trianguladas', [])

                response_wall_time = time.perf_counter()
                shared_data['decisions'] = decisions
                shared_data['memoria_interna'] = memoria
                shared_data['razonamiento'] = razonamiento
                shared_data['luces_trianguladas'] = luces_trianguladas
                shared_data['decision_ts'] = current_sensor_ts
                shared_data['llm_request_wall_time'] = request_wall_time
                shared_data['llm_response_wall_time'] = response_wall_time
                shared_data['llm_latency_s'] = response_wall_time - request_wall_time
                last_processed_sensor_ts = current_sensor_ts

            
            except Exception as e:
                print(f"❌ [CEREBRO CENTRAL ERROR]: {e}")
            finally:
                shared_data['processing'] = False

        # Evitar consumo excesivo de CPU en el bucle de espera
        time.sleep(0.05)

def map_parser():
    file = 'mereli/models/maps/map1.txt'
    with open(file) as f:
        map_mat = np.array([[int(ch) if ch != '' else 0 for ch in line.split(';')[0].split(' ')] for line in f.readlines()])
    import pdb; pdb.set_trace()


class GlobalMap(object):
    def __init__(self, *args, **kwargs):
        self.Nx = 200
        self.Ny = 200
        self.dx = 0.05 #metres 
        self.dy = 0.05 #metres
        self._map = np.zeros((self.Nx, self.Ny)).astype(int) 

    def get_tile(self, position):
        """  Returns the value of the discrete map and the (i,j) indices correspoding to position (x,y). """
        i = int(position[0] // self.dx + self.Nx // 2)
        j = int(position[1] // self.dy + self.Ny // 2)

        return (i, j), self.map[i, j]
    
    def set_tile(self, position, value):
        """ Updats the value of the discrete map at the (i,j) indices correspoding to position (x,y). """
        i = int(position[0] // self.dx + self.Nx // 2)
        j = int(position[1] // self.dy + self.Ny // 2)
        self._map[i,j] = value
    
    def build_map(self, entities):
        pass
        # for name, entity in entities.items():
        #     if 'wall' in name:
        #         # self.paint_wall(entity)
        #     elif isinstance(entity, GroundArea):
        #         self.paint_ground_area(entity)
            
        # __import__('pdb').set_trace() 
    
    def paint_ground_area(self, ground_area):
        pos = ground_area.position[:2]
        R = ground_area.radius
        for x in np.linspace(pos[0]-R, pos[0]+R, num=int((2*R)//self.dx)):
            for y in np.linspace(pos[1]-R, pos[1]+R, num=int((2*R)//self.dy)):
                if np.linalg.norm(np.r_[x, y] - pos) < R:
                    self.set_tile((x,y),2)

    def paint_wall(self, wall):
        pos = wall.position[:2]
        ori = wall.orientation[-1]
        W = wall.width
        H = wall.height
        for x in np.linspace(pos[0]-W/2, pos[0]+W/2, num=int(W//self.dx)):
            for y in np.linspace(pos[1]-H/2, pos[1]+H/2, num=int(H//self.dx)):
                self.set_tile((x, y), 1)

        

class World(object):
    # language=rst
    """ Base class of the world or environment. This class is never used directly in an experiment but 
    any experiment environment/world must inherit from it. The main function of World classes is to act as
    containers and orchestrator of the simulation. It stores all the objects that have been instantiated, 
    calls the :py:meth:`mereli.objects.Robot.step` method of every robot in order to map states into 
    actions and communicates with the physics and render engines in order to simulate and visualize rigid object 
    realistic physics and collisions. 
    Currently only 2D and 3D square arenas are implemented. 

    :param Engine physics_engine: physics engine to be used.
    :param float height: height in metres of the square arena.
    :param float width: width in metres of the square arena.
    :param float world_delay: deprecated, to be removed in next ver.
:var dict hierarchy: dictionary storing all the entities instantiated in the world.
    :var dict initializers: dictionary mapping groups of entities to initializers of the positions 
            and orientations
    :var dict env_perturbations: dictionary mapping object groups to environmental perturbations (``EnvironmentalPerturbation``) applied 
        to robot states or actions.

    Example::

    >>> # Example of an obstacle avoidance experiment with 5 robots in 3D. 
    >>> from mereli import SquareArena
    >>> from mereli.physics_engines import PybulletEngine
    >>> from mereli.objects import Robot3D
    >>> from mereli.controllers import BasicObstacleAvoider
    >>> from mereli.utils.initializers import InitializerHandler, RandomUniformInitializer
    >>>
    >>> n_robots = 5
    >>> phy_engine = PybulletEngine(dt=0.02)
    >>> world = SquareArena(phy_engine, height=10, width=10)
    >>> world_cfg = {
    >>>     "world_delay" : 1,
    >>>     "height": 10,
    >>>     "width":  10,
    >>>     "objects" : {
    >>>        "robotA" : {
    >>>            "type" : "robot",
    >>>            "num_instances" : n_robots,
    >>>            "controller" : "basic_obstable_avoider",
    >>>            "sensors" : {
    >>>                "distance_sensor" : {"n_sectors" : 8, "range" : 1}
    >>>            },  
    >>>            "actuators" : {
    >>>                "joint_velocity_actuator" : {"joint_ids" : [0, 1], "max_velocity" : 5}
    >>>            },
    >>>            "initializers" : {
    >>>                "positions" : {"name" : "random_uniform",  "params" : {"low" : [-3, -3], "high" : [3, 3], "size" : 2}},
    >>>                "orientations" : {"name" : "random_uniform",  "params" : {"low":0, "high" : 6.28, "size" : 1}}
    >>>            },
    >>>            "perturbations" : {
    >>>            },
    >>>            "params" : {"trainable" : True}
    >>>        }
    >>>     }
    >>> }
    >>> world.build_from_dict(world_cfg)
    >>> with world:
    >>>     while(True):
    >>>         state, action = world.step()

    """
    def __init__(self, physics_engine):
        self.physics_engine = physics_engine
        self.render = global_states.RENDER

        #* Dict storing all objects
        self.hierarchy = {}
        self.task_manager = None
        #* Dict mapping object names to object groups
        self.groups = {}
        #* Dict storing how objects should be initialized as a group.
        self.initializers = {}
        #* Dict mapping object groups to environmental perturbations
        self.env_perturbations = {}
        self.neighbors = {}
        self.virtual_space = None
        self.neighbor_matrix = None
        self.done_signal = None
        self.data_logger = None
        self.t = 0
        self.paused = global_states.INTERACTIVE
        self.start_paused = False
        self.dashboard_conn = DashboardConnection() if global_states.INTERACTIVE else None 
        self.animated_layout = None
        self.__robots = {} 
        self._is_done = False
        self.global_map = GlobalMap()
        
        # LLM Central attributes
        self.llm_manager = None
        self.llm_shared_data = None
        self.llm_process = None
        self.llm_orders = {}  # dict of robot_name -> routine
        self.central_llm_initial_request_sent = False
        self.central_llm_decision_ready = False
        self.last_read_ts = 0
        

    def log_llm_timing(self, request_step, apply_step, latency_s, decision, scope="central", robot="all"):
        blind_steps = int(apply_step) - int(request_step)
        async_ratio = blind_steps / latency_s if latency_s > 0 else 0.0
        output_dir = os.environ.get("CURRENT_EXP_FOLDER", "outputs")
        metrics_path = os.path.join(output_dir, "llm_tiempos.csv")
        file_exists = os.path.exists(metrics_path)
        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists or os.path.getsize(metrics_path) == 0:
                writer.writerow([
                    "scope", "robot", "request_step", "apply_step",
                    "blind_steps", "latency_s", "async_ratio", "decision"
                ])
            writer.writerow([
                scope, robot, int(request_step), int(apply_step),
                blind_steps, f"{latency_s:.6f}", f"{async_ratio:.6f}", json.dumps(decision)
            ])

    def euclidean_2d(self, a, b):
        return float(np.linalg.norm(np.array(a[:2], dtype=float) - np.array(b[:2], dtype=float)))

    def collect_real_red_lights(self):
        real_lights = []
        for name, obj in self.hierarchy.items():
            if getattr(obj, "color", None) != "red" or not hasattr(obj, "position"):
                continue
            real_lights.append({
                "name": name,
                "position": [float(obj.position[0]), float(obj.position[1])],
            })
        return real_lights

    def nearest_real_light(self, source_position, real_lights):
        if not real_lights:
            return None, float("nan")
        best_light = min(real_lights, key=lambda light: self.euclidean_2d(source_position, light["position"]))
        return best_light, self.euclidean_2d(source_position, best_light["position"])

    def save_spatial_light_errors(self):
        if self.llm_shared_data is None:
            return

        real_lights = self.collect_real_red_lights()
        if not real_lights:
            return

        rows = []
        triangulated = self.llm_shared_data.get('luces_trianguladas', [])
        for idx, item in enumerate(triangulated):
            source_position = item.get('coordenada_estimada') or item.get('position') or item.get('pos')
            if source_position is None or len(source_position) < 2:
                continue
            source_position = [float(source_position[0]), float(source_position[1])]
            real_light, error = self.nearest_real_light(source_position, real_lights)
            if real_light is None:
                continue
            rows.append({
                "metric_type": "central_triangulated",
                "highlight": "CENTRAL_LLM",
                "real_light_name": real_light["name"],
                "real_x": real_light["position"][0],
                "real_y": real_light["position"][1],
                "source": f"triangulated_{idx}",
                "source_x": source_position[0],
                "source_y": source_position[1],
                "error_euclidean": error,
                "matched_by": "nearest_real",
            })

        for robot_name, robot in self.robots.items():
            controller = getattr(robot, "controller", None)
            found_red_lights = getattr(controller, "found_red_lights", {})
            for key, info in found_red_lights.items():
                source_position = info.get("light_position")
                if source_position is None or len(source_position) < 2:
                    continue
                source_position = [float(source_position[0]), float(source_position[1])]
                real_light, error = self.nearest_real_light(source_position, real_lights)
                if real_light is None:
                    continue
                rows.append({
                    "metric_type": "robot_registered",
                    "highlight": "",
                    "real_light_name": real_light["name"],
                    "real_x": real_light["position"][0],
                    "real_y": real_light["position"][1],
                    "source": robot_name,
                    "source_x": source_position[0],
                    "source_y": source_position[1],
                    "error_euclidean": error,
                    "matched_by": "nearest_real",
                })

        if not rows:
            return

        output_dir = os.environ.get("CURRENT_EXP_FOLDER", "outputs")
        csv_path = os.path.join(output_dir, "spatial_light_errors.csv")
        fieldnames = [
            "metric_type", "highlight", "real_light_name", "real_x", "real_y",
            "source", "source_x", "source_y", "error_euclidean", "matched_by"
        ]
        rows.sort(key=lambda row: (row["metric_type"] != "central_triangulated", row["real_light_name"], row["source"]))
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                formatted = dict(row)
                for col in ["real_x", "real_y", "source_x", "source_y", "error_euclidean"]:
                    formatted[col] = f"{float(formatted[col]):.6f}"
                writer.writerow(formatted)

        table_path = os.path.join(output_dir, "spatial_light_errors_table.txt")
        with open(table_path, "w") as f:
            f.write("SPATIAL LIGHT ERRORS\n")
            f.write("====================\n\n")
            for metric_type, title in [
                ("central_triangulated", "CENTRAL TRIANGULATED"),
                ("robot_registered", "ROBOT REGISTERED"),
            ]:
                metric_rows = [row for row in rows if row["metric_type"] == metric_type]
                if not metric_rows:
                    continue
                f.write(f"{title}\n")
                f.write("real_light           real_pos             source              estimated_pos        error\n")
                f.write("----------------------------------------------------------------------------------------\n")
                for row in metric_rows:
                    mark = "  <-- CENTRAL_LLM" if row["highlight"] == "CENTRAL_LLM" else ""
                    real_pos = f"({row['real_x']:.3f},{row['real_y']:.3f})"
                    source_pos = f"({row['source_x']:.3f},{row['source_y']:.3f})"
                    f.write(
                        f"{row['real_light_name']:<20} {real_pos:<20} "
                        f"{row['source']:<19} {source_pos:<20} "
                        f"{row['error_euclidean']:.6f}{mark}\n"
                    )
                f.write("\n")

    def avisar_central(self, robot_name, mensaje):
        # Aviso de un robot al LLM central.
        if self.llm_shared_data is None:
            return
        ordenes = [
            self.llm_orders.get(name, self.llm_shared_data.get('default_decision', 'Espera'))
            for name in self.robots.keys()
        ]
        memoria = self.llm_shared_data.get('memoria_interna', '')
        ready_robots = []
        charging_robots = []
        estado_baterias = ""
        for name, robot in self.robots.items():
            bat_azul = robot.state.get('blue_battery_sensor', [0])[0]
            estado_baterias += f"- {name}: bat_azul={bat_azul:.2f}\n"
            order = self.llm_orders.get(name, self.llm_shared_data.get('default_decision', 'Espera'))
            # Si este robot acaba de terminar de cargar, se considera LISTO
            # independientemente del orden anterior.
            if name == robot_name and "carga" in mensaje.lower() and "completada" in mensaje.lower():
                ready_robots.append(name)
            elif "cargar" in order.lower() or "load" in order.lower():
                charging_robots.append(name)
            elif bat_azul >= 0.90 and order == 'Espera':
                ready_robots.append(name)
        # El central decide con el aviso, no con sensores nuevos.
        self.llm_shared_data['contexto'] = (
            f"t={self.t}\n"
            f"AVISO LOCAL: {robot_name}: {mensaje}\n"
            f"Ordenes actuales: {ordenes}\n"
            f"--- ESTADO REAL DE LAS BATERÍAS ---\n{estado_baterias}\n"
            f"Robots en espera listos para explorar: {ready_robots}\n"
            f"Robots actualmente cargando: {charging_robots}\n"
            f"Memoria previa: {memoria}\n"
            "Replanifica usando este aviso local. No mires baterias actuales para retirar robots que ya estaban explorando.\n"
        )
        self.llm_shared_data['sensor_ts'] = max(int(self.t), int(self.llm_shared_data.get('sensor_ts', 0)) + 1)
        self.llm_shared_data['last_t'] = self.t
        print(f"📨 [LOCAL -> CENTRAL] {robot_name}: {mensaje}")


    def update_neighbor_matrix(self):
        rad = 1 
        # rad = 1 
        if self.virtual_space is not None:
            rad = 200 
            if self.virtual_space.randomize_neighbors: 
                rad = 2 
        positions = np.vstack([robot.position[:2] for robot in self.robots.values()])
        aux_mat = np.multiply.outer(np.ones(len(self.robots)), positions)
        dist_mat = np.linalg.norm(aux_mat - np.transpose(aux_mat, (1, 0, 2)), axis=2)
        self.neighbor_matrix = np.all((dist_mat < rad, dist_mat > 0), axis=0)
        robot_names = np.array(tuple(self.robots.keys()))
        for i in range(len(robot_names)):
            self.neighbors[robot_names[i]] = robot_names[self.neighbor_matrix[i]] 
        

    def schedule_workload(self):
        num_robots = len(self.robots)
        Ts = 10 # Control loop executed every 10 times the sim dt.
        # Ts=1
        t_iter = int(self.t % Ts)
        niter = np.floor(num_robots / Ts)
        nremaining = num_robots % Ts
        
        if self.t < Ts:
            return np.arange(num_robots) if self.t == 0 else []
        if t_iter < nremaining:
            nsel = niter + 1 
            selected = np.arange(t_iter * (niter + 1), (t_iter + 1) * (niter + 1))
        else:
            nsel = niter
            selected = np.arange(nremaining * (niter + 1) + (t_iter - nremaining) * niter, nremaining * (niter + 1) + (t_iter - nremaining + 1) * niter)
        return selected
        # print(f'In t={self.t} and titer= {t_iter} {nsel} robots where selected')

    # @increase_time
    @mov_average_timeit
    def step(self):
        # language=rst
        """ Step function of the world to run it one timestep. This method is must be executed at every step of 
        the simulation in order to iterate the physics and robot controllers.
        
        The main functions of the method are:

        * It computes the swarm rewards based on the previous action and states.
        * For each instantiated controllable entity, the :py:meth:`step` method is executed. This results in the partially observable state measured by the robot sensors and the corresponding 
          actions elaborated by the controller. These states and actions are python ``dict`` objects mapping sensor 
          and actuator names to numpy arrays of measured states and actions. The states and actions of all the robots 
          in the swarm are gathered as a numpy array of python ``dict`` objects (each corresponding to a robot).
            
          Example::

          >>> state_obj = {'distance_sensor' : np.array([0, 0, 0, 1]), 'ground_sensor' : np.array([0])}
          >>> action_obj = {'joint_velocity_actuator' : np.array([0.4, -0.1])}

        * Perturbations are applied to the planned actions. For example, a robot communication transmitter can 
          be broken and its action is, therefore, inhibited.
        * Actuators of the robots are executed with the actions planned by the controllers.
        * The render and physics engines are iterated. The render engine is iterated only if 
          :py:attr:`mereli.World.render` is ``True``. 

        :returns: A tuple with state and action numpy arrays of length equal to the number of robots. 
                  Each of these arrays contain python ``dict`` objects representing the states and actions of each controllable entity.
        """
        if self.physics_engine.paused and not self.physics_engine.paused_step:
            if self.render:
                self.physics_engine.step_render()
            return {}, {}
        t0 = time.time() 
        states = deque()
        actions = deque()
        if len(self.robots) > 0:
            self.update_neighbor_matrix()
            # selected = np.arange(len(self.robots))#self.schedule_workload()
            selected = self.schedule_workload()
        #* Step controllers
        for idx, (obj_name, obj) in enumerate(self.controllable_objects.items()):
            # if not issubclass(type(obj), Robot):
            #     # Step non-robot entities (e.g. dynamic lights)
            #     obj.step()
            #     continue
            obj.awaken = idx in selected 

            # Update neighborhood of robots 
            obj.neighbor_names = self.neighbors[obj_name]
            obj.neighbors = [self.robots[ngh] for ngh in self.neighbors[obj_name]]
            obj.step()
            # if True:
                # Apply sensor perturbations/constrains if any
                # #* Compute robot reward 
                # reward = self.reward_generator(self.prev_actions, self.prev_states, obj, info=self.hierarchy.values())\
                #         if self.reward_generator is not None else None
                 
                # Actual step of the robot - Executes sensors, control and actuator planners
                # state_obj, action_obj = obj.step(self.hierarchy.values(), perturbations=pre_perturbations) #!
            # else:
                # print(obj.state, obj.actions)
                
            states.append(obj.state)
            actions.append(obj.actions)
        # return [],[]
        if len(states) > 0:
            states = np.stack(states)
        if len(actions) > 0:
            actions = np.stack(actions)
        # print('Real.sp t0ime: ', time.time() - t0)

        #* Actuate based on the actions planned by the robot.step() method
        for obj in self.robots.values():
            obj.actuate()
        # Apply task manager (if any) when there is an opt or eval process on top 
        if self.task_manager is not None:
            self.task_manager(self.robots)#self.hierarchy)
        # Step the virtual/communication space controllers (if any).
        if self.virtual_space is not None:
            self.virtual_space.step()

        # Actualizar LLM central si existe. El central solo planifica una vez al
        # inicio; despues los robots locales gestionan sus subtareas.
        if self.llm_shared_data is not None:

            # --- LÓGICA DE TRIANGULACIÓN AL FINAL DE LA SIMULACIÓN ---
            if self.is_done and not getattr(self, 'triangulacion_completada', False):
                self.triangulacion_completada = True
                todas_luces_encontradas = []
                
                # Recopilamos las memorias de todos los robots
                for robot_name, robot in self.robots.items():
                    if hasattr(robot.controller, 'found_red_lights'):
                        for info in robot.controller.found_red_lights.values():
                            if info.get('light_position') is not None:
                                todas_luces_encontradas.append((robot_name, info['light_position']))
                
                msg_luces = "¡SIMULACIÓN TERMINADA! Aquí tienes los reportes de luces exploradas por todos los robots:\n"
                for r_name, pos in todas_luces_encontradas:
                    msg_luces += f"- {r_name} reporta luz roja en {pos}\n"
                msg_luces += "\nCRUZA LOS DATOS, DESCARTA DUPLICADOS Y DEDUCE LAS COORDENADAS REALES DE LAS LUCES EN EL JSON."
                
                print(f"\n🌍 [MUNDO]: Fin de simulación (Step {self.t}). Disparando triangulación al Cerebro Central...")
                
                # Guardamos el timestamp actual del LLM para saber cuándo ha respondido
                ts_espera = self.llm_shared_data.get('decision_ts', 0)
                
                self.avisar_central("SISTEMA", msg_luces)
                
                print("⏳ Esperando el veredicto final de triangulación del LLM...")
                # Bucle que congela el cierre de la simulación hasta que el LLM responda
                while self.llm_shared_data.get('decision_ts', 0) <= ts_espera:
                    time.sleep(0.5)
                
                # Imprimimos el resultado glorioso
                print("\n" + "="*60)
                print("🎯 [VEREDICTO FINAL - TRIANGULACIÓN DE LUCES]")
                print("="*60)
                print(f"🧠 RAZONAMIENTO:\n{self.llm_shared_data.get('razonamiento', '')}")
                print(f"\n📍 LUCES TRIANGULADAS (JSON):\n{json.dumps(self.llm_shared_data.get('luces_trianguladas', []), indent=2)}")
                self.save_spatial_light_errors()
                print("="*60 + "\n")

            if not self.central_llm_initial_request_sent:
                # Recopilar estado global
                contexto = f"t={self.t}\n"
                for i, (robot_name, robot) in enumerate(self.robots.items()):
                    bat_azul = robot.state.get('blue_battery_sensor', [0])[0]
                    bat_roja = robot.state.get('red_battery_sensor', [0])[0]
                    n_luces = getattr(robot.controller.routines.get('turn_yellow_lights_OFF'), 'lights_off', 0) if hasattr(robot.controller, 'routines') else 0
                    contexto += f"Robot {i} ({robot_name}): bat_azul={bat_azul:.2f}, bat_roja={bat_roja:.2f}, luces_apagadas={n_luces}\n"
                self.llm_shared_data['contexto'] = contexto
                self.llm_shared_data['sensor_ts'] = max(1, int(self.t))
                self.llm_shared_data['last_t'] = self.t
                self.central_llm_initial_request_sent = True
                
            current_decision_ts = self.llm_shared_data.get('decision_ts', 0)
            if current_decision_ts > self.last_read_ts:
                decisions = self.llm_shared_data.get('decisions', [])
                razonamiento = self.llm_shared_data.get('razonamiento', '')
                memoria = self.llm_shared_data.get('memoria_interna', '')
                for i, robot_name in enumerate(self.robots.keys()):
                    if i < len(decisions):
                        self.llm_orders[robot_name] = decisions[i]
                self.central_llm_decision_ready = True
                latency_s = float(self.llm_shared_data.get('llm_latency_s', 0.0))
                self.log_llm_timing(current_decision_ts, self.t, latency_s, decisions, scope="central", robot="all")
                self.last_read_ts = current_decision_ts
                print(f"\n🧠 [RAZONAMIENTO CENTRAL]: {razonamiento}")
                print(f"📖 [MEMORIA GLOBAL]: {memoria}")
                print(f"🎯 [DECISIONES | t={self.t}]: {decisions}\n")

        #* Render and physics step.
        self.physics_engine.step_physics()
        if self.render:
            self.physics_engine.step_render()
            if not self.physics_engine.paused and self.animated_layout is not None:
                if self.physics_engine.camera_options['focus']:
                    focus_id = self.physics_engine.camera_options['focus_target']
                    if focus_id in selected:
                        target_robot = list(self.robots.values())[focus_id]
                        self.animated_layout.update(target_robot)
        if global_states.LOG:
            if self.is_done:
                self.data_logger.set_log_file()
                self.data_logger.save_pickle()
                print('Logs saved!')
            else:
                self.data_logger.update()
        self.t += 1
        # print('Simulation step elapsed ', time.time() - t0)
        # self.global_map.build_map(self.hierarchy)
        return states, actions

    def focused_robot(self):
        isfocus = self.physics_engine.camera_options['focus']
        if isfocus:
            focus_id = self.physics_engine.camera_options['focus_target']
            target_robot = list(self.robots.values())[focus_id]
            return target_robot

    def register_entity(self, name, obj, group=None):
        """ 
        Adds an object to the world registry. Assigns a unique identifier to the object.
        Additionally, if the object belongs to a group of world objects it also registers it.

        :param str name: name of the object.
        :param WorldObject obj: instance of the world object to be added.
        :param str group: name of the group to which obj belong to. If none a new group is created with
                           obj as unique element.
        """
        self.hierarchy.update({name : obj})
        if issubclass(type(obj), Robot):
            self.__robots.update({name : obj})
            obj.world = self  # Referencia al world para controladores
            obj.name = name  # Nombre del robot para controladores
        #* Register group element
        if group is None:
            group = name
        if group in self.groups.keys():
            self.groups[group].append(name)
        else:
            self.groups[group] = [name]
        obj.group = group 
        obj.gid = len(self.groups[group]) - 1

    def set_initializer(self, group_name, initializer_pos, initializer_ori=None):
        """ Bounds and registers entity initializers to groups of entities. 
        Therefore, every time that an experiment is reset, the settled initializers are used 
        to establish the positions and orientations (only if the entity is a robot) of 
        the world entities. 

        :param str group_name: name of the group of entities to be created.
        :param Initializer initializer_pos: initializer class of the positions of the group.
        :param Initializer initializer_ori: initializer class of the orientations of the group.
        """
        self.initializers[group_name] = {
            'positions' : initializer_pos,
            'orientations' : initializer_ori
        }

    def add_group(self, group_name, entity_cls,  initializer_pos, initializer_ori=None, controller=None):
        """ Adds entities belonging to a certain group to the world. For instance, it can create all the 
        homogeneous robots within a swarm.

        :param str group_name: name of the group of entities to be created.
        :param WorldObject entity_cls: precise class of the entities of the group.
        :param Initializer initializer_pos: initializer class of the positions of the group.
        :param Initializer initializer_ori: initializer class of the orientations of the group.
        :param Controller controller: controller (if any) of the entities of the group. If the entity is not 
                controllable then its content must be None
        """
        #* Create group intializers.
        self.initializers[group_name] = {
            'positions' : initializer_pos,
            'orientations' : initializer_ori
        }
        positions = self.initializers[group_name]['positions']()
        orientations = self.initializers[group_name]['orientations']()\
                        if initializer_ori is not None else 5*[[0,0,0]]
        for i, pos, ori in enumerate(zip(positions, orientations)):
            entity = entity_cls(pos, ori, controller=controller)
            ent_name = group_name + '_' + i
            self.add_entity(ent_name, entity_cls, pos, ori, controller=controller, group_name=group_name)
        
    def create_virtual_space(self, topology=None, **vspace_cfg):
        from mereli.register import comm_spaces
        comm_space_cls = comm_spaces[vspace_cfg['name']]
        self.virtual_space = comm_space_cls(**vspace_cfg.get('params', {}))
        for robot_name, robot in self.robots.items():
            self.virtual_space.add_particle(robot_name, robot)
            if vspace_cfg['name'] != "VirtualPhysicsCommSpace":
                self.virtual_space.particles[robot_name].set_controller(topology=topology)
        if isinstance(vspace_cfg['landmarks'], list):
            for lmk in vspace_cfg['landmarks']:
                self.virtual_space.add_landmark(lmk['state'], mass=lmk['mass'])
        else:
            if vspace_cfg['landmarks']['num_lmarks'] > 0:
                for lmk in range(vspace_cfg['landmarks']['num_lmarks']):
                    pos = vspace_cfg['landmarks']['positions'] 
                    lm_pos = vspace_cfg['landmarks'].get('scale',1)*np.array(pos[lmk]) if pos != "random" else None
                    self.virtual_space.add_landmark(lm_pos)

    def config_data_logger(self, log_info):
        self.data_logger = DataLogger()
        self.data_logger.configure(self, log_info)
        

    # def add_robot(self, robot_type, group, position=[0,0,0], orientation=0.0, controller=None):
    #     robot = Epuck()


    def build_from_dict(self, world_dict, ann_topology=None, user_instructions=""):
        """ 
        Initializes all the entities and adds them to the world/environment using a ``dict`` structure as input.
        The ``world_dict`` fully defines the environment and the instatiated robots and the ``ann_topology`` 
        entirely establishes the ANN controller topology (if :py:class:`mereli.controllers.NeuralController` is used).
        For a dedicated description of the configuration files fields see `Configuration Files <configuration_files.html>`__ .

        :param dict world_dict: configuration ``dict`` of the environment (parameters, objects, ...).
        :param dict ann_topology:  configuration ``dict`` of the neural network.
        """
        if 'task_manager' in world_dict:
            self.task_manager = TaskManager(duration=world_dict.get('task_manager',{}).get('total_duration', 1000), 
                                    num_slots=world_dict.get('task_manager',{}).get('num_slots', 1), 
                                    use_done=world_dict.get('task_manager',{}).get('use_done', False))
            for task in world_dict.get('task_manager', {}).get('tasks', []):
                self.task_manager.add_task(task['name'], **task['params'])
        if 'done_signal' in world_dict:
            self.done_signal = dones[world_dict['done_signal']['name']](**world_dict['done_signal'].get('params', {}))
        for obj_name, obj in world_dict['objects'].items():
            object_cls = world_objects[self.physics_engine.engine_type][obj['type']]
            #! Prov implementation for TFM regarding the task scheduler
            if object_cls.__name__ == 'TaskScheduler':
                world_obj = object_cls(None, np.zeros(3), np.zeros(3), **obj['params']) #! ojo 2D
                self.register_entity(obj_name + '_' + str(i), world_obj, group=obj_name)
                continue
            num_entities = obj['num_instances']
            #* Loop entities and add them to the world.
            if issubclass(object_cls, Robot):# or issubclass(object_cls, Robot3D):
                for i in range(num_entities):
                    controller = None
                    if obj.get('controller', False):
                        #* Create Controller and add sensors and actuators
                        controller_cls = controllers[obj['controller'] if not isinstance(obj['controller'], dict) else obj['controller']['name']]
                        params = obj['controller']['params'] if isinstance(obj['controller'], dict) else {}
                        controller = controller_cls(**params)
                        controller.add_sensors_from_dict(obj['sensors'])
                        controller.add_actuators_from_dict(obj['actuators'])
                        if issubclass(controller_cls, controllers['neural_controller']):
                            controller.add_ann_from_dict(ann_topology[obj['controller']['topology']])
                    else:
                        controller = controllers['dummy_controller']()
                        # controller.add_sensors_from_dict({})
                        # controller.add_actuators_from_dict({})

                    positions = obj['positions']
                    orientations = obj['orientations']
                    pos_i = [0,0,0]
                    ori_i = 0.0
                    if isinstance(positions, list):
                        pos_i = positions if not isinstance(positions[0], list) else positions[i]
                    if not isinstance(orientations, str):
                        ori_i = orientations if not isinstance(orientations, list) else orientations[i]

                    #* Instantiate robot entity
                    robot = object_cls(pos_i, [0,0,ori_i], controller=controller, **obj.get('params',{}))
                    controller.controller_owner = robot
                    if isinstance(positions, dict):
                        robot.pos_init_method = positions
                    if orientations == 'random':
                        robot.ori_init_method = 'random' 
                    if 'battery' in obj:
                        if isinstance(obj['battery'], bool):
                            if obj['battery']:
                                robot.add_battery()
                        else:
                            robot.add_battery(battery_conf=obj['battery'])
                    #* If any, initialize robot's reward generator
                    # robot.reward_generator = rewards.get(obj.get('reward'))()
                    #* Add communication system (if any)
                    if "comm_sys" in obj:
                        robot.add_communication(communication_systems[obj['comm_sys']['name']](**obj['comm_sys']['params']))
                    self.register_entity(obj_name + '_' + str(i), robot, group=obj_name)
                    robot.group = obj_name
                #* Add perturbations (if any) to the robot states and actions (not physical perturbs)
                #* For example: inhibit a certain sensor reading or ignore some action of a robot.
                if obj.get('perturbations'):
                    object_perturbations = []
                    for pert_name, perturbations in obj['perturbations'].items():
                        if not isinstance(perturbations, list):
                            object_perturbations.append(env_perturbations[pert_name](obj['num_instances'], **perturbations))
                        else:
                            for i, pert in enumerate(perturbations):
                                object_perturbations.append(env_perturbations[pert_name](obj['num_instances'], **pert))
                    self.env_perturbations.update({obj_name : object_perturbations})
            else: #* Non robot objects
                for i in range(num_entities):
                    positions = obj['positions']
                    orientations = obj.get('orientations', 0.0)
                    pos_i = [0,0,0]
                    ori_i = 0.0
                    if isinstance(positions, list):
                        pos_i = positions if not isinstance(positions[0], list) else positions[i]
                    if not isinstance(orientations, dict):
                        ori_i = orientations if not isinstance(orientations, list) else orientations[i]
                    controller_cls = controllers.get(obj.get('controller'))
                    controller = controller_cls is not None and controller_cls() or None
                    world_obj = object_cls(pos_i, [0,0,ori_i], **obj.get('params',{}))
                    self.register_entity(obj_name + '_' + str(i), world_obj, group=obj_name)
                    world_obj.group = obj_name
                    if isinstance(positions, dict):
                        world_obj.pos_init_method = positions

        # Inicializar LLM central si hay robots con controlador astorekeeperLLMcentral
        has_central_llm = any(robot.controller.__class__.__name__ == 'AStoreKeeperLLMcentralController' for robot in self.robots.values())
        has_exploration_llm = any(robot.controller.__class__.__name__ == 'ExplorationGroupController' for robot in self.robots.values())
        if has_central_llm:
            self.initialize_central_llm(len(self.robots), user_instructions, mode="astorekeeper")
        elif has_exploration_llm:
            self.initialize_central_llm(len(self.robots), user_instructions, mode="exploration_group")

    def initialize_central_llm(self, num_robots, user_instructions, mode="astorekeeper"):
        """ Inicializa el LLM central para controlar múltiples robots. """
        self.llm_manager = Manager()
        self.llm_shared_data = self.llm_manager.dict()
        
        default_decision = "Espera" if mode == "exploration_group" else "simple_forage"
        # Estado inicial
        self.llm_shared_data['decision'] = [default_decision] * num_robots
        self.llm_shared_data['memoria_interna'] = 'Inicio de misión con múltiples robots.'
        self.llm_shared_data['razonamiento'] = 'Inicializando...'
        self.llm_shared_data['contexto'] = ''
        self.llm_shared_data['last_t'] = 0
        self.llm_shared_data['sensor_ts'] = 0
        self.llm_shared_data['decision_ts'] = 0
        self.llm_shared_data['processing'] = False
        self.llm_shared_data['default_decision'] = default_decision
        self.central_llm_initial_request_sent = False
        self.central_llm_decision_ready = False
        self.last_read_ts = 0

        if mode == "exploration_group":
            self.llm_rules = f"""
            Eres el cerebro central de un equipo de {num_robots} robots e-puck.
            Tu objetivo es asignar una misión a cada robot segun las instrucciones del usuario y el estado global.

            MACRO-TAREAS PERMITIDAS:
            - Puedes enviar al robot en busqueda de luces rojas para que las registre.
            - Puedes mantener al robot en espera hasta que le quieras asignar una nueva tarea.
            - Puedes enviar al robot a cargar su batería azul.

            INSTRUCCIONES DEL USUARIO: {user_instructions}

            ESTRUCTURA DE RESPUESTA (JSON):
            {{
            "razonamiento": "OBLIGATORIO usar esta fórmula -> Objetivo: N robots. Listos (bat_azul>=0.90): X. Cargando actualmente: Y. Faltan por asignar carga: N - (X+Y) = Z. Conclusión: [Explica a quién asignas en base a Z]. || Si recibes el aviso de 'SIMULACIÓN TERMINADA', explica tu razonamiento espacial para deducir cuántas luces únicas hay y dónde están...", Ignora por completo las baterías. Escribe AQUÍ tu análisis espacial.
            "memoria_interna": "Diario global breve indicando específicamente qué robots están en 'Espera' (listos) y cuáles están en cargando la bateria azul.",
            "decisions": ["tarea_robot0", "tarea_robot1", "tarea_robot2"] // UNA tarea permitida por robot. Exactamente {num_robots} elementos.
            "luces_trianguladas": [
                {{"coordenada_estimada": [x, y], "observaciones_agrupadas": 2}}
            ] // AÑADE ESTE CAMPO SOLO SI RECIBES REPORTES DE LUCES. Si no hay reporte aún, envíalo vacío [].
            }}

            REGLAS ESTRICTAS DE COORDINACIÓN Y ASIGNACIÓN:
            1. REGLA DE BATERÍA MÍNIMA: Para salir a explorar luces rojas, un robot DEBE tener bat_azul >= 0.90.
            2. REGLA DE SIMULTANEIDAD: Si el usuario pide N robots explorando, NINGÚN robot debe salir a explorar hasta que haya N robots listos SIMULTÁNEAMENTE. Los que ya estén listos (bat_azul >= 0.90) deben recibir la tarea "Espera".
            3. CALCULO DE RECLUTAMIENTO (¡CRITICO!):
            - LISTOS: Robots con bat_azul >= 0.90.
            - CARGANDO: Robots cuya tarea *actual* ya es cargar la bateria azul.
            - FALTAN: N - (LISTOS + CARGANDO).
            4. REGLA DE PACIENCIA: 
            - Si FALTAN > 0: Deben ir a cargar la batería azul SOLO al número exacto de robots que faltan (elige los que tengan la bat_azul más alta).
            - Si FALTAN <= 0: NO MANDES A NADIE MÁS A CARGAR. Mantén a los que están cargando en dicha tarea, a los listos en "Espera" y ten paciencia hasta que los que cargan lleguen a 0.90.
            5. REGLA DE DESPLIEGUE: Cuando LISTOS >= N, elige exactamente a N de esos robots listos y envialos INMEDIATAMENTE en busqueda de luces rojas en la misma decisión.
            6. FORMATO ESTRICTO: El array "decisions" DEBE tener exactamente {num_robots} elementos.
            7. TRIANGULACIÓN FINAL: Al final de la simulación, recibirás todas las coordenadas vistas por los robots. Tu tarea es hacer 'clustering': promedia las coordenadas que estén muy juntas para devolver la posición real de las luces.
            """
        else:
            self.llm_rules = f"""
            Eres el cerebro central de un equipo de {num_robots} robots e-puck en una misión de recolección y mantenimiento.
            Tu objetivo es asignar tareas a cada robot basado en las instrucciones iniciales del usuario y el estado actual de cada robot.
            
            ROBOTS Y BATERÍAS:
            Cada robot tiene dos baterías (azul y roja), con valores de 0.0 a 1.0.
            
            RUTINAS PERMITIDAS (¡PROHIBIDO USAR OTRAS!)
            - "simple_forage": encuentra objetos y los deposita en zona específica (modo patrulla por defecto).
            - "turn_yellow_lights_OFF": apaga las luces amarillas.
            - "load_blue_battery": recarga batería azul.
            - "load_red_battery": recarga batería roja.

            INSTRUCCIONES DEL USUARIO: {user_instructions}

            ESTRUCTURA DE RESPUESTA (JSON):
            {{
            "razonamiento": "Análisis detallado: qué nivel de batería azul y roja  tiene cada robot, cuáles están altos, cuáles bajos, y por qué asignas esas tareas.",
            "memoria_interna": "Diario mental actualizado con contadores e historial.",
            "decisions": ["tarea_robot0", "tarea_robot1", "tarea_robot2"]  // UNA tarea por robot
            }}

            REGLAS DE DECISIÓN:
            1. Asigna rutinas como simple_forage o turn_yellow_lights_OFF a los robots con la bateria azul mas alta.
            2. El array "decisions" DEBE tener exactamente {num_robots} elementos y SOLO puede contener rutinas de la lista PERMITIDA. Si después de aplicar los porcentajes o cantidades solicitadas por el usuario quedan robots sin tarea asignada (por ejemplo, por decimales o redondeos): Bajo ninguna circunstancia puedes asignar una rutina que no haya sido solicitada explícitamente en las instrucciones del usuario solo para rellenar huecos.

            EJEMPLO 1 - Sensor data: Robot0: bat_azul=0.8, Robot1: bat_azul=0.2, Robot2: bat_azul=0.65
            Usuario: "Manda 2 robots a apagar luces y uno a cargar batería"
            Respuesta: {{
                        "razonamiento": "Robot0 tiene bat_azul=0.8 (ALTO) -> ideal para apagar luces. Robot1 tiene bat_azul=0.2 (EMERGENCIA) -> debe cargar. Robot2 tiene bat_azul=0.65 (ALTO) -> puede apagar luces también.",
                        "memoria_interna": "t=100. Instrucciones: 2 luces, 1 carga. Batería azul: [0.8, 0.2, 0.65]. Asignaciones: Robot0->luces (alto), Robot1->carga (emergencia), Robot2->luces (alto). Contadores: luces_apagadas=2, cargas=1.",
                        "decisions": ["turn_yellow_lights_OFF", "load_blue_battery", "turn_yellow_lights_OFF"]
                        }}
            """

        # Lanzar el proceso del cerebro
        self.llm_process = Process(
            target=llm_brain_loop, 
            args=(self.llm_shared_data, self.llm_rules, "http://127.0.0.1:11434", num_robots),
            daemon=True
        )
        self.llm_process.start()

        # Inicializar órdenes
        for i, robot_name in enumerate(self.robots.keys()):
            self.llm_orders[robot_name] = default_decision

        print(f"📁 Guardando experimento en: {os.environ.get('CURRENT_EXP_FOLDER', 'outputs')}")

    def reset(self, seed=None):
        """ Resets the world and all its objects. It also initializes
        the dynamics (positions, orientation, ...) of entities.

        :param int seed: seed to initialize at some known random state. If no seed is used then the 
                argument to be fed must be None
        """
        self.t = 0
        self._is_done = False
        self.physics_engine.paused = self.start_paused
        if self.task_manager is not None:
            self.task_manager.reset(seed=seed)
        if self.virtual_space is not None:
            self.virtual_space.reset(seed=seed)
        #* Reset objects
        for obj in self.hierarchy.values():
            obj.reset(seed=seed)
        #* Reset robot environmental perturbations.
        for group_pert in self.env_perturbations.values():
            for pert in group_pert:
                pert.reset()
        # OJO TO BE IMPROVED
        for robot in self.robots.values():
            robot.static_neighbors = self.lights 
        self.physics_engine.reset()
            
    def connect(self):
        """ Connect to the physics engine. """
        self.physics_engine.connect(self.hierarchy.values())

    def disconnect(self):
        """ Disconnect physics engine. """
        self.physics_engine.disconnect()

    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()

    def run_initializers(self, seed=None):
        """ Executes the initializers of the positions and orientations of each group of world entities.
        As all entities in a group are initialized jointly, initializers are associated to groups.
        
        :param int seed: seed to initialize to a known random state or None if no seed is used.
        """
        if seed is not None:
            np.random.seed(seed)
        for group in self.groups:
            if group in self.initializers.keys():
                group_initializer = self.initializers[group]
                group_elements = self.group_objects(group)
                #* Initialize positions
                if 'positions' in group_initializer:
                    positions = group_initializer['positions']()
                    for pos, obj in zip(positions, group_elements):
                        obj.position = pos
                #* Initialize orientations
                if 'orientations' in group_initializer and group_initializer['orientations'] is not None:
                    orientations = group_initializer['orientations']()
                    for orientation, obj in zip(orientations, group_elements):
                        obj.orientation = orientation
            # else:
            #     for obj in self.group_objects(group):
            #         obj.position = obj.init_position
            #         obj.orientation = obj.init_orientation
        if seed is not None:
            np.random.seed()

    def group_objects(self, group):
        """ List all the objects belonging to a group.

        :param str group: name of the group to be listed.
        :returns: List of WorldObjects belonging to the group.
        """
        return [self.hierarchy[element] for element in self.groups[group]]

    def group_of(self, obj_name):
        """ Get the group to which the object belongs. """
        return [key for key, group_members in self.groups.items() if obj_name in group_members][0]

    def entities(self, obj_type):
        """ Dict with all world objects of some object type (robot, light_source, ...).
        TODO: Not finished: 2D implementation pending
        """
        # if obj_type not in world_objects['3D'].keys():
        #     logging.warning('Wrong world object. Known world objects are: {}'.format(tuple(world_objects)))
        #     return {}
        obj_cls = world_objects['3D'][obj_type]
        return {name : obj for name, obj in self.hierarchy.items()\
                if isinstance(obj, obj_cls)}

    @property
    def robots(self):
        """ Dict with all robots. """
        return self.__robots
        # return {name : obj for name, obj in self.hierarchy.items()\
        #     if issubclass(type(obj), Robot)}

    @property
    def is_done(self):
        if self.task_manager is not  None:
            self._is_done = self.task_manager.is_done
        return self._is_done
    
    @is_done.setter
    def is_done(self, is_done):
        self._is_done = is_done

    @property
    def lights(self):
        """ Dict with all light sources. """
        return {name : obj for name, obj in self.hierarchy.items()\
                if type(obj).__name__ in ['LightSource']}
    @property
    def controllable_objects(self):
        """ Dict with all controllable objects (ie with a controller). """
        return self.__robots
        # return {name : obj for name, obj in self.hierarchy.items()\
        #         if obj.controllable}
    @property
    def uncontrollable_objects(self):
        """ Dict with all uncontrollable objects. """
        return {name : obj for name, obj in self.hierarchy.items()\
                if not obj.controllable}
    @property
    def tangible_objects(self):
        """ Dict with all tangible objects. """
        return {name : obj for name, obj in self.hierarchy.items()\
                if obj.tangible}
    @property
    def intangible_objects(self):
        """ Dict with all intangible objects. """
        return {name : obj for name, obj in self.hierarchy.items()\
                if not obj.tangible}
    @property
    def moving_objects(self):
        """ Dict with all objects with movement capabilities. """
        return {name : obj for name, obj in self.hierarchy.items() if not obj.static}
    @property
    def static_objects(self):
        """ Dict with all static objects. """
        return {name : obj for name, obj in self.hierarchy.items() if obj.static}
    @property
    def luminous_objects(self):
        """ Dict with all luminous objects. """
        return {name : obj for name, obj in self.hierarchy.items() if obj.luminous}

    def set_camera_focus(self, obj, distance):
        """  Sets the focus of the camera of the render engine on the position of an entity.

        :param WorldObject obj: focused entity.
        :param float distance: distance between the entity and the camera.
        """
        self.physics_engine.set_camera_focus(obj.position, distance)

    def measure_time(self):
        import cProfile
        import pstats
        profile = cProfile.Profile()
        res = profile.runctx('self.step()', globals(), locals())
        ps = pstats.Stats(profile)
        ps.print_stats()
        profile.dump_stats('profile.prof')

    def create_animated_layout(self, **kwargs):
        self.animated_layout = AnimatedLayout(**kwargs)


@world_registry(name='flat_world')
class FlatWorld(World):
    """ World class for square arenas of a given height and width. 

    :param float width: width of the square arena in meters.
    :param float height: height of the square arena in meters.
    """
    def __init__(self, *args, **kwargs):
        super(FlatWorld, self).__init__(*args, **kwargs)




@world_registry(name='square_arena')
class SquareArena(World):
    """ World class for square arenas of a given height and width. 

    :param float width: width of the square arena in meters.
    :param float height: height of the square arena in meters.
    """
    def __init__(self, *args, width=10, height=10, **kwargs):
        super(SquareArena, self).__init__(*args, **kwargs)
        self.height = height
        self.width = width
        #* Add world limits
        self.__add_limiting_walls()
        # self.paint_walls()

        

    def __add_limiting_walls(self):
        """ Private method for customizing the size of the limiting walls of the arena. 
        It creates the wall objects individually. They are stored under the group 'side_wall'.
        """
        self.register_entity('wall_side_up', Wall([self.width/2, 0, .1], [0, 0, 0], height=0.1,\
            width=self.width-.1), group='side_wall')
        self.register_entity('wall_side_bottom', Wall([-self.width/2, 0, .1], [0, 0, 0], height=.1,\
            width=self.width-.1), group='side_wall')
        self.register_entity('wall_side_left', Wall([0, self.height/2, .1], [0, 0, 0], height=self.height+.1,\
             width=.1), group='side_wall')
        self.register_entity('wall_side_right', Wall([0, -self.height/2, .1], [0, 0, 0], height=self.height+.1,\
            width=.1), group='side_wall')


@world_registry(name='circular_arena')
class CircularArena(World):
    """ World class for environments with an empty circular arena. 

    :param float radius: radius of the circular wall contraining the arena.
    """
    def __init__(self, *args, radius=5.0, **kwargs):
        super(CircularArena, self).__init__(*args, **kwargs)
        self.radius = radius
        self.__resize_circle_arena()
        self.register_entity('map', Map('circle_arena/circle_arena', np.zeros(3), np.zeros(3)), group='maps')
        
    def __resize_circle_arena(self):
        """ Private method for customizing the radius of the circular arena. 
        It reads the 3D obj mesh file, modifies the vertex positions and saves the file with the changes.
        """
        file = 'mereli/models/maps/circle_arena/circle_arena.obj'
        with open(file, "r") as f:
            lines = f.readlines()
            vertices = []
            for line in lines:
                elems = line.rstrip('\n').split(' ')
                if elems[0] == 'v':
                    vert = np.array(elems[1:]).astype(float)
                    vertices.append(vert)
            vertices = np.vstack(vertices)
            old_rads = np.unique(np.sqrt(vertices[:,0] ** 2 + vertices[:,2] ** 2).round(3))
            # assert len(old_rads) == 2
            scaling = self.radius / old_rads.min()
            vertices[:,[0,2]] *= scaling
        with open(file, "r+") as f:
            lines = f.readlines()
            f.seek(0)
            vert_iter = iter(vertices)
            lines = ['v {} {} {}\n'.format(*tuple(next(vert_iter))) if line.split(' ')[0] == 'v' else line for line in lines ]
            f.writelines(lines)
            f.truncate()


@world_registry(name='custom_world')
class CustomWorld(World):
    """ World class for environments with custom map. The map is defined by means of a previously 
    defined and stored URDF file (with the corresponding obj files). The map file must be stored 
    in the folder 'mereli/models/maps/'.

    :param str model_file: path to the URDF file defining the map. It is relative to 'mereli/models/maps/' 
        and the file extension is not required
    """
    def __init__(self, *args, map_file='simple_map_1/simple_map_1', **kwargs):
        super(CustomWorld, self).__init__(*args, **kwargs)
        self.map_file = map_file
        self.register_entity('map', Map(self.map_file, np.zeros(3), np.zeros(3)), group='maps')




@world_registry(name='arena')
class Arena(World):
    def __init__(self, *args, arena_size=3, map_file='map1.txt', **kwargs):
        super(Arena, self).__init__(*args, **kwargs)
        self.map_file = map_file
        self.arena_size = arena_size
        self.paint_arena_txt()
        

    def step(self):
        states, actions = super().step()
        return states, actions

    def paint_arena_txt(self):
        arr = np.loadtxt(f"mereli/models/maps/{self.map_file}",delimiter=",", dtype=int)
        dX = self.arena_size / arr.shape[0]
        dY = self.arena_size / arr.shape[1]
        # dX = min(dX, dY)
        # dY = min(dX,dY)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if arr[i][j] == 1:
                    posX = i * dX - dX / 2 * arr.shape[0]
                    posY = j * dY - dY / 2 * arr.shape[1]
                    self.register_entity(f'wall_{i}{j}', Wall([posX, posY, 0], [0, 0, 0], height=dX,\
                        width=dY), group='side_wall')
        # __import__('pdb').set_trace()

    def paint_arena_image(self):
        from PIL import Image
        arr = 1 - np.array(Image.open(r"mereli/models/maps/Maze.png"))[:,:, 0] / 255
        # arr = arr[:70,:70]
        __import__('pdb').set_trace()
        dX = self.arena_size / arr.shape[0]
        dY = self.arena_size / arr.shape[1]
        # dX = min(dX, dY)
        # dY = min(dX,dY)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if arr[i][j] == 1:
                    posX = i * dX - dX / 2 * arr.shape[0]
                    posY = j * dY - dY / 2 * arr.shape[1]
                    self.register_entity(f'wall_{i}{j}', Wall([posX, posY, 0], [0, 0, 0], height=dX,\
                        width=dY), group='side_wall')
        # __import__('pdb').set_trace()



class MultiWorldWrapper:
    """ Wrapper class for parallelizing genotype evaluations. 
    
    :param int n_cpu: number of cores (and parallel simulations).
    :param World world: created instance of world or environment to be 
        cloned and parallelized.

    .. todo:: #TODO: Needs to be revisited!
    """
    def __init__(self, n_cpu, world):
        self.n_cpu = n_cpu
        #! Mucho ojo. Son objetos totalmente desacoplados?
        self._worlds = [SquareArena(PybulletEngine(), height=7, width=7) for _ in range(n_cpu + 1)]
        # self._worlds = [copy.deepcopy(world)] * (n_cpu + 1)

    def build_from_dict(self, world_dict, ann_topology=None):
        """Build all the created worlds from the config dicts. """
        for world in self._worlds:
            world.build_from_dict(world_dict, ann_topology=ann_topology)

    @property
    def all(self):
        """ Return all the worlds as a list. 

        :returns: list of World instances of length N_worlds + 1.
        """
        return self._worlds

    @property
    def robots(self):
        """ Return all the robots of the first world. """
        return self._worlds[0].robots

    def get_world(self, idx):
        """ 
        Get world by index. 
        
        :param int idx: index of the world queried within [0, N_worlds]

        :returns: World instance requested.
        """
        return self._worlds[idx]

# class World2D(World):
#     """ World class of 2D bounded arenas. """
#     def __init__(self, *args, **kwargs):
#         physics_engine = Engine2D()
#         super(World2D, self).__init__(physics_engine, *args, **kwargs)
#         self.add_limiting_walls()

#     def add_limiting_walls(self):
#         """ Creates the limiting walls of the 2D arena. """
#         self.register_entity('wall_side_up', Wall([self.width/2, 0], np.pi/2, height=0.5,\
#             width=self.width), group='side_wall')
#         self.register_entity('wall_side_bottom', Wall([-self.width/2,0], np.pi/2, height=0.5,\
#             width=self.width), group='side_wall')
#         self.register_entity('wall_side_left', Wall([0, self.height/2], -np.pi/2, height=self.height-0.5,\
#             width=0.5), group='side_wall')
#         self.register_entity('wall_side_right', Wall([0, -self.height/2], -np.pi/2, height=self.height-0.5,\
#             width=0.5), group='side_wall')

#     def assign_unique_id(self):
#         """ Returns a unique identifier to be assigned to a new entity. """
#         obj_id = np.random.randint(1000)
#         while(len(self.hierarchy) > 0 and obj_id in [obj.id for obj in self.hierarchy.values()]):
#             obj_id = np.random.randint(1, 1000)
#         return obj_id

#     def register_entity(self, name, obj, group=None):
#         """ Adds entity to the world registry. """
#         obj.id = self.assign_unique_id()
#         super().register_entity(name, obj, group=group)

class GymWorld(object):
    def __init__(self, env_name):
        pass
