import numpy as np
from mereli.controllers import RobotController
from mereli.register import controller_registry, controllers
from mereli.utils import compute_angle

import datetime
import os

@controller_registry(name="navigate")
class NavigateController(RobotController):
    def __init__(self, *args,  **kwargs):
        super(NavigateController, self).__init__(*args, **kwargs)
        self.flag = True 

    def step(self, state, reward=0):
        print("ENTRO EN NAVIGATE")
        self.get_actuator('joint_velocity_actuator').action = np.ones(2) 



@controller_registry(name="load_blue_battery")
class LoadBlueBatteryController(RobotController):
    def __init__(self, *args,  wait_full_load=True, **kwargs):
        super(LoadBlueBatteryController, self).__init__(*args, **kwargs)
        self.flag = False
        self.bat_threshold = 0.7
        self.wait_full_load = True
        self.charging = False

    def step(self, state, reward=0):

        bat_lv_array = self.get_sensor_reading('blue_battery_sensor')
        bat_lv = bat_lv_array[0]

        action = np.array([0,0])
        self.flag = False 
        if self.wait_full_load and bat_lv >= 0.95:
            self.charging = False
        if bat_lv <= self.bat_threshold or self.charging and bat_lv < 0.95:
            self.charging = self.wait_full_load 
            ls_read = self.get_sensor_reading('blue_light_sensor')
            if np.max(ls_read) > 0.9:
                action = np.zeros(2)
                self.flag = True
            elif ls_read[0] * ls_read[7] == 0:
                self.flag = True 
                light_left = np.sum(ls_read[[7,6,5,4]])
                light_right = np.sum(ls_read[[0,1,2,3]])
                if light_right > light_left:
                    action = 0.2*np.array([-1, 1]) 
                else: 
                    action = 0.2*np.array([1, -1]) 
        self.get_actuator('joint_velocity_actuator').action = action

@controller_registry(name="load_red_battery")
class LoadRedBatteryController(RobotController):
    def __init__(self, *args,  wait_full_load=True, **kwargs):
        super(LoadRedBatteryController, self).__init__(*args, **kwargs)
        self.flag = False
        self.bat_threshold = 0.6
        self.wait_full_load = True
        self.charging = False

    def step(self, state, reward=0):
        #print("ENTRO EN LOAD RED BATTERY")
        bat_lv_array = self.get_sensor_reading('red_battery_sensor')
        bat_lv = bat_lv_array[0]

        action = np.array([0,0])
        self.flag = False 
        if self.wait_full_load and bat_lv >= 0.95:
            self.charging = False
        if bat_lv <= self.bat_threshold or self.charging and bat_lv < 0.95:
            self.charging = self.wait_full_load 
            ls_read = self.get_sensor_reading('red_light_sensor')
            if np.max(ls_read) > 0.9:
                action = np.zeros(2)
                self.flag = True
            elif ls_read[0] * ls_read[7] == 0:
                self.flag = True 
                light_left = np.sum(ls_read[[7,6,5,4]])
                light_right = np.sum(ls_read[[0,1,2,3]])
                if light_right > light_left:
                    action = 0.2*np.array([-1, 1]) 
                else: 
                    action = 0.2*np.array([1, -1]) 
        self.get_actuator('joint_velocity_actuator').action = action


@controller_registry(name='astorekeeper')
class AStoreKeeperController(RobotController):
    """
    """
    def __init__(self, *args, routines={'navigate' : 0}, **kwargs):
        super(AStoreKeeperController, self).__init__(*args, **kwargs)
        self.routines = {}
        self.priorities = {}
        self.activations = {}
        for rt, pr in routines.items(): 
            priority = pr if isinstance(pr, int) else pr['priority']
            rt_params = pr.get('params', {}) if isinstance(pr, dict) else {}
            self.priorities[rt] = priority
            self.routines[rt] =  controllers[rt](**rt_params)
            self.activations[rt] = np.zeros(2)

        #Memoria
        self.memory_positions = []      
        self.red_battery_done = False   
        self.battery_was_low = False # Para detectar que la bateria ha llegado por debajo del theshold

        # --- CARPETAS POR FECHA ---
        self.output_dir = os.environ.get("CURRENT_EXP_FOLDER", "outputs")
        self.log_name = os.path.join(self.output_dir, "recorrido_robot.csv")

        if not os.path.exists(self.log_name):
            with open(self.log_name, "w") as f:
                f.write("step,x,y,bat_azul\n")

        print(f"📁 Guardando experimento en: {self.output_dir}")

    def step(self, state, reward=0.0):
        
        # Obtener datos actuales
        t = self.controller_owner.t 
        # La posición es un array [x, y, z]
        x, y = self.controller_owner.position[0], self.controller_owner.position[1]
        # El sensor de batería azul devuelve un array, cogemos el primer valor
        val_azul = state['blue_battery_sensor'][0] 
        
        # 2. Guardar en el CSV cada 10 pasos
        if t % 10 == 0:
            with open(self.log_name, "a") as f:
                f.write(f"{t},{x:.3f},{y:.3f},{val_azul:.3f}\n")


        for k, routine in self.routines.items():
            action = routine.step(state)
            self.activations[k] = self.get_actuator('joint_velocity_actuator').action
        
        off_routine = self.routines.get('turn_yellow_lights_OFF')
        on_routine = self.routines.get('turn_yellow_lights_ON')
        bat_level = self.get_sensor_reading('red_battery_sensor')[0]

        # 1º Guardar posiciones cuando off termine
        if off_routine and not off_routine.flag and len(off_routine.recorded_positions) > 0 and not self.memory_positions:
            self.memory_positions = off_routine.recorded_positions[:] 
            print(f"STOREKEEPER: Fase 1 terminada. Posiciones guardadas: {len(self.memory_positions)}")
        
        # 2º: Fin de carga de batería roja
        # 1. Comprobar si la batería está baja (< 0.65)
        if bat_level < 0.65:
            if not self.battery_was_low:
                print(f"MAIN: Detectada batería baja ({bat_level:.2f}). Esperando ciclo de carga...")
            self.battery_was_low = True

        # 2. Comprobar si se ha cargado al completo (DISPARAMOS)
        if self.battery_was_low and bat_level > 0.95:
            self.red_battery_done = True
            self.battery_was_low = False # Reseteamos para que no se dispare mas
            print(f"MAIN: >>> CICLO COMPLETADO (Estaba baja -> Ahora {bat_level:.2f}). ACTIVANDO FASE FINAL. <<<")
    
        # 3º: Transferir posiciones a rutina on si batería roja cargada
        if on_routine:
            if self.red_battery_done and self.memory_positions:
                if not on_routine.targets:
                    print("MAIN: Transfiriendo objetivos a Rutina ON...")
                    on_routine.set_targets(self.memory_positions) 
                    on_routine.flag = True
            else:
                on_routine.flag = False

        return self.coordinate()
    
    def coordinate(self):
        names = [*self.priorities.keys()]
        names.sort(key=self.priorities.get)
        for k in names:   
            if self.routines[k].flag:
                action_wheels = self.activations[k]
                self.get_actuator('joint_velocity_actuator').action = np.array(action_wheels)
                break

    def reset(self):
        for rt in self.routines.values():
            rt.controller_owner = self.controller_owner
            rt.reset()

@controller_registry(name="simple_forage")
class SimpleForageController(RobotController):
    def __init__(self, *args,  wait_full_load=True, **kwargs):
        super(SimpleForageController, self).__init__(*args, **kwargs)
        self.flag = False

    def step(self, state, reward=0):
        print("ENTRO EN SIMPLE FORAGE")
        mgs_read = self.get_sensor_reading('memory_ground_sensor')
        self.flag = False 
        if mgs_read == 1: # Garbage collected
            ls_read = self.get_sensor_reading('red_light_sensor')
            if ls_read[0] * ls_read[7] == 0:
                self.flag = True
                light_left= np.sum(ls_read[[7,6,5,4]])
                light_right = np.sum(ls_read[[0,1,2,3]])
                action = np.array([0., 0.])
                if light_right > light_left:
                    action = .1*np.array([-1, 1]) 
                else: 
                    action = .1*np.array([1, -1]) 
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
        self.flag = True #so it is the first thing the robot does
        self.lights_off = 0
        self.target_lights = 3
        self.light_threshold = 0.1  #distance to consider the light is reached

        self.recorded_positions = []

        
    def step(self, state, reward=0):

        if self.lights_off >= self.target_lights:
            self.flag = False #desactivar rutina al apagar todas las luces amarillas
            action_wheels = np.array([0., 0.])
            action_light = 0.0


        ls_read = self.get_sensor_reading('yellow_light_sensor')
        max_light = np.max(ls_read)  #Luz más cercana

        action_wheels = np.array([1., 1.]) # Moverse hacia adelante (Exploración)
        action_light = 0.0  # Por defecto: No intentar apagar

        # Si la luz es muy intensa (estamos muy cerca), paramos para asegurar el apagado y registramos.
        if max_light > 0.9: 
            
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
        self.proximity_threshold = 0.15
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
            