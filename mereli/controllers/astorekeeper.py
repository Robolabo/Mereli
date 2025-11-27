import numpy as np
from mereli.controllers import RobotController
from mereli.register import controller_registry, controllers
from mereli.utils import compute_angle


@controller_registry(name="navigate")
class NavigateController(RobotController):
    def __init__(self, *args,  **kwargs):
        super(NavigateController, self).__init__(*args, **kwargs)
        self.flag = True 

    def step(self, state, reward=0):
        print("ENTRO EN NAVIGATE")
        self.get_actuator('joint_velocity_actuator').action = np.ones(2) 



@controller_registry(name="load_blue_battery")
class LoadBatteryController(RobotController):
    def __init__(self, *args,  wait_full_load=True, **kwargs):
        super(LoadBatteryController, self).__init__(*args, **kwargs)
        self.flag = False
        self.bat_threshold = 0.5
        self.wait_full_load = True
        self.charging = False

    def step(self, state, reward=0):
        print("ENTRO EN LOAD BLUE BATTERY")
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

    def step(self, state, reward=0.0):
        for k, routine in self.routines.items():
            action = routine.step(state)
            self.activations[k] = self.get_actuator('joint_velocity_actuator').action
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