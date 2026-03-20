import os
import json
import glob
import time
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# 1. Configuración del modelo local en Calculon
llm = ChatOllama(
    model="gpt-oss:20b",
    temperature=0,
    base_url="http://127.0.0.1:11434"
)

# 2. Variables de control
contador_forage = 0
ultima_rutina = "ninguna"
mision_luces_completada = False # Para que solo se ordene UNA vez en la vida

# 3. Define las reglas del sistema
system_rules = SystemMessage(content="""
Eres el cerebro de un robot e-puck.
Tu objetivo es elegir la rutina técnica correcta basada en los sensores y el historial.
Sabiendo que el robot tiene dos baterías (azul y roja), con valores de 0.0 a 1.0, y puede realizar cuatro rutinas:
 "simple_forage", en la encuentra objetos y los deposita en una zona especifica,
  "turn_yellow_lights_OFF", que apaga las tres luces amarillas, y
 "load_blue_battery" y "load_red_battery",  que recargan las respectivas baterías.

REGLA: Responde SOLO el nombre de la rutina: [simple_forage, load_red_battery, load_blue_battery, turn_yellow_lights_OFF].

JERARQUÍA DE DECISIÓN (Sigue este orden):
1. PERSISTENCIA DE CARGA: Si tu 'Rutina actual' es una de carga (load) y la batería NO ha llegado a 0.9, DEBES seguir respondiendo esa misma rutina de carga.
2. EMERGENCIA: Si no estabas cargando y una batería baja de 0.3, manda cargarla. PRIORIZA carga (Red > Blue).
3. MISIÓN LUCES: Si 'forages' >= 3 y 'mision_luces' es 'No', manda 'turn_yellow_lights_OFF' hasta que 'num_luces' sea 3.
4. FORAGE: En cualquier otro caso, manda 'simple_forage'.

Escribe simple_forage solo si ambas baterías están por encima de 0.3.

### EJEMPLO DE COMPORTAMIENTO (One-shot):
Usuario: "Sensores: {'bat_azul': 0.10, 'bat_roja': 0.90}"
Respuesta: "load_blue_battery"
""")


def obtener_sensores_desde_csv():
    """Busca la última línea del CSV más reciente en la carpeta de Mereli."""
    try:
        # Busca todas las carpetas 'run_' y elige la más nueva
        carpetas = glob.glob("outputs/run_*")
        if not carpetas: return None
        
        ultima_carpeta = max(carpetas, key=os.path.getmtime)
        csv_path = os.path.join(ultima_carpeta, "recorrido_robot.csv")
        
        # Lee la última fila del CSV
        df = pd.read_csv(csv_path)
        if df.empty: return None
        
        fila = df.iloc[-1]
        return {
            "bat_azul": round(float(fila['bat_azul']), 2),
            "bat_roja": round(float(fila['bat_roja']), 2),
            "num_luces": int(fila['num_luces'])
        }
    except Exception as e:
        print(f"⚠️ Error leyendo CSV: {e}")
        return None

print(" Cerebro activado. Esperando datos de Mereli...")

while True:
    # A. PERCEPCIÓN: Leer datos reales del hardware 
    robot_state = obtener_sensores_desde_csv()
    
    if robot_state:
        # B. PLANIFICACIÓN

        #si se ordena apagar las luces y 
        if robot_state['num_luces'] >= 3:
            mision_luces_completada = True # Marcamos que ya se gastó el cartucho de las luces

        contexto_memoria = (
            f"Sensores actuales: {robot_state}. "
            f"Rutina que estás ejecutando AHORA: {ultima_rutina}. "
            f"Historial: Has ordenado forage {contador_forage} veces. "
            f"¿Misión de luces completada?: {'Sí' if mision_luces_completada else 'No'}."
        )

        human_data = HumanMessage(content=contexto_memoria)
        
        try:
            response = llm.invoke([system_rules, human_data])
            decision = response.content.strip()

            #si ha ordenado simple forage y todabia no se han apagado las luces, suma 1
            if decision == "simple_forage" and not mision_luces_completada:
                    contador_forage += 1
            
            ultima_rutina = decision

            
            # C. ACTUACIÓN: Escribir en JSON con TIMESTAMP 
            # El timestamp permite al robot detectar órdenes nuevas
            data_para_robot = {
                "active_routine": decision,
                "timestamp": time.time() 
            }
            
            with open('brain_decision.json', 'w') as f:
                json.dump(data_para_robot, f)
            
            print(f" Sensores: {robot_state} | 🧠 Decisión: {decision}")
            
        except Exception as e:
            print(f"❌ Error en LLM: {e}")
    
    # Frecuencia de realimentación (Punto 17 de tu anteproyecto)
    # Esperamos 5 segundos entre decisiones para no saturar
    time.sleep(3)
        
