import os
import json
import glob
import time
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# 1. Configura llave
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# 2. Inicializa el modelo
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 3. Define las reglas del sistema
system_rules = SystemMessage(content="""
Eres el cerebro de un robot e-puck.
Tu objetivo es elegir la rutina técnica correcta basada en los sensores.
Sabiendo que el robot tiene dos baterías (azul y roja), con valores de 0.0 a 1.0, y puede realizar tres rutinas:
 "simple_forage", en la encuentra objetos y los deposita en una zona especifica, y
 "load_blue_battery" y "load_red_battery",  que recargan las respectivas baterías.

REGLAS:
Si una batería es < 0.3, PRIORIZA cargarla
Responde SOLO el nombre de la rutina: [simple_forage, load_red_battery, load_blue_battery].
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
            "bat_roja": round(float(fila['bat_roja']), 2)
        }
    except Exception as e:
        print(f"⚠️ Error leyendo CSV: {e}")
        return None

print(" Cerebro activado. Esperando datos de Mereli...")

while True:
    # A. PERCEPCIÓN: Leer datos reales del hardware 
    robot_state = obtener_sensores_desde_csv()
    
    if robot_state:
        # B. PLANIFICACIÓN: El LLM decide [cite: 22]
        human_data = HumanMessage(content=f"Sensores: {robot_state}")
        
        try:
            response = llm.invoke([system_rules, human_data])
            decision = response.content.strip()
            
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
    time.sleep(5)
        
