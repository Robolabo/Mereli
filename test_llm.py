import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# 1. Configura tu llave
os.environ["GOOGLE_API_KEY"] = "AIzaSyCS_dsv-xaAACAm9YSOQkLFwsEOF_MpLrY  "

# 2. Inicializa el modelo
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


# 3. Simula un estado del robot
robot_state = {
    "bat_azul": 0.90,
    "bat_roja": 0.30,
    "objeto_encontrado": False
}

# 4. Prompt para decidir
prompt = f"""
Eres el cerebro de un robot. Tu estado actual es: {robot_state}.
Tus opciones de rutinas son: "simple_forage", "load_blue_battery", "load_red_battery".
Responde ÚNICAMENTE con el nombre de la rutina elegida.
"""

# 5. Llamada y escritura en JSON
response = llm.invoke([HumanMessage(content=prompt)])
decision = response.content.strip()

with open('brain_decision.json', 'w') as f:
    json.dump({"active_routine": decision}, f)

print(f"Cerebro decidió: {decision}")