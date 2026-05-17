import click
import os
import subprocess
import logging
import time
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
try:
    from mpi4py import MPI
    USE_MPI = True
except:
    USE_MPI = False
from mereli import MultiWorldWrapper
from mereli.register import fitness_functions
from mereli.config_parser import json_parser
from mereli.register import algorithms, worlds, physics_engines
from mereli.globals import global_states

def get_irin_exp(num):
    experiments = ["irin/HelloWorld.json", "irin/TestWheels.json", "irin/TestContact.json", "irin/TestProximity.json", "irin/TestRedLightSensor.json",
     "irin/TestBlueLightSensor.json", "irin/TestGreenLightSensor.json", "irin/TestLED.json", "irin/TestBattery.json", 
     "irin/TestEncoder.json", "irin/ObstacleAvoidance.json", "irin/SubsumptionLightExp.json", "irin/SubsumptionGarbageExp.json", 
     "irin/MotorSchemas1Exp.json", "irin/MotorSchemas2Exp.json", "irin/NeuronEvoAvoidExp.json", "irin/AStorekeeperExp.json", "irin/SubsumptionLucia.json", "irin/AStorekeeperExpLLM", "irin/AStorekeeperExpLLM2", "irin/AStoreKeeperLLMcentral", "irin/ExplorationGroup.json"]
    if num > len(experiments):
        print('Experiment Code does not exist!')
        exit(0)
    return experiments[num]

def print_welcome():
    print("")
    print("WELCOME TO MERELI:\n")
    print("You forgot the configuration file required to properly run an experiment. ") 
    print("In the table below you can find some basic experiments. \n")
    print("In order to run an experiment please execute:\n ")
    print("python main.py -f CODE ")
    print("or")
    print("python main.py -f config_file_path")
    print("")
    print("+------------------------------------------------------------------------------+") 
    print("                          BASIC IRIN EXAMPLES                                   ")
    print("+----------------------+------+------------------------------------------------+") 
    print("| EXPERIMENT           | CODE |      CONFIG FILE LOCATION                      |")
    print("+----------------------+------+------------------------------------------------+") 
    print("| HELLO WORLD          |  0   | mereli/config/irin/HelloWorld.json             |")
    print("+----------------------+------+------------------------------------------------+") 
    print("| TEST WHEELS          |  1   | mereli/config/irin/TestWheels.json             |")
    print("+----------------------+------+------------------------------------------------+") 
    print("| TEST CONTACT         |  2   | mereli/config/irin/TestContact.json            |")
    print("+----------------------+------+------------------------------------------------+") 
    print("| TEST PROXIMITY       |  3   | mereli/config/irin/TestProximity.json          |")
    print("+----------------------+------+------------------------------------------------+") 
    print("| TEST RED             |  4   | mereli/config/irin/TestRedLightSensor.json     |")
    print("| LIGHT SENSOR         |      |                                                |")
    print("+----------------------+------+------------------------------------------------+") 
    print("| TEST BLUE            |  5   | mereli/config/irin/TestBlueLightSensor.json    |")
    print("| LIGHT SENSOR         |      |                                                |")
    print("+----------------------+------+------------------------------------------------+") 
    print("| TEST GREEN           |  6   | mereli/config/irin/TestGreenLightSensor.json   |")
    print("| LIGHT SENSOR         |      |                                                |")
    print("+----------------------+------+------------------------------------------------+") 
    print("| TEST LED             |  7   | mereli/config/irin/TestLED.json                |")
    print("+----------------------+------+------------------------------------------------+") 
    print("| TEST BATTERY         |  8   | mereli/config/irin/TestBattery.json            |")
    print("+----------------------+------+------------------------------------------------+") 
    print("| TEST ENCONDER        |  9   |  mereli/config/irin/TestEncoder.json           |")
    print("+----------------------+------+------------------------------------------------+") 
    print("| BASIC OBSTACLE       | 10   |  mereli/config/irin/ObstacleAvoidance.json     |")
    print("|   AVOIDANCE          |      |                                                |")
    print("+----------------------+------+------------------------------------------------+") 
    print("| SUBSUMPTION LIGHT    | 11   |  mereli/config/irin/SubsumptionLightExp.json   |")
    print("+----------------------+------+------------------------------------------------+") 
    print("| SUBSUMPTION GARBAGE  | 12   |  mereli/config/irin/SubsumptionGarbageExp.json |")
    print("+----------------------+------+------------------------------------------------+") 
    print("| MOTOR SCHEMAS LIGHT  | 13   |  mereli/config/irin/MotorSchemas1Exp.json      |")
    print("+----------------------+------+------------------------------------------------+") 
    print("| MOTOR SCHEMAS        | 14   |  mereli/config/irin/MotorSchemas2Exp.json      |")
    print("|   GARBAGE            |      |                                                |")
    print("+----------------------+------+------------------------------------------------+") 
    print("| EVOLVED OBSTACLE     | 15   |  mereli/config/irin/NeuronEvoAvoidExp.json     |")
    print("|   AVOIDANCE          |      |                                                |")
    print("+----------------------+------+------------------------------------------------+") 
    print("| A STOREKEEPER        | 16   |  mereli/config/irin/AStorekeeperExp.json       |")
    print("+----------------------+------+------------------------------------------------+")
    print("| SUBSUMPTION LUCIA    | 17   |  mereli/config/irin/SubsumptionLucia.json      |")
    print("+----------------------+------+------------------------------------------------+")
    print("| ASTOREKEEPERLLM JSON | 18   |  mereli/config/irin/AStorekeeperExpLLM.json   |")
    print("+----------------------+------+------------------------------------------------+")
    print("| ASTOREKEEPERLLM2 DICT| 19   |  mereli/config/irin/AStorekeeperExpLLM2.json  |")
    print("+----------------------+------+------------------------------------------------+")
    print("| ASTOREKEEPERLLMCENTRAL| 20  |  mereli/config/irin/AStoreKeeperLLMcentral.json|")
    print("+----------------------+------+------------------------------------------------+")
    print("| EXPLORATION GROUP    | 21   |  mereli/config/irin/ExplorationGroup.json     |")
    print("+----------------------+------+------------------------------------------------+")

    print("")

def generate_battery_decision_plot(df, output_path, title, robot_label=None, include_red=True):
    """Pinta baterias y cambios de decision/tarea sobre el tiempo."""
    if df.empty or 'step' not in df.columns or 'bat_azul' not in df.columns:
        return False

    task_col = None
    for candidate in ('decision', 'tarea'):
        if candidate in df.columns:
            task_col = candidate
            break

    plot_df = df.copy()
    plot_df['step'] = pd.to_numeric(plot_df['step'], errors='coerce')
    plot_df['bat_azul'] = pd.to_numeric(plot_df['bat_azul'], errors='coerce')
    if 'bat_roja' in plot_df.columns:
        plot_df['bat_roja'] = pd.to_numeric(plot_df['bat_roja'], errors='coerce')
    plot_df = plot_df.dropna(subset=['step'])
    if plot_df.empty:
        return False

    fig, ax = plt.subplots(figsize=(14, 6))
    label_suffix = f" {robot_label}" if robot_label else ""
    ax.plot(plot_df['step'], plot_df['bat_azul'], color='royalblue', linewidth=2, label=f'Bateria azul{label_suffix}')

    if include_red and 'bat_roja' in plot_df.columns and plot_df['bat_roja'].notna().any():
        ax.plot(plot_df['step'], plot_df['bat_roja'], color='crimson', linewidth=2, label=f'Bateria roja{label_suffix}')

    if task_col is not None:
        task_colors = {
            'simple_forage': 'mediumseagreen',
            'load_red_battery': 'crimson',
            'load_blue_battery': 'royalblue',
            'turn_yellow_lights_OFF': 'goldenrod',
            'go_to_coordenadas': 'darkviolet',
            'navigate': 'mediumseagreen',
            'orient_red_light': 'darkorange',
            'approach_red_light': 'crimson',
            'annotate_red_light_position': 'deeppink',
            'basic_obstacle_avoider': 'black',
            'stop': 'grey',
            'none': 'lightgrey',
        }
        task_series = plot_df[task_col].fillna('none').astype(str)
        change_mask = task_series.ne(task_series.shift(1))
        changes = plot_df.loc[change_mask, ['step']].copy()
        changes[task_col] = task_series.loc[change_mask].values
        changes = changes[changes[task_col] != 'basic_obstacle_avoider']

        ymin, ymax = ax.get_ylim()
        text_y = ymax - (ymax - ymin) * 0.04
        last_label_step = None
        min_label_gap = max((plot_df['step'].max() - plot_df['step'].min()) * 0.035, 1)

        for _, change in changes.iterrows():
            step = change['step']
            task = change[task_col]
            task_color = task_colors.get(task, 'black')
            ax.axvline(step, color=task_color, linestyle='--', linewidth=1.1, alpha=0.65)
            if last_label_step is None or abs(step - last_label_step) >= min_label_gap:
                ax.text(
                    step,
                    text_y,
                    task,
                    rotation=90,
                    va='top',
                    ha='right',
                    fontsize=10,
                    alpha=0.85,
                )
                last_label_step = step

    ax.set_title(title)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Nivel de bateria')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='lower left')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True

def generate_plots_classic(folder, arena_params=None):
    """ Función original para los experimentos antiguos (Ej: 19) """
    import ast
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import re

    csv_path = os.path.join(folder, "recorrido_robot.csv")
    if not os.path.exists(csv_path):
        print(f"⚠️ No se encontró {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # Gráfica Trayectoria
    plt.figure(figsize=(8, 8))
    plt.plot(df['x'], df['y'], color='green', alpha=0.6)
    plt.scatter(df['x'].iloc[0], df['y'].iloc[0], color='red', s=100, label='Inicio', zorder=5)
    plt.scatter(df['x'].iloc[-1], df['y'].iloc[-1], color='blue', s=100, label='Fin', zorder=5)
    luces_path = os.path.join(folder, "luces.csv")
    if os.path.exists(luces_path):
        df_luces = pd.read_csv(luces_path)
        for _, luz in df_luces.iterrows():
            name_lower = luz['name'].lower()
            if 'red' in name_lower:
                plt.scatter(luz['x'], luz['y'], color='red', marker='*', s=400,
                            edgecolor='black', label='Luz roja', zorder=10)
            elif 'blue' in name_lower:
                plt.scatter(luz['x'], luz['y'], color='blue', marker='*', s=400,
                            edgecolor='black', label='Luz azul', zorder=10)
            elif 'yellow' in name_lower:
                plt.scatter(luz['x'], luz['y'], color='gold', marker='*', s=400,
                            edgecolor='black', label='Luz amarilla', zorder=10)

    consola_path = os.path.join(folder, "consola.log")
    if os.path.exists(consola_path):
        found_pattern = re.compile(r"\[found_red_lights\]\s+([^:]+):\s+(\{.*\})")
        with open(consola_path, "r") as f:
            for line in f:
                match = found_pattern.search(line)
                if not match:
                    continue
                try:
                    found_light = ast.literal_eval(match.group(2))
                except (SyntaxError, ValueError):
                    continue
                light_position = found_light.get("light_position")
                if light_position is None:
                    continue
                plt.scatter(
                    light_position[0],
                    light_position[1],
                    color='deeppink',
                    marker='o',
                    s=120,
                    edgecolor='black',
                    label='Luz ubicada por robot',
                    zorder=11,
                )
    plt.title(f'Trayectoria - {os.path.basename(folder)}')
    if arena_params:
        width = arena_params.get('width')
        height = arena_params.get('height')
        if width is not None and height is not None:
            plt.xlim(-width / 2, width / 2)
            plt.ylim(-height / 2, height / 2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--', alpha=0.3)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.savefig(os.path.join(folder, "trayectoria.png"))
    plt.close()

    # Gráfica Batería AZUL
    if 'bat_azul' in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df['step'], df['bat_azul'], color='blue')
        plt.title(f'Batería Azul - {os.path.basename(folder)}')
        plt.savefig(os.path.join(folder, "bateria_azul.png"))
        plt.close()

    # Gráfica Batería ROJA
    if 'bat_roja' in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df['step'], df['bat_roja'], color='red')
        plt.title(f'Batería Roja - {os.path.basename(folder)}')
        plt.savefig(os.path.join(folder, "bateria_roja.png"))
        plt.close()

    generate_battery_decision_plot(
        df,
        os.path.join(folder, "bateria_decisiones_robot_0.png"),
        f'Baterias y decisiones - Robot 0 - {os.path.basename(folder)}',
        robot_label='0',
        include_red=True,
    )

    if 'tarea' in df.columns:
        plt.figure(figsize=(12, 3))
        colores_tareas = {
            'basic_obstacle_avoider': 'tomato',
            'stop': 'lightgrey',
            'navigate': 'mediumseagreen',
            'orient_red_light': 'orange',
            'approach_red_light': 'gold',
            'load_blue_battery': 'royalblue',
            'annotate_red_light_position': 'purple',
            'none': 'lightgrey'
        }
        colores_extra = ['cyan', 'pink', 'brown', 'olive', 'black']
        tareas_legend = set()

        df['cambio_tarea'] = (df['tarea'] != df['tarea'].shift(1)).cumsum()
        bloques = df.groupby(['cambio_tarea', 'tarea']).agg(
            inicio=('step', 'min'),
            fin=('step', 'max')
        ).reset_index()

        for _, bloque in bloques.iterrows():
            tarea = bloque['tarea']
            inicio = bloque['inicio']
            duracion = (bloque['fin'] - inicio) + 1
            if tarea not in colores_tareas:
                colores_tareas[tarea] = colores_extra.pop(0) if colores_extra else 'black'

            label = tarea if tarea not in tareas_legend else None
            plt.barh(0, duracion, left=inicio, height=0.55,
                     color=colores_tareas[tarea], label=label,
                     edgecolor='black', linewidth=0.5)
            tareas_legend.add(tarea)

        plt.yticks([0], ['Robot 0'])
        plt.title(f'Diagrama de Gantt: Tareas - {os.path.basename(folder)}')
        plt.xlabel('Step de Simulación')
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)
        plt.legend(title="Rutinas Ejecutadas", loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(os.path.join(folder, "gantt_tareas.png"))
        plt.close()
        
    print(f"✅ Gráficas clásicas generadas en {folder}")

def generate_plots_centralized(folder, arena_params=None):
    """ Función interna para generar las gráficas tras la simulación """
    import ast
    import glob
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm # Añadimos esta importación para los colores dinámicos
    import re

    # 1. Buscar todos los CSV (0, 1, y 2)
    csv_files = glob.glob(os.path.join(folder, "recorrido_robot_*.csv"))
    if not csv_files:
        print(f"⚠️ No se encontraron archivos CSV en {folder}")
        return

    num_robots = len(csv_files)
    # Si tienes hasta 20 robots usa 'tab20' (colores muy distinguibles), si tienes más, usa 'hsv'
    if num_robots <= 20:
        mapa_colores = cm.get_cmap('tab20', num_robots)
    else:
        mapa_colores = cm.get_cmap('hsv', num_robots)

    try:
        # --- Gráfica Trayectoria Superpuesta ---
        plt.figure(figsize=(8, 8))
        robot_colors = {}
        for i, csv_path in enumerate(csv_files):
            df = pd.read_csv(csv_path)
            if len(df) == 0: continue # Evitar error si Ctrl+C cortó el archivo vacío
            
            robot_id = os.path.basename(csv_path).replace("recorrido_robot_", "").replace(".csv", "")
            color = mapa_colores(i)
            robot_colors[robot_id] = color
            
            plt.plot(df['x'], df['y'], color=color, alpha=0.6, label=f'Robot {robot_id}')
            plt.scatter(df['x'].iloc[0], df['y'].iloc[0], color=color, marker='o', s=100, zorder=5) # Inicio
            plt.scatter(df['x'].iloc[-1], df['y'].iloc[-1], color='red', marker='X', s=100, zorder=5) # Fin

        luces_path = os.path.join(folder, "luces.csv")
        if os.path.exists(luces_path):
            df_luces = pd.read_csv(luces_path)
            for _, luz in df_luces.iterrows():
                name_lower = luz['name'].lower()
                # Pintar luces rojas y azules
                if 'red' in name_lower:
                    l_color = 'red'
                elif 'blue' in name_lower:
                    l_color = 'blue'
                elif 'yellow' in name_lower:
                    l_color = 'gold'
                else:
                    continue
                plt.scatter(luz['x'], luz['y'], color=l_color, marker='*', s=400, edgecolor='black', label=f'Luz {l_color.capitalize()}', zorder=10)

        consola_path = os.path.join(folder, "consola.log")
        if os.path.exists(consola_path):
            found_pattern = re.compile(r"\[found_red_lights\]\s+([^:]+):\s+(\{.*\})")
            with open(consola_path, "r") as f:
                for line in f:
                    match = found_pattern.search(line)
                    if not match:
                        continue
                    try:
                        found_light = ast.literal_eval(match.group(2))
                    except (SyntaxError, ValueError):
                        continue

                    light_position = found_light.get("light_position") or found_light.get("light_pos")
                    if light_position is None:
                        continue

                    robot_name = str(found_light.get("robot_id") or match.group(1).split("_red_light_")[0])
                    robot_id = robot_name.split("_")[-1] if "_" in robot_name else robot_name
                    color = robot_colors.get(robot_id, "deeppink")
                    plt.scatter(
                        light_position[0],
                        light_position[1],
                        color=color,
                        marker='P',
                        s=140,
                        edgecolor='black',
                        label=f'Luz encontrada Robot {robot_id}',
                        zorder=11,
                    )

        plt.title(f'Trayectorias Superpuestas - {os.path.basename(folder)}')
        if arena_params:
            width = arena_params.get('width')
            height = arena_params.get('height')
            if width is not None and height is not None:
                plt.xlim(-width / 2, width / 2)
                plt.ylim(-height / 2, height / 2)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True, linestyle='--', alpha=0.3)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.savefig(os.path.join(folder, "trayectorias.png"))
        plt.close()

        # --- Gráfica Batería AZUL Superpuesta ---
        plt.figure(figsize=(10, 5))
        for i, csv_path in enumerate(csv_files):
            df = pd.read_csv(csv_path)
            if len(df) == 0 or 'bat_azul' not in df.columns: continue
            
            robot_id = os.path.basename(csv_path).replace("recorrido_robot_", "").replace(".csv", "")
            plt.plot(df['step'], df['bat_azul'], color=mapa_colores(i), label=f'Robot {robot_id}')

        plt.title(f'Batería Azul - {os.path.basename(folder)}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(folder, "bateria_azul.png"))
        plt.close()

        # --- Gráfica Batería ROJA Superpuesta ---
        plt.figure(figsize=(10, 5))
        for i, csv_path in enumerate(csv_files):
            df = pd.read_csv(csv_path)
            if len(df) == 0 or 'bat_roja' not in df.columns: continue
            
            robot_id = os.path.basename(csv_path).replace("recorrido_robot_", "").replace(".csv", "")
            plt.plot(df['step'], df['bat_roja'], color=mapa_colores(i), label=f'Robot {robot_id}')

        plt.title(f'Batería Roja - {os.path.basename(folder)}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(folder, "bateria_roja.png"))
        plt.close()

        # --- Gráfica por robot: baterias + tarea asignada ---
        for csv_path in sorted(csv_files):
            df = pd.read_csv(csv_path)
            if len(df) == 0:
                continue

            robot_id = os.path.basename(csv_path).replace("recorrido_robot_", "").replace(".csv", "")
            include_red = 'bat_roja' in df.columns
            generate_battery_decision_plot(
                df,
                os.path.join(folder, f"bateria_decisiones_robot_{robot_id}.png"),
                f'Baterias y tareas - Robot {robot_id} - {os.path.basename(folder)}',
                robot_label=robot_id,
                include_red=include_red,
            )

        # EVOLUCIÓN DE TAREAS ---
        plt.figure(figsize=(12, 4))
        
        # Diccionario de colores fijos para que tenga sentido visual
        colores_tareas = {
            'simple_forage': 'mediumseagreen',    # Verde para patrullar/buscar
            'load_red_battery': 'tomato',         # Rojo para la batería roja
            'load_blue_battery': 'royalblue',     # Azul para la batería azul
            'turn_yellow_lights_OFF': 'gold',     # Amarillo para las luces
            'go_to_coordenadas': 'darkviolet',
            'navigate': 'mediumseagreen',
            'orient_red_light': 'darkorange',
            'approach_red_light': 'crimson',
            'annotate_red_light_position': 'deeppink',
            'basic_obstacle_avoider': 'black',
            'stop': 'lightgrey',
            'none': 'lightgrey'
        }
        colores_extra = ['purple', 'orange', 'cyan', 'pink', 'brown'] # Por si hay tareas nuevas
        
        tareas_legend = set()
        nombres_robots = []
        
        # Ordenamos los archivos para que salgan Robot 0, 1, 2 en orden
        for i, csv_path in enumerate(sorted(csv_files)): 
            df = pd.read_csv(csv_path)
            if len(df) == 0 or 'tarea' not in df.columns: continue
            
            robot_id = os.path.basename(csv_path).replace("recorrido_robot_", "").replace(".csv", "")
            nombres_robots.append(f"Robot {robot_id}")
            y_pos = i  # Altura en el eje Y para este robot
            
            # Agrupar tiempos donde la tarea no cambia para dibujar un solo bloque
            df['cambio_tarea'] = (df['tarea'] != df['tarea'].shift(1)).cumsum()
            bloques = df.groupby(['cambio_tarea', 'tarea']).agg(
                inicio=('step', 'min'),
                fin=('step', 'max')
            ).reset_index()
            
            for _, bloque in bloques.iterrows():
                tarea = bloque['tarea']
                inicio = bloque['inicio']
                # Le sumamos 1 al fin para que los bloques se toquen perfectamente
                duracion = (bloque['fin'] - inicio) + 1 
                
                # Asignar color
                if tarea not in colores_tareas:
                    colores_tareas[tarea] = colores_extra.pop(0) if colores_extra else 'black'
                
                color_barra = colores_tareas[tarea]
                
                # Dibujar el rectángulo. Solo le ponemos 'label' la primera vez para la leyenda
                if tarea not in tareas_legend:
                    plt.barh(y_pos, duracion, left=inicio, height=0.6, color=color_barra, 
                             label=tarea, edgecolor='black', linewidth=0.5)
                    tareas_legend.add(tarea)
                else:
                    plt.barh(y_pos, duracion, left=inicio, height=0.6, color=color_barra, 
                             edgecolor='black', linewidth=0.5)

        # Configurar ejes y aspecto
        plt.yticks(range(len(nombres_robots)), nombres_robots)
        plt.title(f'Diagrama de Gantt: Asignación de Tareas - {os.path.basename(folder)}')
        plt.xlabel('Step de Simulación')
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        # Leyenda bonita fuera de la gráfica
        plt.legend(title="Rutinas Ejecutadas", loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(os.path.join(folder, "gantt_tareas.png"))
        plt.close()

        # Forzar el print a la consola original para que lo veas sí o sí
        import sys
        sys.__stdout__.write(f"✅ Gráficas generadas automáticamente en {folder}\n")

    except Exception as e:
        import sys
        sys.__stdout__.write(f"❌ Error al pintar (probablemente por Ctrl+C a medias): {e}\n")

@click.command()
@click.option('-R', '--render', default=False, is_flag=True, help='Execute in render mode.')
@click.option('-d', '--debug', default=False, is_flag=True,  help='Execute in debug mode.')
@click.option('-r', '--resume', default=False, is_flag=True,\
        help='Resume optimization stored in the checkpoint settled in the JSON config.')
@click.option('-e', '--eval', default=False, is_flag=True, \
        help='Execute in eval mode. No optimization will be carried out.')
@click.option('-v', '--verbose', default=False, is_flag=True,\
        help='Execute in verbose mode (info msgs enabled).')
@click.option('-l', '--log', default=False, is_flag=True, help='Log data into a file.')
@click.option('-n', '--ncpu', default=1, help='Number of CPU cores.')
@click.option('-f', '--cfg', default=None, help='Name of the JSON config. file.')
@click.option('-i', '--interactive', default=False,is_flag=True, help='Run in interactive mode.')
def main(render, resume, cfg, debug, eval, verbose, log, interactive, ncpu):
    if cfg is None:
       print_welcome() 
       exit(0)
    # if interactive:
    #     process = subprocess.Popen(["panel", "serve", "--port" , "8086", 'mereli/dashboard/dashboard.py'])
    #* Set globals
    if len(cfg) <=3: # Is an exp code
        if not render and int(cfg) != 11:
            render = True
        cfg = get_irin_exp(int(cfg))
    global_states.set_states(render=render, eval=eval, debug=debug, log=log, info=verbose, interactive=interactive)
    
    #* Parse JSON
    cfg_dict = json_parser(cfg)
    
    # if log:
    #     logs_folder = cfg_dict.get('logging', {}).get('file', cfg)
    #     logs_path = os.path.join(os.getcwd(), 'mereli', 'logs', logs_folder)
    #     if not os.path.isdir(logs_path):
    #         os.mkdir(logs_path)
    #     now = datetime.now()
    #     logs_path = os.path.join(logs_path, logs_folder + now.strftime("_%d-%m-%Y_%H:%M:%S")) 
    #     global_states.set_data_logging(logs_path)

    # Set loggings
    # if log:
    #     logs_folder = cfg_dict.get('logging', {}).get('file', cfg)
    #     logs_path = os.path.join(os.getcwd(), 'mereli', 'logs', logs_folder)
    #     if not os.path.isdir(logs_path):
    #         os.mkdir(logs_path)

    # __import__('pdb').set_trace()

    # --- MODIFICACIÓN: Crear carpeta de experimento ---
    now = datetime.now().strftime("%m%d_%H%M%S")
    exp_folder = os.path.join("outputs", f"run_{now}")
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)
    
    # Guardamos la ruta en global_states o una variable para que el controlador la use
    # Por ahora, la pasaremos de forma sencilla
    os.environ["CURRENT_EXP_FOLDER"] = exp_folder 
    print(f"📁 Iniciando experimento en: {exp_folder}")

    # --- Redirigir consola a un archivo log ---
    import sys
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w")
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            self.terminal.flush()
            self.log.flush()

        def fileno(self):
            return self.terminal.fileno()

    sys.stdout = Logger(os.path.join(exp_folder, "consola.log"))

    #* Create World
    physics_engine = physics_engines[cfg_dict['world'].get('engine', 'pybullet')](
                        dt=cfg_dict['world'].get('physics_dt', 0.02), 
                        T_control=cfg_dict['world'].get('T_control', 0.1))
    world_cls = worlds[cfg_dict['world'].get('name', 'square_arena')]
    arena_params = cfg_dict['world'].get('arena_params', {})
    world = world_cls(physics_engine, **arena_params)
        
    # --- PREGUNTAR AL USUARIO PARA LOS EXPERIMENTOS CON LLM CENTRAL ---
    instrucciones = "No hay instrucciones específicas. Asigna tareas por defecto."
    if "AStoreKeeperLLMcentral" in cfg or "ExplorationGroup" in cfg:
        print("\n🧠 [CEREBRO CENTRAL]: Ingresa instrucciones iniciales para el LLM (ej: 'Manda 2 robots a cargar batería...'): ")
        entrada = input().strip()
        if entrada != "":
            instrucciones = entrada
        print(f"🧠 [CEREBRO CENTRAL]: Instrucciones enviadas a la IA: {instrucciones}\n")
    
    # Pasamos las instrucciones al mundo al construirlo
    world.build_from_dict(cfg_dict['world'], ann_topology=cfg_dict.get('topology', {}), user_instructions=instrucciones)
    
    if log:
        world.config_data_logger(cfg_dict['logging']['data'])
        world.data_logger.set_log_file(cfg_dict.get('logging', {}).get('file', cfg))
    if render:
        simulation_config = cfg_dict.get('simulation', {})
        world.start_paused = simulation_config.get('start_paused', False)
    #     if 'animated_layout' in simulation_config:
    #         anim_config = simulation_config.get('animated_layout')
    #         world.create_animated_layout()
    #         world.animated_layout.add_plots(anim_config.get('plots'), grid=anim_config['grid'])
    #         world.animated_layout.initialize()

    # Create virtual space (if any)
    if 'virtual_space' in cfg_dict:

        # is_neural_ctlr = cfg_dict['virtual_space']['controller']['name'] == 'neural_controller'
        topology_name = cfg_dict['virtual_space'].get('controller',{}).get('topology')
        world.create_virtual_space(**cfg_dict['virtual_space'], topology=cfg_dict.get('topology',{}).get(topology_name))

    # import copy
    # world2 = copy.deepcopy(world)
    # world.connect()
    # world2.connect()
    # print(world.physics_engine.client)
    # print(world2.physics_engine.client)
    # import pdb; pdb.set_trace()

    if cfg_dict.get('algorithm', False) and len(cfg_dict['algorithm']):
        alg_config = cfg_dict['algorithm']
        if alg_config['name'] == 'multi_EA':
            opt_alg = algorithms['multi_EA'](world, alg_config['generations'], alg_config['population_size'], None, 
                        num_evaluations=alg_config['num_evaluations'],
                        resume=resume, fitness_fn=alg_config['fitness_function'], 
                        novelty_search=alg_config.get('novelty_search'), 
                        checkpoint_name=cfg_dict["checkpoint_file"])
            for i, alg_cfg in enumerate(alg_config['algs'].values()):
                alg_cls = algorithms[alg_cfg['name']]
                alg = alg_cls(world, alg_config['generations'], 
                        alg_config['population_size'], alg_cfg['targets'],
                        resume=resume, fitness_fn=alg_config['fitness_function'], 
                        checkpoint_name=cfg_dict["checkpoint_file"]+'_'+str(i+1), **alg_cfg["alg_params"])
                alg.initialize(alg_cfg['gene_info'], cfg_dict['topology'])
                opt_alg.add_algorithm(alg)
        else:
            algorithm_cls = algorithms[cfg_dict['algorithm']['name']]
            opt_alg = algorithm_cls(world, alg_config['generations'], 
                        alg_config['population_size'], alg_config['targets'],
                        num_evaluations=alg_config['num_evaluations'],
                        resume=resume, fitness_fn=alg_config['fitness_function'], 
                        novelty_search=alg_config.get('novelty_search'), 
                        checkpoint_name=cfg_dict["checkpoint_file"], **alg_config["alg_params"])
            # opt_alg.create_world(cfg_dict['world'], ann_config=cfg_dict['topology'])
            opt_alg.initialize(alg_config["gene_info"], cfg_dict['topology'])
        #* Run GA
        if render:
            simulation_config = cfg_dict.get('simulation', {})
            opt_alg.evaluator.world.start_paused = simulation_config.get('start_paused', False)
            if 'animated_layout' in simulation_config and simulation_config['animated_layout'].get('enabled', False):
                anim_config = simulation_config.get('animated_layout')
                opt_alg.evaluator.world.create_animated_layout()
                opt_alg.evaluator.world.animated_layout.add_plots(anim_config.get('plots'), grid=anim_config['grid'])
                opt_alg.evaluator.world.animated_layout.initialize(world)
        if not eval:
            opt_alg.run()
        else:
            opt_alg.validate()
    else: #* Non-optimizable simulation
        simulation_config = cfg_dict.get('simulation', {})
        seed = simulation_config.get('seed', None)
        world.connect()
        print('Connected!')
        timesteps = simulation_config.get('timesteps', 10000)
        trials = simulation_config.get('trials', 1)
        if render:
            world.start_paused = simulation_config.get('start_paused', False)
            if 'camera_options' in simulation_config:
                world.physics_engine.set_camera_options(**simulation_config['camera_options'])
            if 'animated_layout' in simulation_config and simulation_config['animated_layout'].get('enabled', False):
                anim_config = simulation_config.get('animated_layout')
                world.create_animated_layout(figsize=anim_config.get('figsize'))
                world.animated_layout.add_plots(anim_config.get('plots'), grid=anim_config['grid'])
                world.animated_layout.initialize(world)
        np.random.seed(seed)
        # Interacción inicial con LLM ya se hizo antes
        try: 
            for tr in range(trials):
                world.reset()
                try:
                    with open(os.path.join(exp_folder, "luces.csv"), "w") as f:
                        f.write("name,x,y\n")
                        for light_name, light_obj in world.hierarchy.items():
                            if 'light' in light_name.lower() and hasattr(light_obj, 'position'):
                                f.write(f"{light_name},{light_obj.position[0]:.3f},{light_obj.position[1]:.3f}\n")
                except Exception as e:
                    print(f"Error guardando luces: {e}")

                if "AStoreKeeperLLMcentral" in cfg:
                    # --- NUEVO: GUARDAR POSICIÓN DE LAS LUCES TRAS INICIALIZAR ---
                    try:
                        with open(os.path.join(exp_folder, "luces.csv"), "w") as f:
                            f.write("name,x,y\n")
                            # Buscamos en world.hierarchy, que es donde están todos los objetos
                            for light_name, light_obj in world.hierarchy.items():
                                f.write(f"{light_name},{light_obj.position[0]:.3f},{light_obj.position[1]:.3f}\n")
                    except Exception as e:
                        print(f"Error guardando luces: {e}")
                     #2. Mandar estado inicial a la IA y esperar su respuesta antes de arrancar la simulación
                    if world.llm_shared_data is not None:
                        print("\n⏳ [SISTEMA]: Sincronizando con el Cerebro Central...")
                       
                        # --- GENERAMOS EL CONTEXTO LEYENDO TUS CLASES REALES ---
                        contexto_ini = f"t=0 (INICIO DE MISIÓN CON DATOS REALES)\n"
                        nombres_ordenados = sorted(world.robots.keys())
                        for i, name in enumerate(nombres_ordenados):
                            rob = world.robots[name]
                           
                            b_azul = 0.0
                            b_roja = 0.0
                            # Magia pura: Leemos las variables exactas que me has pasado
                            if getattr(rob, 'battery_enabled', False) and rob.battery is not None:
                                for color_bat, nivel in zip(rob.battery_colors, rob.battery.level):
                                    if color_bat == 'blue': b_azul = nivel
                                    if color_bat == 'red': b_roja = nivel
                           
                            contexto_ini += f"Robot {i} ({name}): bat_azul={b_azul:.2f}, bat_roja={b_roja:.2f}, luces_apagadas=0\n"
                       
                        # Enviamos el contexto y "despertamos" a la IA
                        world.llm_shared_data['contexto'] = contexto_ini
                        world.llm_shared_data['sensor_ts'] = 1
                       
                        print("... IA pensando con datos reales ...")
                       
                        # --- BUCLE DE ESPERA SEGURO ---
                        while world.llm_shared_data.get('decision_ts', 0) == 0:
                            time.sleep(0.5)
                       
                        # --- APLICAMOS LAS DECISIONES AL MUNDO ---
                        decisions = world.llm_shared_data.get('decisions', [])
                        for i, robot_name in enumerate(nombres_ordenados):
                            if i < len(decisions):
                                world.llm_orders[robot_name] = decisions[i]
                               
                        print("🚀 [SISTEMA]: Órdenes recibidas. ¡Arrancando simulación!\n") 
                
                t0 = time.time()
                while (world.t < timesteps):
                    if world.t == timesteps - 1:
                        world.is_done = True
                    state, action = world.step()
                time_elapsed = time.time() - t0 
                # print(np.hstack([rob.position[:2] for rob in world.robots.values()]))
                print(f'Simulation of trial {tr} ended in {time_elapsed} after {timesteps} cycles. ')
        except KeyboardInterrupt:
            print("\n🛑 Simulación interrumpida.")
        finally:
            # EL MAIN DECIDE QUÉ GRÁFICA USAR SEGÚN EL EXPERIMENTO
            if "AStoreKeeperLLMcentral" in cfg or "ExplorationGroup" in cfg:
                generate_plots_centralized(exp_folder, arena_params=arena_params)
            else:
                generate_plots_classic(exp_folder, arena_params=arena_params)
if __name__ == "__main__":
    main()

