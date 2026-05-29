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
        if 'decision' in plot_df.columns and 'decision_ts' in plot_df.columns:
            task_col = 'decision'
            task_series = plot_df[task_col].fillna('none').astype(str)
            decision_ts = pd.to_numeric(plot_df['decision_ts'], errors='coerce').fillna(-1)
            change_mask = decision_ts.gt(0) & decision_ts.ne(decision_ts.shift(1))
        else:
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

def generate_skill_usage(folder):
    """Calcula el porcentaje de timesteps que cada robot pasa en cada skill."""
    import glob
    import os
    import pandas as pd

    csv_files = sorted(glob.glob(os.path.join(folder, "recorrido_robot_*.csv")))
    if not csv_files:
        single_csv = os.path.join(folder, "recorrido_robot.csv")
        csv_files = [single_csv] if os.path.exists(single_csv) else []

    rows = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        if df.empty or 'step' not in df.columns:
            continue

        task_col = 'tarea' if 'tarea' in df.columns else 'decision' if 'decision' in df.columns else None
        if task_col is None:
            continue

        if 'robot' in df.columns and df['robot'].notna().any():
            robot_id = str(df['robot'].dropna().iloc[0])
        elif os.path.basename(csv_path).startswith("recorrido_robot_"):
            robot_id = os.path.basename(csv_path).replace("recorrido_robot_", "").replace(".csv", "")
        else:
            robot_id = "robot_0"

        work_df = df[['step', task_col]].copy()
        work_df['step'] = pd.to_numeric(work_df['step'], errors='coerce')
        work_df = work_df.dropna(subset=['step']).sort_values('step')
        if work_df.empty:
            continue

        steps = work_df['step'].to_numpy()
        deltas = pd.Series(steps).diff().shift(-1)
        deltas.index = work_df.index
        positive_deltas = deltas[deltas > 0]
        default_delta = positive_deltas.median() if not positive_deltas.empty else 1
        work_df['duration_steps'] = deltas.fillna(default_delta).clip(lower=0)
        total_steps = float(work_df['duration_steps'].sum())
        if total_steps <= 0:
            continue

        grouped = work_df.groupby(task_col)['duration_steps'].sum().reset_index()
        for _, row in grouped.iterrows():
            duration = float(row['duration_steps'])
            rows.append({
                'robot': robot_id,
                'skill': row[task_col],
                'steps': duration,
                'total_steps': total_steps,
                'percentage': 100.0 * duration / total_steps,
            })

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(folder, "skill_usage.csv"), index=False)

def generate_llm_timing_plots(folder, separate=False):
    """Genera graficas de latencia y steps ciegos a partir de llm_tiempos.csv."""
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    csv_path = os.path.join(folder, "llm_tiempos.csv")
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    for col in ['request_step', 'latency_s', 'blind_steps']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['request_step'])
    if df.empty:
        return

    df['serie'] = df['scope'].astype(str) + ':' + df['robot'].astype(str)

    if separate:
        plt.figure(figsize=(12, 5))
        for serie, serie_df in df.groupby('serie'):
            serie_df = serie_df.sort_values('request_step')
            plt.plot(serie_df['request_step'], serie_df['latency_s'], marker='o', label=serie)
        plt.title(f'Latencia LLM - {os.path.basename(folder)}')
        plt.xlabel('Step de solicitud')
        plt.ylabel('Tiempo de respuesta real (s)')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(folder, "llm_latencias.png"), dpi=150)
        plt.close()

        plt.figure(figsize=(12, 5))
        for serie, serie_df in df.groupby('serie'):
            serie_df = serie_df.sort_values('request_step')
            plt.plot(serie_df['request_step'], serie_df['blind_steps'], marker='o', label=serie)
        plt.title(f'Steps ciegos esperando LLM - {os.path.basename(folder)}')
        plt.xlabel('Step de solicitud')
        plt.ylabel('Steps hasta aplicar respuesta')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(folder, "llm_blind_steps.png"), dpi=150)
        plt.close()
        return

    fig, ax_left = plt.subplots(figsize=(12, 5))
    ax_right = ax_left.twinx()
    handles = []
    labels = []

    for serie, serie_df in df.groupby('serie'):
        serie_df = serie_df.sort_values('request_step')
        blind_line, = ax_left.plot(
            serie_df['request_step'],
            serie_df['blind_steps'],
            marker='o',
            linestyle='-',
            label=f'{serie} blind steps',
        )
        latency_line, = ax_right.plot(
            serie_df['request_step'],
            serie_df['latency_s'],
            marker='s',
            linestyle='--',
            color=blind_line.get_color(),
            label=f'{serie} latencia',
        )
        handles.extend([blind_line, latency_line])
        labels.extend([blind_line.get_label(), latency_line.get_label()])

    ax_left.set_title(f'Asincronia y latencia LLM - {os.path.basename(folder)}')
    ax_left.set_xlabel('Step de solicitud')
    ax_left.set_ylabel('Blind steps')
    ax_right.set_ylabel('Latencia real (s)')
    ax_left.grid(True, linestyle='--', alpha=0.3)
    ax_left.legend(handles, labels, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(folder, "llm_tiempos_async.png"), dpi=150)
    plt.close(fig)

def generate_skill_usage_pies(folder):
    """Genera graficos de queso con el porcentaje de uso de cada skill."""
    import os
    import re
    import pandas as pd
    import matplotlib.pyplot as plt

    csv_path = os.path.join(folder, "skill_usage.csv")
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    if df.empty or 'robot' not in df.columns or 'skill' not in df.columns or 'steps' not in df.columns:
        return

    df['steps'] = pd.to_numeric(df['steps'], errors='coerce')
    df = df.dropna(subset=['steps'])
    df = df[df['steps'] > 0]
    if df.empty:
        return

    skill_colors = {
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
        'stop': 'lightgrey',
        'none': 'lightgrey',
    }

    def safe_name(value):
        return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value)).strip('_') or 'robot'

    def plot_pie(plot_df, title, output_name):
        grouped = plot_df.groupby('skill')['steps'].sum().sort_values(ascending=False)
        if grouped.empty:
            return
        colors = [skill_colors.get(skill, None) for skill in grouped.index]
        plt.figure(figsize=(8, 8))
        plt.pie(
            grouped.values,
            labels=grouped.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
        )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(folder, output_name), dpi=150)
        plt.close()

    for robot, robot_df in df.groupby('robot'):
        plot_pie(
            robot_df,
            f'Uso de skills - {robot} - {os.path.basename(folder)}',
            f"skill_usage_pie_{safe_name(robot)}.png",
        )

    if df['robot'].nunique() > 1:
        plot_pie(
            df,
            f'Uso de skills total - {os.path.basename(folder)}',
            "skill_usage_pie_total.png",
        )


def generate_light_discovery_metrics(folder):
    """Calcula progreso de descubrimiento de luces rojas frente al tiempo de mision."""
    import ast
    import glob
    import os
    import re
    import pandas as pd

    consola_path = os.path.join(folder, "consola.log")
    luces_path = os.path.join(folder, "luces.csv")
    if not os.path.exists(consola_path) or not os.path.exists(luces_path):
        return

    try:
        lights_df = pd.read_csv(luces_path)
    except Exception:
        return
    if lights_df.empty or 'name' not in lights_df.columns:
        return

    red_lights = sorted(lights_df[lights_df['name'].astype(str).str.lower().str.contains('red')]['name'].astype(str).tolist())
    total_red_lights = len(red_lights)
    if total_red_lights == 0:
        return

    total_steps = None
    end_pattern = re.compile(r"after\s+(\d+)\s+cycles")
    found_pattern = re.compile(r"\[found_red_lights\]\s+([^:]+):\s+(\{.*\})")
    found_rows = []

    with open(consola_path, "r") as f:
        for line in f:
            end_match = end_pattern.search(line)
            if end_match:
                total_steps = int(end_match.group(1))

            match = found_pattern.search(line)
            if not match:
                continue
            try:
                info = ast.literal_eval(match.group(2))
            except (SyntaxError, ValueError):
                continue

            step = info.get('step')
            light_id = info.get('light_id')
            robot = info.get('robot_id') or match.group(1).split('_red_light_')[0]
            if step is None or light_id is None or robot is None:
                continue

            light_name = f"light_red_{light_id}"
            if light_name not in red_lights:
                # En algunos logs el id interno no coincide con el indice del nombre real.
                light_name = str(light_id)

            try:
                step = int(step)
            except (TypeError, ValueError):
                continue

            found_rows.append({
                'step': step,
                'robot': str(robot),
                'light_id': str(light_id),
                'light_name': light_name,
            })

    csv_files = sorted(glob.glob(os.path.join(folder, "recorrido_robot_*.csv")))
    if total_steps is None:
        max_steps = []
        for csv_path in csv_files:
            try:
                df_steps = pd.read_csv(csv_path, usecols=['step'])
            except Exception:
                continue
            if not df_steps.empty:
                max_steps.append(pd.to_numeric(df_steps['step'], errors='coerce').max())
        total_steps = int(max(max_steps)) if max_steps else 0

    if not found_rows:
        summary = pd.DataFrame([{
            'scope': 'global',
            'robot': 'all_active',
            'red_lights_total': total_red_lights,
            'active_robots': 0,
            'discoveries': 0,
            'expected_discoveries': 0,
            'unique_lights_discovered': 0,
            'completed_all_lights': False,
            'last_discovery_step': '',
            'total_mission_steps': total_steps,
            'effective_time_steps': total_steps,
            'discoveries_per_1000_steps': 0.0,
            'unique_lights_per_1000_steps': 0.0,
        }])
        summary.to_csv(os.path.join(folder, "light_discovery_metrics.csv"), index=False)
        with open(os.path.join(folder, "light_discovery_metrics_table.txt"), "w") as f:
            f.write("LIGHT DISCOVERY METRICS\n")
            f.write("No red lights were registered by robots.\n")
        return

    found_df = pd.DataFrame(found_rows).drop_duplicates(subset=['robot', 'light_id']).sort_values(['step', 'robot'])
    active_robots = sorted(found_df['robot'].unique().tolist())
    active_robot_count = len(active_robots)
    expected_discoveries = total_red_lights * active_robot_count
    discoveries = len(found_df)
    unique_lights_discovered = found_df['light_id'].nunique()

    robot_counts = found_df.groupby('robot')['light_id'].nunique()
    all_active_complete = bool(active_robot_count > 0 and (robot_counts >= total_red_lights).all())
    last_discovery_step = int(found_df['step'].max())
    effective_global_time = last_discovery_step if all_active_complete else total_steps
    if effective_global_time <= 0:
        effective_global_time = total_steps or 1

    rows = [{
        'scope': 'global',
        'robot': 'all_active',
        'red_lights_total': total_red_lights,
        'active_robots': active_robot_count,
        'discoveries': discoveries,
        'expected_discoveries': expected_discoveries,
        'unique_lights_discovered': unique_lights_discovered,
        'completed_all_lights': all_active_complete,
        'last_discovery_step': last_discovery_step,
        'total_mission_steps': total_steps,
        'effective_time_steps': effective_global_time,
        'discoveries_per_1000_steps': 1000.0 * discoveries / effective_global_time,
        'unique_lights_per_1000_steps': 1000.0 * unique_lights_discovered / effective_global_time,
    }]

    for robot in active_robots:
        robot_df = found_df[found_df['robot'] == robot]
        robot_discoveries = int(robot_df['light_id'].nunique())
        robot_complete = robot_discoveries >= total_red_lights
        robot_last_step = int(robot_df['step'].max())
        robot_effective_time = robot_last_step if robot_complete else total_steps
        if robot_effective_time <= 0:
            robot_effective_time = total_steps or 1
        rows.append({
            'scope': 'robot',
            'robot': robot,
            'red_lights_total': total_red_lights,
            'active_robots': active_robot_count,
            'discoveries': robot_discoveries,
            'expected_discoveries': total_red_lights,
            'unique_lights_discovered': robot_discoveries,
            'completed_all_lights': robot_complete,
            'last_discovery_step': robot_last_step,
            'total_mission_steps': total_steps,
            'effective_time_steps': robot_effective_time,
            'discoveries_per_1000_steps': 1000.0 * robot_discoveries / robot_effective_time,
            'unique_lights_per_1000_steps': 1000.0 * robot_discoveries / robot_effective_time,
        })

    metrics_path = os.path.join(folder, "light_discovery_metrics.csv")
    pd.DataFrame(rows).to_csv(metrics_path, index=False)

    events_path = os.path.join(folder, "light_discovery_events.csv")
    found_df.to_csv(events_path, index=False)

    table_path = os.path.join(folder, "light_discovery_metrics_table.txt")
    with open(table_path, "w") as f:
        f.write("LIGHT DISCOVERY METRICS\n")
        f.write(f"Red lights in world: {total_red_lights}\n")
        f.write(f"Active robots with at least one registration: {active_robot_count}\n")
        f.write(f"Total mission steps: {total_steps}\n")
        f.write(f"Mission complete for all active robots: {all_active_complete}\n")
        f.write(f"Effective global time used: {effective_global_time}\n")
        f.write("\nSUMMARY\n")
        f.write("scope      robot           found/expected   last_step   effective_steps   discoveries_per_1000_steps\n")
        f.write("-----------------------------------------------------------------------------------------------\n")
        for row in rows:
            f.write(
                f"{row['scope']:<10} {row['robot']:<14} "
                f"{row['discoveries']}/{row['expected_discoveries']:<12} "
                f"{str(row['last_discovery_step']):<10} "
                f"{row['effective_time_steps']:<16} "
                f"{row['discoveries_per_1000_steps']:.6f}\n"
            )
        f.write("\nDISCOVERY EVENTS\n")
        f.write("step       robot           light_id\n")
        f.write("-----------------------------------\n")
        for _, event in found_df.sort_values('step').iterrows():
            f.write(f"{int(event['step']):<10} {event['robot']:<14} {event['light_id']}\n")

def save_ground_areas_csv(world_obj, folder):
    csv_path = os.path.join(folder, "ground_areas.csv")
    with open(csv_path, "w") as f:
        f.write("name,x,y,radius,color,role\n")
        for area_name, area_obj in world_obj.hierarchy.items():
            if not (hasattr(area_obj, "radius") and hasattr(area_obj, "color")):
                continue
            area_color = str(area_obj.color).lower()
            if area_color == "grey":
                role = "recoleccion"
            elif area_color == "black":
                role = "depositacion"
            else:
                role = area_color
            f.write(
                f"{area_name},{area_obj.position[0]:.3f},{area_obj.position[1]:.3f},"
                f"{area_obj.radius:.3f},{area_obj.color},{role}\n"
            )

def draw_ground_areas(folder):
    from matplotlib.patches import Circle

    areas_path = os.path.join(folder, "ground_areas.csv")
    if not os.path.exists(areas_path):
        return

    df_areas = pd.read_csv(areas_path)
    label_seen = set()
    styles = {
        "recoleccion": {"facecolor": "#eeeeee", "edgecolor": "#bdbdbd", "label": "Zona de recoleccion"},
        "depositacion": {"facecolor": "#d0d0d0", "edgecolor": "#9e9e9e", "label": "Zona de depositacion"},
    }

    ax = plt.gca()
    for _, area in df_areas.iterrows():
        role = str(area.get("role", "")).lower()
        style = styles.get(role)
        if style is None:
            continue
        label = style["label"] if role not in label_seen else None
        label_seen.add(role)
        ax.add_patch(
            Circle(
                (area["x"], area["y"]),
                area["radius"],
                facecolor=style["facecolor"],
                edgecolor=style["edgecolor"],
                linewidth=1.0,
                alpha=0.85,
                label=label,
                zorder=0,
            )
        )

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
    draw_ground_areas(folder)
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
            'stop': '#bdbdbd',
            'Espera': '#bdbdbd',
            'none': '#bdbdbd',
            'load_blue_battery': '#1f77b4',
            'load_red_battery': '#1f77b4',
            'Cargar batería azul': '#1f77b4',
            'Cargar bateria azul': '#1f77b4',
            'simple_forage': '#2ca02c',
            'navigate': '#2ca02c',
            'Buscar luces rojas': '#2ca02c',
            'orient_red_light': '#ff8c00',
            'approach_red_light': '#ff9999',
            'annotate_red_light_position': '#9467bd',
            'basic_obstacle_avoider': '#5c4033',
        }
        colores_extra = ['#17becf', '#e377c2', '#bcbd22', '#7f7f7f', '#000000']
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
        mapa_colores = plt.colormaps.get_cmap('tab20').resampled(num_robots)
    else:
        mapa_colores = plt.colormaps.get_cmap('hsv').resampled(num_robots)

    try:
        # --- Gráfica Trayectoria Superpuesta ---
        plt.figure(figsize=(8, 8))
        draw_ground_areas(folder)
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
        red_lines = 0
        plt.figure(figsize=(10, 5))
        for i, csv_path in enumerate(csv_files):
            df = pd.read_csv(csv_path)
            if len(df) == 0 or 'bat_roja' not in df.columns: continue
            
            robot_id = os.path.basename(csv_path).replace("recorrido_robot_", "").replace(".csv", "")
            plt.plot(df['step'], df['bat_roja'], color=mapa_colores(i), label=f'Robot {robot_id}')
            red_lines += 1

        if red_lines > 0:
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
            'stop': '#bdbdbd',
            'Espera': '#bdbdbd',
            'none': '#bdbdbd',
            'load_blue_battery': '#1f77b4',
            'load_red_battery': '#1f77b4',
            'Cargar batería azul': '#1f77b4',
            'Cargar bateria azul': '#1f77b4',
            'simple_forage': '#2ca02c',
            'navigate': '#2ca02c',
            'Buscar luces rojas': '#2ca02c',
            'orient_red_light': '#ff8c00',
            'approach_red_light': '#ff9999',
            'annotate_red_light_position': '#9467bd',
            'basic_obstacle_avoider': '#5c4033',
            'turn_yellow_lights_OFF': '#bcbd22',
            'go_to_coordenadas': '#17becf',
        }
        colores_extra = ['#e377c2', '#8c564b', '#bcbd22', '#7f7f7f', '#000000'] # Por si hay tareas nuevas
        
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

    # --- Salidas por trial ---
    import sys

    class Logger(object):
        def __init__(self, filename, terminal):
            self.terminal = terminal
            self.log = open(filename, "w")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

        def fileno(self):
            return self.terminal.fileno()

        def close(self):
            self.log.close()

    terminal_stdout = sys.stdout
    active_logger = {"logger": None}
    exp_folder = None

    def create_trial_folder(trial_idx=None):
        now = datetime.now().strftime("%m%d_%H%M%S_%f")
        suffix = f"_trial_{trial_idx}" if trial_idx is not None else ""
        folder = os.path.join("outputs", f"run_{now}{suffix}")
        os.makedirs(folder, exist_ok=True)
        return folder

    def set_trial_output_folder(folder):
        if active_logger["logger"] is not None:
            active_logger["logger"].flush()
            active_logger["logger"].close()
        os.environ["CURRENT_EXP_FOLDER"] = folder
        active_logger["logger"] = Logger(os.path.join(folder, "consola.log"), terminal_stdout)
        sys.stdout = active_logger["logger"]
        print(f"📁 Iniciando trial en: {folder}")

    def update_controller_output_paths(controller, folder):
        if controller is None:
            return
        if hasattr(controller, "output_dir"):
            controller.output_dir = folder
        if hasattr(controller, "metrics_log_name"):
            controller.metrics_log_name = os.path.join(folder, "llm_tiempos.csv")
        if hasattr(controller, "log_name"):
            controller.log_name = os.path.join(folder, "recorrido_robot.csv")
        if hasattr(controller, "_log_initialized"):
            controller._log_initialized = False
        for attr in ["survival_controller", "secondary_controller"]:
            child = getattr(controller, attr, None)
            if child is not None and child is not controller:
                update_controller_output_paths(child, folder)
        routines = getattr(controller, "routines", {})
        if isinstance(routines, dict):
            for child in routines.values():
                if child is not controller:
                    update_controller_output_paths(child, folder)

    def update_world_output_paths(world_obj, folder):
        os.environ["CURRENT_EXP_FOLDER"] = folder
        for robot in world_obj.robots.values():
            update_controller_output_paths(getattr(robot, "controller", None), folder)

    def finalize_trial_outputs(folder):
        generate_skill_usage(folder)
        generate_llm_timing_plots(folder, separate=("ExplorationGroup" in cfg))
        generate_skill_usage_pies(folder)
        if "ExplorationGroup" in cfg:
            generate_light_discovery_metrics(folder)
        if "AStoreKeeperLLMcentral" in cfg or "ExplorationGroup" in cfg:
            generate_plots_centralized(folder, arena_params=arena_params)
        else:
            generate_plots_classic(folder, arena_params=arena_params)

    exp_folder = create_trial_folder(0)
    set_trial_output_folder(exp_folder)

    arena_params = cfg_dict['world'].get('arena_params', {})

    # --- PREGUNTAR AL USUARIO PARA LOS EXPERIMENTOS CON LLM CENTRAL ---
    instrucciones = "No hay instrucciones específicas. Asigna tareas por defecto."
    if "AStoreKeeperLLMcentral" in cfg or "ExplorationGroup" in cfg:
        print("\n🧠 [CEREBRO CENTRAL]: Ingresa instrucciones iniciales para el LLM (ej: 'Manda 2 robots a cargar batería...'): ")
        entrada = input().strip()
        if entrada != "":
            instrucciones = entrada
        print(f"🧠 [CEREBRO CENTRAL]: Instrucciones enviadas a la IA: {instrucciones}\n")

    def build_world_instance():
        physics_engine = physics_engines[cfg_dict['world'].get('engine', 'pybullet')](
                            dt=cfg_dict['world'].get('physics_dt', 0.02),
                            T_control=cfg_dict['world'].get('T_control', 0.1))
        world_cls = worlds[cfg_dict['world'].get('name', 'square_arena')]
        world_obj = world_cls(physics_engine, **arena_params)
        world_obj.build_from_dict(
            cfg_dict['world'],
            ann_topology=cfg_dict.get('topology', {}),
            user_instructions=instrucciones,
        )

        if log:
            world_obj.config_data_logger(cfg_dict['logging']['data'])
            world_obj.data_logger.set_log_file(cfg_dict.get('logging', {}).get('file', cfg))

        if render:
            simulation_config = cfg_dict.get('simulation', {})
            world_obj.start_paused = simulation_config.get('start_paused', False)

        if 'virtual_space' in cfg_dict:
            topology_name = cfg_dict['virtual_space'].get('controller', {}).get('topology')
            world_obj.create_virtual_space(
                **cfg_dict['virtual_space'],
                topology=cfg_dict.get('topology', {}).get(topology_name),
            )

        update_world_output_paths(world_obj, os.environ.get("CURRENT_EXP_FOLDER", "outputs"))
        return world_obj

    # import copy
    # world2 = copy.deepcopy(world)
    # world.connect()
    # world2.connect()
    # print(world.physics_engine.client)
    # print(world2.physics_engine.client)
    # import pdb; pdb.set_trace()

    if cfg_dict.get('algorithm', False) and len(cfg_dict['algorithm']):
        world = build_world_instance()
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
        timesteps = simulation_config.get('timesteps', 10000)
        trials = simulation_config.get('trials', 1)
        np.random.seed(seed)
        # Interacción inicial con LLM ya se hizo antes
        current_trial_folder = exp_folder
        current_trial_finalized = False
        current_world = None
        try: 
            for tr in range(trials):
                current_trial_finalized = False
                if tr == 0:
                    current_trial_folder = exp_folder
                else:
                    current_trial_folder = create_trial_folder(tr)
                    set_trial_output_folder(current_trial_folder)
                exp_folder = current_trial_folder
                world = build_world_instance()
                current_world = world
                if render:
                    world.start_paused = simulation_config.get('start_paused', False)
                    if 'camera_options' in simulation_config:
                        world.physics_engine.set_camera_options(**simulation_config['camera_options'])
                    if 'animated_layout' in simulation_config and simulation_config['animated_layout'].get('enabled', False):
                        anim_config = simulation_config.get('animated_layout')
                        world.create_animated_layout(figsize=anim_config.get('figsize'))
                        world.animated_layout.add_plots(anim_config.get('plots'), grid=anim_config['grid'])
                        world.animated_layout.initialize(world)
                world.connect()
                print(f'Connected trial {tr}!')
                world.reset()
                try:
                    save_ground_areas_csv(world, exp_folder)
                except Exception as e:
                    print(f"Error guardando zonas de suelo: {e}")
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
                finalize_trial_outputs(exp_folder)
                current_trial_finalized = True
                if getattr(world, "llm_process", None) is not None and world.llm_process.is_alive():
                    world.llm_process.terminate()
                    world.llm_process.join(timeout=1)
                world.disconnect()
                current_world = None
        except KeyboardInterrupt:
            print("\n🛑 Simulación interrumpida.")
        finally:
            if not current_trial_finalized and exp_folder is not None:
                finalize_trial_outputs(exp_folder)
            if current_world is not None:
                if getattr(current_world, "llm_process", None) is not None and current_world.llm_process.is_alive():
                    current_world.llm_process.terminate()
                    current_world.llm_process.join(timeout=1)
                current_world.disconnect()
            if active_logger["logger"] is not None:
                active_logger["logger"].flush()
                active_logger["logger"].close()
                sys.stdout = terminal_stdout
if __name__ == "__main__":
    main()

