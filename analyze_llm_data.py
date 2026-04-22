import csv
import matplotlib.pyplot as plt
import os
import glob

# Buscar el CSV más reciente en outputs/
csv_files = glob.glob("outputs/**/recorrido_robot.csv", recursive=True)
if csv_files:
    csv_file = max(csv_files, key=os.path.getmtime)  # El más reciente por modificación
    print(f"Usando archivo: {csv_file}")
else:
    print("No se encontraron archivos CSV en outputs/")
    exit()

if os.path.exists(csv_file):
    steps = []
    xs = []
    ys = []
    bat_azuls = []
    bat_rojas = []
    num_luces = []
    decisions = []
    decision_tss = []

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            steps.append(int(row[0]))
            xs.append(float(row[1]))
            ys.append(float(row[2]))
            bat_azuls.append(float(row[3]))
            bat_rojas.append(float(row[4]))
            num_luces.append(int(row[5]))
            decisions.append(row[6] if len(row) > 6 else '')
            decision_tss.append(int(row[7]) if len(row) > 7 else 0)

    print("Datos cargados:")
    print(f"Total steps: {len(steps)}")

    # Obtener el directorio del CSV
    output_dir = os.path.dirname(csv_file)

    # Gráfico de baterías vs tiempo (superpuestas)
    plt.figure(figsize=(10, 6))
    plt.plot(steps, bat_azuls, label='Batería Azul', color='blue')
    plt.plot(steps, bat_rojas, label='Batería Roja', color='red')
    plt.title('Niveles de Batería vs Tiempo')
    plt.xlabel('Step')
    plt.ylabel('Nivel de Batería')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'baterias_superpuestas.png'))
    plt.close()

    # Gráfico de decisiones
    unique_decisions = set(decisions)
    plt.figure(figsize=(10, 6))
    for dec in unique_decisions:
        if dec:
            indices = [i for i, d in enumerate(decisions) if d == dec]
            plt.scatter([bat_azuls[i] for i in indices], [bat_rojas[i] for i in indices], label=dec, alpha=0.7)
    plt.title('Decisiones del LLM vs Niveles de Batería')
    plt.xlabel('Batería Azul')
    plt.ylabel('Batería Roja')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'decisiones.png'))
    plt.close()

    print(f"Gráficos generados en {output_dir}/")

else:
    print(f"Archivo {csv_file} no encontrado.")