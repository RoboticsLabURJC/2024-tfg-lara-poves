import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def extract_velocity(filename):
    match = re.search(r'vel(\d+)', filename)
    return int(match.group(1)) if match else None

def process_csv_files(directory):
    velocities = []
    mean_distances = []

    for file in os.listdir(directory):
        if "vel" not in file or not file.endswith(".csv"):
            continue
        
        velocity = extract_velocity(file)
        if velocity is None:
            continue

        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)

        if "Distance" in df.columns:
            mean_distance = df["Distance"].dropna().mean()
            velocities.append(velocity)
            mean_distances.append(mean_distance - 3) # El Lidar está a 3m del morro del coche

    return velocities, mean_distances

def plot_results(velocities, mean_distances):
    plt.figure(figsize=(8, 6))
    bars = plt.bar(velocities, mean_distances, color='royalblue', edgecolor='black', alpha=0.7)

    labels = [f"{v} m/s\n{v * 3.6:.1f} km/h" for v in sorted(velocities)]

    # Crear una línea ficticia para la leyenda
    safety_legend = mpatches.Patch(color='red', label="Distancia mínima de seguridad (DGT)")

    for bar, v in zip(bars, velocities):
        d = (v * 3.6 / 10) ** 2

        bar_x = bar.get_x()  # Coordenada X de la barra
        bar_width = bar.get_width()  # Ancho de la barra
        plt.plot([bar_x, bar_x + bar_width], [d, d], color='red', linestyle='dashed', linewidth=1.5)
        plt.text(bar_x + bar_width / 2, d + 0.3, f"{d:.1f} m", color='red', fontsize=10, ha='center')

    plt.xlabel("Velocidad")
    plt.ylabel("Distancia media (m)")
    plt.xticks(sorted(velocities), labels)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Agregar la leyenda con solo la línea roja
    plt.legend(handles=[safety_legend], loc="upper left")

    plt.show()

def main(args):
    velocities, mean_distances = process_csv_files(args.dir)
    
    if velocities:
        plot_results(velocities, mean_distances)
    else:
        print("No se encontraron archivos CSV válidos en el directorio especificado.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot mean distances of the front LiDAR",
        usage="python3 %(prog)s --dir <PATH_DIR>"
    )

    parser.add_argument(
        '--dir', 
        type=str, 
        required=True, 
        help='Data dirrectory path to plot.'
    )

    main(parser.parse_args())
