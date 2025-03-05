import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def extract_velocity(filename):
    match = re.search(r'vel(\d+)', filename)
    return int(match.group(1)) if match else None

def process_csv_files(directory):
    velocities = []
    percentiles_90 = []
    percentiles_80 = []
    percentiles_70 = []

    for file in os.listdir(directory):
        if "vel" not in file or not file.endswith(".csv"):
            continue
        
        velocity = extract_velocity(file)
        if velocity is None:
            continue

        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)

        if "Distance" in df.columns:
            distances = df["Distance"].dropna() - 3  # El Lidar está a 3m del morro del coche
            
            p90 = np.percentile(distances, 90)
            p80 = np.percentile(distances, 80)
            p70 = np.percentile(distances, 70)

            velocities.append(velocity)
            percentiles_90.append(p90)
            percentiles_80.append(p80)
            percentiles_70.append(p70)

    return velocities, percentiles_90, percentiles_80, percentiles_70

def plot_results(velocities, p90, p80, p70):
    plt.figure(figsize=(12, 7))
    
    bar_width = 0.2
    x_positions = np.array(velocities)
    
    bars_90 = plt.bar(x_positions - bar_width, p90, width=bar_width, color='lightskyblue', label="Percentil 90")
    bars_80 = plt.bar(x_positions, p80, width=bar_width, color='lightgreen', label="Percentil 80")
    bars_70 = plt.bar(x_positions + bar_width, p70, width=bar_width, color='navajowhite', label="Percentil 70")

    labels = [f"{v} m/s\n{v * 3.6:.1f} km/h" for v in sorted(velocities)]

    # Crear la línea roja para la leyenda
    safety_legend = mpatches.Patch(color='red', label="Distancia mínima de seguridad (DGT)")

    for bar, v in zip(bars_90, velocities):
        d = (v * 3.6 / 10) ** 2
        bar_x = bar.get_x()  
        plt.plot([bar_x, bar_x + 3 * bar_width], [d, d], color='red', linestyle='dashed', linewidth=1.5)
        plt.text(bar_x + 1.5 * bar_width, d + 0.3, f"{d:.1f} m", color='red', fontsize=10, ha='center')

    plt.xlabel("Velocidad")
    plt.ylabel("Distancia Percentil (m)")
    plt.title("Distancia Percentil vs Velocidad")
    plt.xticks(sorted(velocities), labels)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Agregar la leyenda correctamente
    plt.legend(handles=[safety_legend, bars_90, bars_80, bars_70], loc="upper left")

    plt.show()

def main(args):
    velocities, p90, p80, p70 = process_csv_files(args.dir)
    
    if velocities:
        plot_results(velocities, p90, p80, p70)
    else:
        print("No se encontraron archivos CSV válidos en el directorio especificado.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot percentiles (90, 80, 70) of distances from front LiDAR",
        usage="python3 %(prog)s --dir <PATH_DIR>"
    )

    parser.add_argument(
        '--dir', 
        type=str, 
        required=True, 
        help='Data directory path to plot.'
    )

    main(parser.parse_args())

