import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse

# Configurar fuente similar a LaTeX
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "text.usetex": False,  # Si tienes LaTeX instalado, ponlo en True
    "axes.titlesize": 16,
    "axes.labelsize": 20,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15
})

def main(args):
    data = {}

    with open(args.file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            for key, value in row.items():
                if key not in data:
                    data[key] = []
                data[key].append(float(value))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 2 gráficos en la misma fila

    # Diccionario para personalizar cada variable
    config = {
        'velocity': {
            'title': 'las velocidades',
            'xlabel': 'Velocidad en m/s',
            'color': 'blue',
            'bin_width': 0.5,
            'xlim': (0, 25),
            'ylim': (0, 250),
            'index': 0
        },
        'deviation': {
            'title': 'la desviación del carril',
            'xlabel': 'Desviación en píxeles',
            'color': 'red',
            'bin_width': 2,
            'xlim': (-50, 50),  # Fijamos el rango de la desviación
            'ylim': (0, 1000),
            'index': 1
        }
    }

    for key, values in data.items():
        if key in config:
            cfg = config[key]

            values = np.array(values)
            bins = np.arange(min(values), max(values) + cfg['bin_width'], cfg['bin_width'])

            ax = axes[cfg['index']]
            ax.hist(values, bins=bins, edgecolor='black', alpha=0.7, color=cfg['color'])
            ax.set_xlabel(cfg["xlabel"], fontsize=16, fontweight='bold')
            ax.set_ylabel('Frecuencia', fontsize=16, fontweight='bold')
            ax.set_xlim(cfg["xlim"])  # Aplicar límites del eje X
            ax.set_ylim(cfg["ylim"])
            ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)  # Grid horizontal

            # Calcular la desviación estándar para 'deviation'
            if key == 'deviation':
                std_dev = np.std(values)
                print(f"Desviación estándar de la desviación del carril ({key}): {std_dev:.2f} píxeles")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage="python3 %(prog)s --file <file>"
    )
    parser.add_argument(
        '--file', 
        type=str, 
        required=True, 
        help='CSV file to plot'
    )

    main(parser.parse_args())