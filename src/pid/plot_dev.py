import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse

# Configurar tamaño de las fuentes para mejorar visibilidad
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "text.usetex": False,  # Ponlo en True si tienes LaTeX instalado
    "axes.titlesize": 18,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18
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

    if 'deviation' not in data:
        print("Error: El archivo CSV no contiene la columna 'deviation'.")
        return

    # Configuración de la desviación
    cfg = {
        'title': 'Distribución de la Desviación del Carril',
        'xlabel': 'Desviación en píxeles',
        'color': 'red',
        'bin_width': 2,
        'xlim': (-50, 50),
        'ylim': (0, 1500),
    }

    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convertir datos en array y definir bins
    values = np.array(data['deviation'])
    bins = np.arange(min(values), max(values) + cfg['bin_width'], cfg['bin_width'])

    # Graficar histograma
    ax.hist(values, bins=bins, edgecolor='black', alpha=0.7, color=cfg['color'])
    ax.set_xlabel(cfg["xlabel"], fontsize=22, fontweight='bold')
    ax.set_ylabel('Frecuencia', fontsize=22, fontweight='bold')
    ax.set_xlim(cfg["xlim"])
    ax.set_ylim(cfg["ylim"])
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Calcular la desviación estándar
    std_dev = np.std(values)
    print(f"Desviación estándar de la desviación del carril: {std_dev:.2f} píxeles")

    # Ajustar el diseño y mostrar
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graficar histograma de desviación del carril.")
    parser.add_argument(
        '--file', 
        type=str, 
        required=True, 
        help='Ruta del archivo CSV que contiene la columna "deviation".'
    )

    main(parser.parse_args())
