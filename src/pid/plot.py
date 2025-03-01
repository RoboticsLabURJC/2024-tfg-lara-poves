import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse
from matplotlib import rcParams

# Configurar la fuente para que sea similar a LaTeX (Computer Modern)
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
rcParams['text.usetex'] = False  # Si tienes LaTeX instalado, puedes cambiar esto a True


def main(args):
    data = {}

    with open(args.file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            for key, value in row.items():
                if key not in data:
                    data[key] = []
                data[key].append(float(value))

    num_columns = 2
    fig, axes = plt.subplots(1, num_columns, figsize=(6 * num_columns, 6))

    if num_columns == 1:
        axes = [axes]

    label_fontsize = 14  # Aumentamos el tamaño de las etiquetas de los ejes

    for i, (key, values) in enumerate(data.items()):
        if key == 'velocity':
            key_title = 'las velocidades'
            key = 'Velocidad en m/s'
            color = 'blue'
            index = 0
            bin_width = 0.5
        elif key == 'deviation':
            key_title = 'la desviación del carril'
            key = 'Desviación en píxeles'
            color = 'red'
            index = 1
            bin_width = 2
        else:
            key_title = ''
            continue  # Si no es una de las claves esperadas, lo ignoramos

        values = np.array(values)
        bins = np.arange(min(values), max(values) + bin_width, bin_width)
        axes[index].hist(values, bins=bins, edgecolor='black', alpha=0.7, color=color)
        axes[index].set_xlabel(key, fontsize=label_fontsize)
        axes[index].set_ylabel('Frecuencia', fontsize=label_fontsize)

        if 'vel' in key.lower():
            axes[index].set_xlim(0, 25)
        elif 'des' in key.lower():
            axes[index].set_xlim(-50, 50)

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