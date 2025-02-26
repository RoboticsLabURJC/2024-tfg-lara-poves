import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse

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

    for i, (key, values) in enumerate(data.items()):
        if key == 'velocity':
            key_tittle = 'las velocidades'
            key = 'Velocidad en m/s'
            color = 'blue'
            index = 0
            bin_width = 0.5
        elif key == 'deviation':
            key_tittle = 'la desviación del carril'
            key = 'Desviación en píxeles'
            color = 'red'
            index = 1
            bin_width = 2
        else:
            key_tittle = ''
        
        if len(key_tittle) > 1:
            values = np.array(values)

            bins = np.arange(min(values), max(values) + bin_width, bin_width) 
            axes[index].hist(values, bins=bins, edgecolor='black', alpha=0.7, color=color)
            axes[index].set_title(f'Histograma de {key_tittle}')
            axes[index].set_xlabel(key)
            axes[index].set_ylabel('Frecuencia')

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