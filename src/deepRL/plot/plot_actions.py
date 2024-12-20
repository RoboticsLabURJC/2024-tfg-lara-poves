import matplotlib.pyplot as plt
import argparse
import csv
import numpy as np
import os
import sys

current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from deepRL.environment import KEY_THROTTLE, KEY_STEER

NUM_ROWS = 1
NUM_COLUMNS = 2

def histogram(data_csv:list[dict], key:str, subplot:int):
    # Extract data from csv
    data = []
    for d in data_csv:
        value = float(d[key]) 
        data.append(value)

    plt.subplot(NUM_ROWS, NUM_COLUMNS, subplot)
    plt.title(key)
    
    # Draw histogram
    extra = 0.05
    if key == KEY_STEER:
        extra /= 2
        bins = np.arange(-0.18, 0.2, 0.02)
    else:
        bins = np.linspace(0.0, 1.0, 11)
     
    counts, edges, _ = plt.hist(data, bins=bins, edgecolor='black', zorder=1)
    if key != KEY_STEER:
        for count, edge in zip(counts, edges):
            plt.text(edge + extra, count, str(int(count)), ha='center', va='bottom')

    plt.xticks(bins, rotation=90)
    plt.ylabel('Frecuency')
    plt.xlabel('Value')

def main(args):
    os.chdir(current_dir)

    data = []
    with open(args.file, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data.append(row)
    file.close()

    plt.figure(figsize=(6 * NUM_COLUMNS, 5 * NUM_ROWS))

    histogram(data_csv=data, key=KEY_THROTTLE, subplot=1)
    histogram(data_csv=data, key=KEY_STEER, subplot=2)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot histograms of the control actions explored during training",
        usage="python3 %(prog)s --file <FILE> "
    )
    parser.add_argument(
        '--file', 
        type=str, 
        required=True, 
        help='Data file csv to plot'
    )

    main(parser.parse_args())
