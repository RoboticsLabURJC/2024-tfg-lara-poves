import matplotlib.pyplot as plt
import argparse
import csv
import numpy as np

NUM_COLUMNS = 2 
NUM_ROWS = 1

def plot_data(data_csv:list[dict], key:str, sub_plot:int, title:str, color:str=None):
    data = []
    for d in data_csv:
        data.append(float(d[key]))

    plt.subplot(NUM_ROWS, NUM_COLUMNS, sub_plot)
    plt.plot(range(len(data)), data, color=color) 
    plt.ylabel(key)
    plt.xlabel('Episode')
    
    plt.title(title)

def main(args):
    plt.figure(figsize=(9 * NUM_COLUMNS, 7 * NUM_ROWS))

    data = []
    with open(args.file, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data.append(row)

    # Plots
    plot_data(data_csv=data, key='Reward', sub_plot=1, title='Reward per episode')
    plot_data(data_csv=data, key='Num_steps', sub_plot=2, title='Steps per epidose', color='orange')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot data of a train",
        usage="python3 %(prog)s --file <FILE>"
    )
    parser.add_argument(
        '--file', 
        type=str, 
        required=True, 
        help='Data file csv to plot'
    )

    main(parser.parse_args())