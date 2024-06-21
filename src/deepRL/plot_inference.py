import matplotlib.pyplot as plt
import argparse
import csv
import numpy as np
import random
import os
import glob

NUM_COLUMNS = 3
NUM_ROWS = 2

def extract_data(key:str, data_csv:list[dict]):
    data = []
    for d in data_csv:
        data.append(float(d[key]))

    return data

def plot_data(data_csv:list[dict], key:str, sub_plot:int, title:str, hist:bool=False,
              label:str=None, color:str=None):
    data = extract_data(key, data_csv)
    plt.subplot(NUM_ROWS, NUM_COLUMNS, sub_plot)

    if not hist:
        plt.plot(range(len(data)), data, color=color, label=label) 
        plt.ylabel(key)
        plt.xlabel('Step')
    else:
        if key == 'Velocity':
            bins = [i for i in range(1, 11)]
        else:
            nsteer = 20
            bins = np.linspace(-0.2, 0.2, nsteer + 1)
            
        plt.hist(data, bins=bins, color=color, edgecolor='black', label=label, zorder=1)
        plt.ylabel('Frecuency')
        plt.xlabel(key)
    
    plt.title(title)
    if sub_plot % 3 == 0:
        plt.legend()

def get_color_random():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def main(args):
    random.seed(6)
    plt.figure(figsize=(5 * NUM_COLUMNS, 4 * NUM_ROWS))

    if len(args.file) == 1 and args.file[0] == 'all':
        dir = '/home/alumnos/lara/2024-tfg-lara-poves/src/deepRL/csv/inference/' 
        if args.env != None:
            dir += args.env + '/'

        text = '*.csv'
        if args.alg != None:
            text = args.alg + text

        if not os.path.isdir(dir):
            print(f"Error: Directory {dir} doesn't exit")
            exit(1)

        csv_files = glob.glob(os.path.join(dir, '**', '*.csv'), recursive=True)
    else:
        csv_files = args.file

    for csv_file in csv_files:
        data = []
        with open(csv_file, 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                data.append(row)

        csv_file = csv_file.split('/')[-1].rsplit('_', 1)[0]
        color = get_color_random()

        # Plots
        plot_data(data_csv=data, key='Reward', sub_plot=2, title='Reward per step', label=csv_file,
                  color=color)
        plot_data(data_csv=data, key='Accumulated reward', sub_plot=3, title='Total reward',
                  label=csv_file, color=color)
        plot_data(data_csv=data, key='Deviation', sub_plot=1, title='Deviation', label=csv_file,
                  color=color)
        plot_data(data_csv=data, key='Speed', sub_plot=4, title='Velocity of the vehicle',
                  label=csv_file, color=color)

        # Histograms
        plot_data(data_csv=data, key='Velocity', sub_plot=5, title='Histogram velocity actions',
                  hist=True, label=csv_file)
        plot_data(data_csv=data, key='Steer', sub_plot=6, title='Histogram steer actions', hist=True,
                  label=csv_file)
        
        file.close()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    possible_envs = [
        "CarlaDiscreteBasic"
    ]
    possible_algs = [
        "DQN"
    ]

    parser = argparse.ArgumentParser(
        description="Plot data of an inference trial",
        usage="python3 %(prog)s --file <FILE> [--env <ENV>] [--alg <ALG>]"
    )
    parser.add_argument(
        '--file', 
        type=str, 
        required=True, 
        nargs='+',
        help='Data file csv to plot, or \'all\' if you want the whole directory'
    )
    parser.add_argument(
        '--env', 
        type=str, 
        required=False, 
        default=None,
        choices=possible_envs,
        help='Gym environment. Possible values are: {' + ', '.join(possible_envs) + '}'
    )
    parser.add_argument(
        '--alg', 
        type=str, 
        required=False, 
        default=None,
        choices=possible_algs,
        help='The algorithm used. Possible values are: {' + ', '.join(possible_algs) + '}'
    )

    main(parser.parse_args())