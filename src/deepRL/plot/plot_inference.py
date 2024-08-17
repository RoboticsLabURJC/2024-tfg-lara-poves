import matplotlib.pyplot as plt
import argparse
import csv
import numpy as np
import random
import os
import glob

NUM_COLUMNS = 3
NUM_ROWS = 2

def extract_data(key:str, data_csv:list[dict], val_abs:bool=False):
    data = []
    for d in data_csv:
        value = float(d[key]) 
        data.append(abs(value) if val_abs else value)

    return data

def plot_data(data_csv:list[dict], num_rows:int, key:str, sub_plot:int, title:str, 
              hist:bool=False, label:str=None, color:str=None):
    data = extract_data(key=key, data_csv=data_csv, val_abs=key=='Deviation')
    plt.subplot(num_rows, NUM_COLUMNS, sub_plot)

    if not hist:
        plt.plot(range(len(data)), data, color=color, label=label) 
        if key == 'Deviation':
            key += ' in pixels'
        plt.ylabel(key)
        plt.xlabel('Step')
    else:
        if key == 'Throttle':
            if 'DQN' in label:
                bins = np.linspace(0.05, 0.55, 6)
                bins_ticks = np.linspace(0.1, 0.5, 5)
            else:
                bins = np.linspace(0.1, 1.0, 10)
                bins_ticks = bins
        else:
            bins = np.linspace(-0.19, 0.19, 20)
            bins_ticks = np.linspace(-0.2, 0.2, 21)
            
        plt.hist(data, bins=bins, color=color, edgecolor='black', label=label, zorder=1)
        plt.xticks(bins_ticks, rotation=90)
        plt.ylabel('Frecuency')
        plt.xlabel(key)
    
    plt.title(title)
    if sub_plot == 3 or sub_plot == 4:
        plt.legend()

def get_color_random():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def main(args):
    random.seed(6)

    if len(args.file) == 1 and args.file[0] == 'all':
        dir = '/home/lpoves/2024-tfg-lara-poves/src/deepRL/csv/inference/' 
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

    plot_data_csv = []
    extra_row = False
    num_rows = NUM_ROWS

    for csv_file in csv_files:
        data = []
        with open(csv_file, 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                data.append(row)
        file.close()

        if not extra_row and float(data[0]['Brake']) >= 0.0:
            num_rows += 1
            extra_row = True

        csv_file = csv_file.split('/')[-1].rsplit('_', 1)[0]
        color = get_color_random()
        plot_data_csv.append([csv_file, data, color])

    # Plot csv data
    plt.figure(figsize=(5 * NUM_COLUMNS, 3.5 * num_rows))

    for csv_file, data, color in plot_data_csv:
        # Plots
        plot_data(data_csv=data, key='Reward', sub_plot=1, title='Reward per step', label=csv_file,
                  color=color, num_rows=num_rows)
        plot_data(data_csv=data, key='Deviation', sub_plot=4, title='Deviation in absolute value',
                  label=csv_file, color=color, num_rows=num_rows)
        plot_data(data_csv=data, key='Speed', sub_plot=5, title='Velocity of the vehicle',
                  label=csv_file, color=color, num_rows=num_rows)
        plot_data(data_csv=data, key='Throttle', sub_plot=2, title='Throttle of the vehicle',
                  label=csv_file, color=color, num_rows=num_rows)

        # Histograms
        plot_data(data_csv=data, key='Throttle', sub_plot=3, title='Histogram throttle actions',
                  hist=True, label=csv_file, num_rows=num_rows)
        plot_data(data_csv=data, key='Steer', sub_plot=6, title='Histogram steer actions', hist=True,
                  label=csv_file, num_rows=num_rows)
        
        # Extra column
        if num_rows > NUM_ROWS:
            plot_data(data_csv=data, key='Accumulated reward', sub_plot=7, color=color,
                      title='Accumulated reward', label=csv_file, num_rows=num_rows)
            plot_data(data_csv=data, num_rows=num_rows, key='Brake', title='Brake of the vehicle',
                      color=color, label=csv_file, sub_plot=8)
            plot_data(data_csv=data, key='Brake', sub_plot=9, title='Histogram brake actions',
                      hist=True, label=csv_file, num_rows=num_rows)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    possible_envs = [
        "CarlaLaneDiscrete",
        "CarlaLaneContinuousComplex",
        "CarlaLaneContinuousSimple"
    ]
    possible_algs = [
        "DQN",
        "PPO"
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