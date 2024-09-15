import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import csv
import numpy as np
import random
import os
import glob

NUM_COLUMNS = 4
NUM_ROWS = 2

def extract_data(key:str, data_csv:list[dict], val_abs:bool=False):
    data = []
    for d in data_csv:
        value = float(d[key]) 
        data.append(abs(value) if val_abs else value)

    return data

def plot_data(data_csv:list[dict], num_rows:int, key:str, init:tuple[int, int], title:str, 
              hist:bool=False, label:str=None, color:str=None, size:int=1):
    # Extract data from csv
    data = extract_data(key=key, data_csv=data_csv, val_abs=key=='Deviation')
    
    # Locate the subplot in the grid
    ax = plt.subplot2grid((num_rows, NUM_COLUMNS), init, colspan=size)
    ax.set_title(title)
    
    # Draw plot
    if not hist:
        if key == 'Deviation':
            key += ' in pixels'

        ax.plot(range(len(data)), data, color=color, label=label) 
        ax.set_ylabel(key)
        ax.set_xlabel('Step')
    else:
        if key == 'Steer':
            if 'DQN' in label:
                bins = np.linspace(-0.19, 0.19, 20)
                bins_ticks = np.linspace(-0.2, 0.2, 21)
            else:
                bins = np.linspace(-0.3, 0.3, 13)
                bins_ticks = bins
        else:
            if 'DQN' in label:
                bins = np.linspace(0.05, 0.55, 6)
                bins_ticks = np.linspace(0.1, 0.5, 5)
            else:
                bins = np.linspace(0.0, 1.0, 11)
                bins_ticks = bins
            
        ax.hist(data, bins=bins, color=color, edgecolor='black', label=label, zorder=1)
        ax.set_xticks(bins_ticks)
        ax.set_xticklabels([f'{tick:.2f}' for tick in bins_ticks], rotation=90)
        ax.set_ylabel('Frequency')
        ax.set_xlabel(key)
    
    if init == (0, 2) or init == (0, 1):
       plt.legend()

def get_color_random():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def main(args):
    random.seed(7)

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
    plt.figure(figsize=(6 * NUM_COLUMNS, 8))
    gs = gridspec.GridSpec(num_rows, NUM_COLUMNS)

    for csv_file, data, color in plot_data_csv:
        # Plots
        plot_data(data_csv=data, key='Reward', init=(0, 0), title='Reward per step', label=csv_file,
                  color=color, num_rows=num_rows)
        plot_data(data_csv=data, key='Deviation', init=(1, 0), title='Deviation in absolute value',
                  label=csv_file, color=color, num_rows=num_rows)
        plot_data(data_csv=data, key='Velocity', init=(0, 2), title='Velocity of the vehicle',
                  label=csv_file, color=color, num_rows=num_rows, size=2)
        plot_data(data_csv=data, key='Throttle', init=(1, 2), title='Throttle of the vehicle',
                  label=csv_file, color=color, num_rows=num_rows, size=2)

        # Histograms
        plot_data(data_csv=data, key='Throttle', init=(1, 1), title='Histogram throttle actions',
                  hist=True, label=csv_file, num_rows=num_rows)
        plot_data(data_csv=data, key='Steer', init=(0, 1), title='Histogram steer actions', hist=True,
                  label=csv_file, num_rows=num_rows)
        
        # Extra row
        if extra_row:
            plot_data(data_csv=data, key='Accumulated reward', init=(2, 0), color=color,
                      title='Accumulated reward', label=csv_file, num_rows=num_rows)
            plot_data(data_csv=data, num_rows=num_rows, key='Brake', title='Brake of the vehicle',
                      color=color, label=csv_file, init=(2, 2), size=2)
            plot_data(data_csv=data, key='Brake', init=(2, 1), title='Histogram brake actions',
                      hist=True, label=csv_file, num_rows=num_rows)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    possible_envs = [
        "CarlaLaneDiscrete",
        "CarlaLaneContinuous",
        "CarlaLane"
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