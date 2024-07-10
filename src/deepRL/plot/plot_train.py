import matplotlib.pyplot as plt
import argparse
import csv
import os
import glob
import random
import numpy as np

NUM_COLUMNS = 1
NUM_ROWS = 3

def plot_data(data_csv:list[dict], key:str, sub_plot:int, title:str, color:str=None,
              label:str=None, one:bool=False):
    plt.subplot(NUM_ROWS, NUM_COLUMNS, sub_plot)
    
    # Extract dats
    data = []
    finish = []
    for d in data_csv:
        data.append(float(d[key]))
        if one:
            if d['Finish'] == 'False':
                f = 0
            else:
                f = 1
            finish.append(f)

    # Only for step graph
    if one: 
        colors = np.where(finish, 'green', 'red')
        plt.scatter(range(len(data)), data, color=colors, s=7)
    else:
        plt.plot(range(len(data)), data, color=color, linewidth=1, label=label)

    plt.ylabel(key)
    plt.xlabel('Episode')
    
    plt.title(title)

def get_color_random():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def main(args):
    random.seed(6)
    plt.figure(figsize=(15 * NUM_COLUMNS, 3 * NUM_ROWS))

    if len(args.file) == 1 and args.file[0] == 'all':
        dir = '/home/alumnos/lara/2024-tfg-lara-poves/src/deepRL/csv/train/' 
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

        if len(csv_files) > 1:
            color = get_color_random()
        else:
            color = None
        csv_file = csv_file.split('/')[-1]

        # Plots
        plot_data(data_csv=data, key='Reward', sub_plot=1, title='Reward per episode',
                  color=color, label=csv_file)
        plt.legend()
        plot_data(data_csv=data, key='Num_steps', sub_plot=2, title='Steps per epidose',
                  color=color, label=csv_file, one=len(csv_files)==1)
        plot_data(data_csv=data, key='Exploration_rate', sub_plot=3, title='Decay exploration rate', 
                  color=color, label=csv_file)

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