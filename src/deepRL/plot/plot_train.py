import matplotlib.pyplot as plt
import argparse
import csv
import os
import matplotlib.patches as mpatches
import sys

current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from deepRL.environment import KEY_FINISH, KEY_EXP_RATE, KEY_REWARD, KEY_STEPS

NUM_COLUMNS = 1

def plot_data(data_csv:list[dict], key:str, sub_plot:int, title:str, num_rows:int, points:bool=False):
    plt.subplot(num_rows, NUM_COLUMNS, sub_plot)
    
    # Extract data
    data = []
    finish = []
    for d in data_csv:
        data.append(float(d[key]))

        if points:
            if d[KEY_FINISH] == 'False': 
                f = 0
            else:
                f = 1
            finish.append(f)

    if points:
        color_map = {0: 'red', 1: 'green'}
        colors = [color_map[f] for f in finish] 
        plt.scatter(range(len(data)), data, color=colors, s=7)

        # Legend
        red_patch = mpatches.Patch(color='red', label='Finish = False')
        green_patch = mpatches.Patch(color='green', label='Finish = True')
        plt.legend(handles=[red_patch, green_patch])  
    else:
        plt.plot(range(len(data)), data, linewidth=1)

    plt.ylabel(key)
    plt.xlabel('Episode')
    
    plt.title(title)

def main(args):    
    os.chdir(current_dir)
    csv_file = args.file

    # Create plot
    num_rows = 3
    fig = plt.figure(figsize=(15 * NUM_COLUMNS, 3 * num_rows))

    data = []
    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data.append(row)

    csv_file = csv_file.split('/')[-1]

    # Plots
    if float(data[0][KEY_EXP_RATE]) < 0.0:
        num_rows = 2

    plot_data(data_csv=data, key=KEY_REWARD, sub_plot=1, title='Reward per episode', num_rows=num_rows)
    plot_data(data_csv=data, key=KEY_STEPS, sub_plot=2, title='Steps per epidose', num_rows=num_rows, points=True)
    
    if num_rows == 3:
        plot_data(data_csv=data, key=KEY_EXP_RATE, sub_plot=3, title='Decay exploration rate', num_rows=num_rows)

    fig.set_size_inches(15 * NUM_COLUMNS, 3 * num_rows)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    possible_envs = [
        "CarlaLaneDiscrete",
        "CarlaLaneContinuousSimple",
        "CarlaLaneContinuousComplex",
        "CarlaObstacle"
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
        help='Data file csv to plot.'
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