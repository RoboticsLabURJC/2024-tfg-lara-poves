import matplotlib.pyplot as plt
import argparse
import csv
import os
import matplotlib.patches as mpatches
import sys
import numpy as np

current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from deepRL.environment import KEY_FINISH, KEY_EXP_RATE, KEY_REWARD, KEY_STEPS, KEY_MEAN_VEL, KEY_COUNTER_LASER

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
        # Scatter plot with colors based on finish
        color_map = {0: 'red', 1: 'green'}
        colors = [color_map[f] for f in finish] 
        plt.scatter(range(len(data)), data, color=colors, s=7)

        # Legend
        red_patch = mpatches.Patch(color='red', label='Finish = False')
        green_patch = mpatches.Patch(color='green', label='Finish = True')
        plt.legend(handles=[red_patch, green_patch])  
    else:
        # Line plot
        if key == KEY_COUNTER_LASER:
            plt.grid(axis='y', linestyle='--', linewidth=0.7) 
            plt.yticks(range(0, int(max(data)) + 10, 10))

        plt.plot(range(len(data)), data, linewidth=1, label=key)

        if key == KEY_REWARD:
            cumulative_avg = np.cumsum(data) / np.arange(1, len(data) + 1)
            plt.plot(range(len(data)), cumulative_avg, label='ep rew mean', linewidth=2.5, color='orange')
            plt.legend()

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
    if float(data[0][KEY_EXP_RATE]) < 0.0 and not args.mean_vel and not args.laser:
        num_rows = 2
    elif args.mean_vel and args.laser:
        num_rows = 4

    plot_data(data_csv=data, key=KEY_REWARD, sub_plot=1, title='Reward per episode', num_rows=num_rows)
    plot_data(data_csv=data, key=KEY_STEPS, sub_plot=2, title='Steps per epidose', num_rows=num_rows, points=True)
    
    if num_rows >= 3:
        if args.mean_vel:
            plot_data(data_csv=data, key=KEY_MEAN_VEL, sub_plot=3, title='Mean velocity histogram', num_rows=num_rows)
        
        if args.laser:
            plot_data(data_csv=data, key=KEY_COUNTER_LASER, sub_plot=num_rows, title='Percentage of the vehicle visible', num_rows=num_rows)

        if not args.laser and not args.mean_vel:
            plot_data(data_csv=data, key=KEY_EXP_RATE, sub_plot=3, title='Decay exploration rate', num_rows=num_rows)

    fig.set_size_inches(15 * NUM_COLUMNS, 3 * num_rows)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot data of an inference trial",
        usage="python3 %(prog)s --file <FILE> --mean_vel <MEAN_VEL>"
    )
    parser.add_argument(
        '--file', 
        type=str, 
        required=True, 
        help='Data file csv to plot.'
    )
    parser.add_argument(
        '--mean_vel', 
        type=int, 
        required=False, 
        default=0,
        help='Display a histogram showing the mean velocities of the episodes. Defaults to 0 (False).'
    )
    parser.add_argument(
        '--laser', 
        type=int, 
        required=False, 
        default=0,
        help='Display a plot showing the percentage of the vehicle visible during each episode. Defaults to 0 (False).'
    )

    main(parser.parse_args())