import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import csv
import numpy as np
import os
import sys

current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from deepRL.environment import KEY_DEV, KEY_THROTTLE, KEY_STEPS, KEY_STEER, KEY_BRAKE, KEY_BACK, KEY_REWARD, KEY_VEL, KEY_ACC_REWARD, KEY_DISTANCE

NUM_COLUMNS = 4
NUM_ROWS = 2

def extract_data(key:str, data_csv:list[dict], val_abs:bool=False):
    data = []
    for d in data_csv:
        value = float(d[key]) 
        data.append(abs(value) if val_abs else value)

    return data

def plot_data(data_csv:list[dict], num_rows:int, key:str, init:tuple[int, int], title:str, 
              hist:bool=False, label:str=None, size:int=1):
    # Extract data from csv
    data = extract_data(key=key, data_csv=data_csv, val_abs=key=='Deviation')
    
    # Locate the subplot in the grid
    ax = plt.subplot2grid((num_rows, NUM_COLUMNS), init, colspan=size)
    ax.set_title(title)
    
    # Draw plot
    if not hist:
        if key == KEY_DEV:
            key += ' in pixels'
        elif key == KEY_THROTTLE:
            ax.set_ylim(-0.1, 1.1)
        elif key == KEY_DISTANCE:
            data = np.nan_to_num(data, nan=10.0)

        ax.plot(range(len(data)), data, label=label) 
        ax.set_ylabel(key)
        ax.set_xlabel(KEY_STEPS)
    else:
        if key == KEY_STEER:
            bins = np.linspace(-0.19, 0.19, 20)
            bins_ticks = np.linspace(-0.2, 0.2, 21)
        else:
            if 'DQN' in label:
                bins = np.linspace(0.05, 0.55, 6)
                bins_ticks = np.linspace(0.1, 0.5, 5)
            else:
                bins = np.linspace(0.0, 1.0, 11)
                bins_ticks = bins
            
        ax.hist(data, bins=bins, edgecolor='black', label=label, zorder=1, color='lightgreen')
        ax.set_xticks(bins_ticks)
        ax.set_xticklabels([f'{tick:.2f}' for tick in bins_ticks], rotation=90)
        ax.set_ylabel('Frequency')
        ax.set_xlabel(key)

def main(args):
    os.chdir(current_dir)

    data = []
    with open(args.file, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data.append(row)
    file.close()

    num_rows = NUM_ROWS
    brake = False
    if float(data[0][KEY_BRAKE]) >= 0.0:
        num_rows += 1
        brake = True

    dist = False
    try:
        if data[0][KEY_DISTANCE] == "nan" or float(data[0][KEY_DISTANCE]) < 15.0:
            if not brake:
                num_rows +=1
            dist = True
    except Exception:
        pass

    back = False
    try:
        if data[0][KEY_BACK] == "nan" or float(data[0][KEY_BACK]) < 15.0:
            back = True
    except Exception:
        pass
    
    csv_file = args.file.split('/')[-1].rsplit('_', 1)[0]
    plt.figure(figsize=(6 * NUM_COLUMNS, 8))
    gs = gridspec.GridSpec(num_rows, NUM_COLUMNS)
   
    # Plots
    plot_data(data_csv=data, key=KEY_REWARD, init=(0, 0), title='Reward per step', label=csv_file,
              num_rows=num_rows)
    plot_data(data_csv=data, key=KEY_DEV, init=(1, 0), title='Deviation in absolute value',
              label=csv_file, num_rows=num_rows)
    plot_data(data_csv=data, key=KEY_VEL, init=(0, 2), title='Velocity of the vehicle',
              label=csv_file, num_rows=num_rows, size=2)
    plot_data(data_csv=data, key=KEY_THROTTLE, init=(1, 2), title='Throttle of the vehicle',
              label=csv_file, num_rows=num_rows, size=2)

    # Histograms
    plot_data(data_csv=data, key=KEY_THROTTLE, init=(1, 1), title='Histogram throttle actions',
                hist=True, label=csv_file, num_rows=num_rows)
    plot_data(data_csv=data, key=KEY_STEER, init=(0, 1), title='Histogram steer actions', hist=True,
                label=csv_file, num_rows=num_rows)
    
    # Extra row
    if num_rows > NUM_ROWS:
        size_init = 1
        size_brake = 2
        init_brake = (2, 1)

        if not dist and brake: # Only brake
            init = (2, 0)
        elif dist and not brake and not back: # Only front laser
            init = (2, 1)
        elif back and not brake: # Both laser
            init = (2, 2)
            size_init = 2
        else: # All
            init = None
            size_brake = 1
            init_brake = (2, 2)

        if brake:
            plot_data(data_csv=data, num_rows=num_rows, key=KEY_BRAKE, title='Brake of the vehicle',
                      label=csv_file, init=(init_brake[0], init_brake[1] + 1), size=size_brake)
            plot_data(data_csv=data, key=KEY_BRAKE, init=init_brake, title='Histogram brake actions',
                      hist=True, label=csv_file, num_rows=num_rows)
            
        if dist:
            plot_data(data_csv=data, key=KEY_DISTANCE, init=(2, 0), label=csv_file,
                      title='Distance front lidar', num_rows=num_rows)
            
        if back:
            plot_data(data_csv=data, key=KEY_BACK, init=(2, 1), label=csv_file, title='Distance back lidar',
                      num_rows=num_rows)
 
        if init != None:
            plot_data(data_csv=data, key=KEY_ACC_REWARD, init=init, label=csv_file,
                      title=KEY_ACC_REWARD, num_rows=num_rows, size=size_init)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot data of an inference trial",
        usage="python3 %(prog)s --file <FILE>"
    )

    parser.add_argument(
        '--file', 
        type=str, 
        required=True, 
        help='Data file csv to plot.'
    )

    main(parser.parse_args())