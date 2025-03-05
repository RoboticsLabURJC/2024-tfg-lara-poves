import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import csv
import numpy as np
import os
import sys

current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from deepRL.environment import KEY_DEV, KEY_THROTTLE, KEY_STEPS, KEY_STEER, KEY_REWARD, KEY_VEL, KEY_ACC_REWARD, KEY_DISTANCE, KEY_LASER_RIGHT

NUM_COLUMNS = 2 # 4
NUM_ROWS = 2

def extract_data(key:str, data_csv:list[dict], val_abs:bool=False):
    data = []
    for d in data_csv:
        value = float(d[key]) 
        data.append(abs(value) if val_abs else value)

    return data

def plot_data(data_csv:list[dict], num_rows:int, key:str, init:tuple[int, int], title:str, 
              hist:bool=False, label:str=None, size:int=1, file:str=None):
    # Extract data from csv
    data = extract_data(key=key, data_csv=data_csv, val_abs=key=='Deviation')
    
    # Locate the subplot in the grid
    ax = plt.subplot2grid((num_rows, NUM_COLUMNS), init, colspan=size)
    ax.set_title(title)
    
    # Draw plot
    if not hist:
        if key == KEY_DEV:
            key += ' in pixels'
            ax.set_ylim(-20, 20)
        elif key == KEY_THROTTLE:
            if file != None and 'Obstacle' in file:
                ax.set_ylim(-0.1, 0.5)
            else:
                ax.set_ylim(-0.1, 1.1)
        elif key == KEY_DISTANCE:
            data = np.nan_to_num(data, nan=10.0)
        elif key == KEY_REWARD:
            ax.set_ylim(bottom=0.0, top=1.0)

        ax.plot(range(len(data)), data, label=label) 
        ax.set_ylabel(key)
        ax.set_xlabel(KEY_STEPS)
    else:
        if key == KEY_STEER:
            bins = np.linspace(-0.19, 0.19, 20)
            bins_ticks = np.linspace(-0.2, 0.2, 21)
        elif key == KEY_DISTANCE:
            data = np.array(data)
            data = data[~np.isnan(data)]
            bins = np.arange(4, 21, 1)
            bins_ticks = bins
        else:
            if 'DQN' in label:
                bins = np.array([0.1, 0.2, 0.3, 0.4, 0.45, 0.5])
                bins_ticks = np.array([0.1, 0.2, 0.3, 0.4, 0.45, 0.5])
            else:
                bins = np.linspace(0.0, 1.0, 11)
                bins_ticks = bins
            
        ax.hist(data, bins=bins, edgecolor='black', label=label, zorder=1, color='lightgreen')
        ax.set_xticks(bins_ticks)
        ax.set_xticklabels([f'{tick:.2f}' for tick in bins_ticks], rotation=90)
        ax.set_ylabel('Frequency')
        ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
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
    dist = False
    try:
        if data[0][KEY_DISTANCE] != "nan" or float(data[0][KEY_DISTANCE]) < 15.0:
            num_rows +=1
            dist = True
    except Exception:
        pass

    back = False
    try:
        if data[0][KEY_ACC_REWARD] == "nan" or float(data[0][KEY_LASER_RIGHT]) < 15.0:
            back = True
    except Exception:
        pass
    
    csv_file = args.file.split('/')[-1].rsplit('_', 1)[0]
    plt.figure(figsize=(6 * NUM_COLUMNS, 5)) # 8

    num_rows = 1 # Para solo hacer los histogramas

    gs = gridspec.GridSpec(num_rows, NUM_COLUMNS)
   
    # Plots
    #plot_data(data_csv=data, key=KEY_REWARD, init=(0, 0), title='Reward per step', label=csv_file, num_rows=num_rows)
    #plot_data(data_csv=data, key=KEY_DEV, init=(1, 0), title='Deviation in absolute value',label=csv_file, num_rows=num_rows)
    #plot_data(data_csv=data, key=KEY_VEL, init=(0, 2), title='Velocity of the vehicle', label=csv_file, num_rows=num_rows, size=2)
    #plot_data(data_csv=data, key=KEY_THROTTLE, init=(1, 2), title='Throttle of the vehicle',label=csv_file, num_rows=num_rows, size=2, file=args.file)

    # Histograms
    plot_data(data_csv=data, key=KEY_THROTTLE, init=(0, 0), title='Histogram throttle actions',
                hist=True, label=csv_file, num_rows=num_rows) # (1,1)
    plot_data(data_csv=data, key=KEY_STEER, init=(0, 1), title='Histogram steer actions', hist=True,
                label=csv_file, num_rows=num_rows) # (0, 1)
    
    # Extra row
    '''
    if num_rows > NUM_ROWS:    
        if dist:
            plot_data(data_csv=data, key=KEY_DISTANCE, init=(2, 2), label=csv_file,
                      title='Distance front LiDAR', num_rows=num_rows, size=2)
            
            if not back:
                plot_data(data_csv=data, key=KEY_DISTANCE, init=(2, 1), label=csv_file, hist=True,
                          title='Histograms distance LiDAR', num_rows=num_rows, size=1)
            
        if back:
            plot_data(data_csv=data, key=KEY_LASER_RIGHT, init=(2, 1), label=csv_file, title='Distance back lidar',
                      num_rows=num_rows)
 

        plot_data(data_csv=data, key=KEY_ACC_REWARD, init=(2, 0), label=csv_file, title=KEY_ACC_REWARD,
                  num_rows=num_rows)
    '''
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