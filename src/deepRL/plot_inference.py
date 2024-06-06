import matplotlib.pyplot as plt
import argparse
import csv
import numpy as np

NUM_COLUMNS = 3
NUM_ROWS = 2

def extract_data(key:str, data_csv:list[dict]):
    data = []
    for d in data_csv:
        data.append(float(d[key]))

    return data

def plot_data(data_csv:list[dict], key:str, sub_plot:int, title:str, hist:bool=False,
              color:str=None, nsteer:int=0):
    data = extract_data(key, data_csv)
    plt.subplot(NUM_ROWS, NUM_COLUMNS, sub_plot)

    if not hist:
        plt.plot(range(len(data)), data, color=color) 
        plt.ylabel(key)
        plt.xlabel('Step')
    else:
        if key == 'Velocity':
            bins = [i for i in range(1, 11)]
        else:
            bins = np.linspace(-0.2, 0.2, nsteer + 1)
            
        plt.hist(data, bins=bins, color=color, edgecolor='black')
        plt.ylabel('Frecuency')
        plt.xlabel(key)
    
    plt.title(title)

def main(args):
    plt.figure(figsize=(5 * NUM_COLUMNS, 4 * NUM_ROWS))

    data = []
    with open(args.file, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data.append(row)

    # Plots
    plot_data(data_csv=data, key='Reward', sub_plot=2, title='Reward per step')
    plot_data(data_csv=data, key='Accumulated reward', sub_plot=3, title='Total reward', color='orange')
    plot_data(data_csv=data, key='Deviation', sub_plot=1, title='Deviation', color='green')
    plot_data(data_csv=data, key='Speed', sub_plot=4, title='Velocity of the vehicle', color='purple')

    # Histograms
    plot_data(data_csv=data, key='Velocity', sub_plot=5, title='Histogram velocity actions',
              hist=True, color='skyblue')
    plot_data(data_csv=data, key='Steer', sub_plot=6, title='Histogram steer actions', hist=True, 
              nsteer=args.nsteer, color='lightpink')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot data of an inference trial",
        usage="python3 %(prog)s --file <FILE> [--nsteer <N_STEER>]"
    )
    parser.add_argument(
        '--file', 
        type=str, 
        required=True, 
        help='Data file csv to plot'
    )
    parser.add_argument(
        '--nsteer', 
        type=int, 
        required=False, 
        default=20,
        help='Number of actions for steer'
    )

    main(parser.parse_args())