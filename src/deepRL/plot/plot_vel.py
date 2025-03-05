import matplotlib.pyplot as plt
import pandas as pd
import argparse
import sys
import os

current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from deepRL.environment import KEY_VEL

def main(args):
    os.chdir(current_dir)

    csv_file = args.file
    data = pd.read_csv(csv_file)

    if KEY_VEL not in data.columns:
        raise ValueError("The CSV file must contain a column named 'velocity'.")
    
    plt.figure(figsize=(15, 12))

    velocities = data[KEY_VEL]
    bins = [5, 6, 7, 8, 9, 10, 11, 12]
    labels = ["vel 5 m/s", "vel 6 m/s", "vel 7 m/s", "vel 8 m/s", 
              "vel 9 m/s", "vel 10 m/s", "not seen car", ""]

    # Ajuste de tamaño de fuente
    plt.xticks(bins, labels, rotation=0, fontsize=17)
    plt.yticks(fontsize=17)

    plt.hist(velocities, bins=bins, edgecolor='black', align='left', rwidth=0.8, color='skyblue')

    plt.xlabel("Velocity (m/s)", fontsize=20)
    plt.ylabel("Frequency", fontsize=20)

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a histogram of velocities from a CSV file.")
    parser.add_argument(
        '--file', 
        type=str, 
        required=True, 
        help="Path to the CSV file containing the 'velocity' column."
    )
    args = parser.parse_args()

    main(args)