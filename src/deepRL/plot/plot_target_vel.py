import matplotlib.pyplot as plt
import csv
import argparse

def main(args):
    file_path = args.file
    velocities = []

    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            for value in row:
                try:
                    # Convert each value to float and add it to the list
                    velocities.append(float(value))
                except ValueError:
                    pass

    # Create the histogram
    bins = [i for i in range(11)]
    plt.hist(velocities, bins=bins, edgecolor='black', alpha=0.7)

    # Customize the plot
    plt.title("Histogram of Target Velocities") 
    plt.xlabel("Velocity Range (m/s)")
    plt.ylabel("Frequency")
    plt.xticks(ticks=bins)

    # Display the plot
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
