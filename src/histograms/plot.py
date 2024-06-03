import matplotlib.pyplot as plt
import csv
import numpy as np

def main():
    csv_path = '/home/alumnos/lara/2024-tfg-lara-poves/src/histograms/dist.csv'

    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)

        num_rows = sum(1 for row in reader)
        csvfile.seek(0) # Go back to the beginning of the file

        titles = []
        data = [[] for _ in range(num_rows)]
        index = 0

        for row in reader:
            titles.append(row[0])  
            data[index].extend(map(float, row[1:]))
            index += 1
            
    if num_rows % 2 != 0:
        num_rows += 1

    fig, axs = plt.subplots(num_rows // 2, 2, figsize=(12, 4.5 * (num_rows // 2)))

    for i, (title, d) in enumerate(zip(titles, data)):
        row = i // 2 
        col = i % 2  
        axs[row, col].hist(d, bins=np.linspace(0, 10, num=101))
        axs[row, col].set_title(title)
        axs[row, col].set_xlabel('Distance (m)')
        axs[row, col].set_ylabel('Frequency')
        axs[row, col].set_ylim(0, 26) 

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()