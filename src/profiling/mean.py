from collections import defaultdict
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors
import numpy as np

MILI_TO_NANO = 10**6

def get_data():
    dir = "/home/lpoves/2024-tfg-lara-poves/src/profiling/times"
    all_tasks = set() 
    data = defaultdict(dict)  

    for file_name in os.listdir(dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(dir, file_name)

            with open(file_path, "r") as file:
                sum = defaultdict(int)
                count = defaultdict(int)

                for line in file:
                    task, time_ns = line.strip().split(": ")
                    time_ns = int(time_ns.split()[0])  
                    if not 'back' in task:
                        if 'front' in task:
                            task = 'Lidar - Get points'
                        sum[task] += time_ns
                        count[task] += 1
                        all_tasks.add(task)

                means = {task: sum[task] / count[task] for task in sum}
                data[file_name] = means

    return all_tasks, data

def main():
    np.random.seed(4)

    all_tasks, data = get_data()
    num_files = len(data)

    colors = list(mcolors.CSS4_COLORS.keys())
    np.random.shuffle(colors)
    colors = colors[:num_files]

    num_columns = 3
    num_rows = num_files // num_columns if num_files % num_columns == 0 else num_files // num_columns + 1
    _, axs = plt.subplots(num_rows, num_columns, figsize=(6.6 * num_columns, num_rows * 7))

    file_index = 0
    for file_name, means, color in zip(data.keys(), data.values(), colors):
        row = file_index // num_columns
        col = file_index % num_columns
        ax = axs[row, col] if num_rows > 1 else axs[col]

        sorted_tasks = sorted(all_tasks)
        times = [means.get(task, 0) / MILI_TO_NANO for task in sorted_tasks]

        init = ""
        for i in range(len(times)):
            if times[i] == 0:
                sorted_tasks[i] = init
                init += ' '
            else:
                ax.text(i, times[i], f"{times[i]:.2f}", ha='center', va='bottom', fontsize=10)

        ax.bar(sorted_tasks, times, color=color)
        ax.set_title(file_name)
        ax.set_ylabel("Time (ms)")
        ax.set_ylim(bottom=0, top=80)
        ax.set_xlabel("Task")
        ax.tick_params(axis='x', rotation=80)

        file_index += 1

    # Hide any empty subplot
    for i in range(file_index, num_rows*3):
        row = i // num_columns
        col = i % num_columns
        axs[row, col].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
