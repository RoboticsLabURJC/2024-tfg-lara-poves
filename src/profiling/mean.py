from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import argparse

MILI_TO_NANO = 10**6

def get_data(file:str):
    all_tasks = set() 
    data = defaultdict(dict)  

    if file.endswith(".txt"):
        with open(file, "r") as file:
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
            data[file] = means

    return all_tasks, data

def main(args):
    np.random.seed(4)

    all_tasks, data = get_data(args.file)

    _, means = list(data.items())[0]  

    sorted_tasks = sorted(all_tasks)
    times = [means.get(task, 0) / MILI_TO_NANO for task in sorted_tasks]

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_tasks, times, color='royalblue', alpha=0.7, edgecolor='black')

    for i, t in enumerate(times):
        if t > 0:
            plt.text(i, t, f"{t:.2f}", ha='center', va='bottom', fontsize=10)

    plt.ylabel("Time (ms)")
    plt.ylim(bottom=0, top=16)
    plt.xlabel("Task")
    plt.xticks(rotation=args.rot)
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a training on a specified Gym environment",
        usage="python3 %(prog)s --file <file_path> --rot <rot>"
    )
    parser.add_argument(
        '--file', 
        type=str, 
        required=True, 
        help='File to plot'
    )
    parser.add_argument(
        '--rot', 
        type=int, 
        required=False, 
        default=0,
        help='For the text'
    )

    main(parser.parse_args())
