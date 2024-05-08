from collections import defaultdict
import argparse

MILI_TO_NANO = 10**6

def main(file):
    sum = defaultdict(int)
    count = defaultdict(int)

    with open(file, "r") as file:
        for line in file:
            task, time_ns = line.strip().split(": ")
            time_ns = int(time_ns.split()[0])  
            sum[task] += time_ns
            count[task] += 1

    means = {task: sum[task] / count[task] for task in sum}

    for task, mean in means.items():
        print(f"{task}: {mean / MILI_TO_NANO:.2f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate mean times of tasks from a file.")
    parser.add_argument("--file", required=True, help="Input file containing task times.")
    args = parser.parse_args()
    main(args.file)