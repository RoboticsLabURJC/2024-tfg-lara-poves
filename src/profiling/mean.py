from collections import defaultdict

MILI_TO_NANO = 10**6

sum = defaultdict(int)
count = defaultdict(int)

with open("times_autopilot.txt", "r") as file:
    for line in file:
        task, time_ns = line.strip().split(": ")
        time_ns = int(time_ns.split()[0])  
        sum[task] += time_ns
        count[task] += 1

means = {task: sum[task] / count[task] for task in sum}

for task, mean in means.items():
    print(f"Mean of {task}: {mean / MILI_TO_NANO:.2f} ms")
