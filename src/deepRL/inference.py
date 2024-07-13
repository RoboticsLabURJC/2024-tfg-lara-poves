from environment import CarlaDiscreteBasic
import argparse
from stable_baselines3 import DQN, A2C, DDPG, TD3, SAC, PPO
import os
import csv

alg_callable = {
    'DQN': DQN,
    'A2C': A2C,
    'DDPG': DDPG, 
    'TD3': TD3,
    'SAC': SAC,
    'PPO': PPO
}

env_callable = {
    'CarlaDiscreteBasic': CarlaDiscreteBasic
}

def main(args):
    alg_class = alg_callable[args.alg]
    env_class = env_callable[args.env]

    dir = '/home/lpoves/2024-tfg-lara-poves/src/deepRL/'
    model_file = dir + 'model/' + args.env + '/' + args.alg + '-' + args.env + '_' + args.n
    try:
        model = alg_class.load(model_file)
    except:
        print("Model", model_file, "doesn't exit")
        exit(1)

    env = env_class(train=False, port=args.port, human=True, normalize=True)
    obs, _ = env.reset()

    total_reward = 0
    step = 0

    dir_csv = dir + 'csv/inference/'
    if not os.path.exists(dir_csv):
        os.makedirs(dir_csv)
    dir_csv += args.env + '/'
    if not os.path.exists(dir_csv):
        os.makedirs(dir_csv)

    files = os.listdir(dir_csv)
    num_files = len(files) + 1
    file_csv = open(dir_csv + args.alg + '_' + 'data_' + args.n + '_' + str(num_files) +
                    '.csv', mode='w', newline='')
    writer_csv = csv.writer(file_csv)
    writer_csv.writerow(["Step", "Reward", "Accumulated reward", "Throttle", "Steer", "Deviation", "Speed"])
    
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            throttle, steer = env.action_to_control[action.item()]
            obs, reward, terminated, truncated, info = env.step(action)

            step += 1
            total_reward += reward
            writer_csv.writerow([step, reward, total_reward, throttle, steer, info['deviation'], 
                                 info['speed']])        

            if terminated or truncated:
                break
    except KeyboardInterrupt:
        return
    
    finally:
        env.close()

if __name__ == "__main__":
    possible_envs = [
        "CarlaDiscreteBasic"
    ]
    possible_algs = list(alg_callable.keys())

    parser = argparse.ArgumentParser(
        description="Execute an inference trial on a specified Gym environment",
        usage="python3 %(prog)s --env {" + ",".join(possible_envs) + \
            "} --alg {" + ",".join(possible_algs) + \
            "} --n <model_number> [--port <port_number>]"
    )
    parser.add_argument(
        '--env', 
        type=str, 
        required=True, 
        choices=possible_envs,
        help='Gym environment. Possible values are: {' + ', '.join(possible_envs) + '}'
    )
    parser.add_argument(
        '--alg', 
        type=str, 
        required=True, 
        choices=possible_algs,
        help='The algorithm to use. Possible values are: {' + ', '.join(possible_algs) + '}'
    )
    parser.add_argument(
        '--n', 
        type=str, 
        required=True, 
        help='Number of model'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        required=False, 
        default=6016,
        help='Port for Carla'
    )

    main(parser.parse_args())