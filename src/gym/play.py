import gymnasium as gym
from stable_baselines3 import DQN, A2C, DDPG, TD3, SAC, PPO
import argparse
import matplotlib.pyplot as plt
import random
import math
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

alg_callable = {
    'DQN': (DQN, 'blue'),
    'A2C': (A2C, 'orange'),
    'DDPG': (DDPG, 'green'),
    'TD3': (TD3, 'red'),
    'SAC': (SAC, 'purple'),
    'PPO': (PPO, 'yellow')
}

possible_envs = [
    "CartPole-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "Acrobot-v1"
]

def main(args):
    seed = random.randint(0, 1000)

    if 'all' in args.alg:
        args.alg = list(alg_callable.keys())

    if 'all' in args.env:
        args.env = possible_envs

    rewards_envs = []
    for env_str in args.env:
        env = gym.make(env_str, render_mode="human")

        env_dir = env_str.split("-")[0]
        if "MountainCar" in env_str:
            env_dir = "MountainCar"

        rewards_models = []
        for alg in args.alg:
            try:
                model_dir = '/home/alumnos/lara/2024-tfg-lara-poves/src/gym/model/' + env_dir.lower()
                model_str = model_dir + '/' + alg + '-' + env_str
                model = alg_callable[alg][0].load(model_str)
                model.set_random_seed(seed)

                print("Algorithm:", alg + ',', "environment:", env_str)
            except FileNotFoundError:
                print("Model", env_dir.lower() + '/' + alg + '-' + env_str, "doesn't exit")
                continue

            total_reward = 0
            rewards = [total_reward]
            obs, info = env.reset()

            while True:
                # _states: internal state of the model
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                total_reward += reward
                rewards.append(total_reward)

                if terminated or truncated:
                    rewards_models.append((alg, rewards))
                    break

        if len(rewards_models) > 0:
            rewards_envs.append((env_str, rewards_models))
        env.close()

    # Plot results
    num_cols = 3
    num_plots = len(rewards_envs)
    if num_plots < num_cols:
        num_cols = num_plots
    elif num_plots % 2 == 0 and num_plots <= 4:
        num_cols = 2

    num_rows = math.ceil(len(rewards_envs) / num_cols)
    plt.figure(figsize=(5 * num_cols, 4 * num_rows))

    for i, (env, rewards_models) in enumerate(rewards_envs):
        plt.subplot(num_rows, num_cols, i + 1)
        
        reverse = False
        if rewards_models[0][1][-1] > 0:
            reverse = True
        rewards_models.sort(key=lambda x: x[1][-1], reverse=reverse)

        last_rewards = []
        offset = 40

        for alg, rewards in rewards_models:
            if rewards[-1] in last_rewards:
                x = len(rewards) - 1 - offset
            else:
                last_rewards.append(rewards[-1])
                x = len(rewards) - 1

            if rewards[-1] < 0:
                va = 'top'
                y = rewards[-1] - 1
            else:
                va = 'bottom'
                y = rewards[-1] + 1 

            plt.plot(range(len(rewards)), rewards, label=alg, color=alg_callable[alg][1])
            plt.text(x, y, f'{int(rewards[-1])}', ha='right', va=va, color=alg_callable[alg][1])

        plt.xlabel('Steps')
        plt.ylabel('Total Reward')
        plt.title(env)
        plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    possible_algs = list(alg_callable.keys())

    parser = argparse.ArgumentParser(
        description="Use a model on a specified Gym environment",
        usage="python3 %(prog)s --env {" + ",".join(possible_envs) + "} \
            [--alg {" + ",".join(possible_algs) + " | all}] [--plays <plays>]"
    )
    parser.add_argument(
        '--env', 
        type=str, 
        required=True, 
        default=['all'],
        choices=possible_envs + ['all'],
        help='The Gym environment ID. Possible values are: {' + ', '.join(possible_envs) + '}'
    )
    parser.add_argument(
        '--alg', 
        type=str, 
        required=True, 
        nargs='+',
        default=['all'],
        choices=possible_algs + ['all'],
        help='The algorithm(s) to use. Possible values are: {' + ', '.join(possible_algs) + 
            '} or "all" to run all algorithms. Default is "all".'
    )

    main(parser.parse_args())