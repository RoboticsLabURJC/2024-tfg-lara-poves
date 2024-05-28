import gymnasium as gym
from stable_baselines3 import DQN, A2C, DDPG, TD3, SAC, PPO
import argparse
import matplotlib.pyplot as plt
import random
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

alg_callable = {
    'DQN': (DQN, 'red'),
    'A2C': (A2C, 'lightgreen', 'green'),
    'DDPG': (DDPG, 'orange', 'lightcoral'),
    'TD3': (TD3, 'blue', 'cornflowerblue'),
    'SAC': (SAC, 'violet', 'purple'),
    'PPO': (PPO, 'gray', 'black')
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
    elif  not isinstance(args.alg, list):
        args.alg = [args.alg]

    if 'all' in args.env:
        args.env = possible_envs
    elif not isinstance(args.env, list):
        args.env = [args.env]

    rewards_envs = []
    for env_str in args.env:
        env = gym.make(env_str, render_mode="human")

        env_dir = env_str.split("-")[0]
        if "MountainCar" in env_str:
            env_dir = "MountainCar"

        rewards_models = []
        for alg in args.alg:
            gym_dir = '/home/alumnos/lara/2024-tfg-lara-poves/src/gym/'
            cont_load =  env_dir.lower() + '/' + alg + '-' + env_str

            # Load model
            try:
                model = alg_callable[alg][0].load(gym_dir + 'model/' + cont_load)
                model.set_random_seed(seed)
            except FileNotFoundError:
                if alg != 'DQN' and env_dir != 'MountainCarContinuous-v0':
                    print("Model", cont_load + '.zip', "doesn't exit")
                continue

            try:
                # Create env
                def make_env(**kwargs) -> gym.Env:
                    spec = gym.spec(env_str)
                    return spec.make(**kwargs)  
                env = make_vec_env(make_env, seed=seed)

                # Load vecnormalize
                norm = alg != 'DQN' and env_str != 'CartPole-v1'
                if norm:
                    vec_normalize = VecNormalize.load(gym_dir + 'vecnormalize/' + cont_load + '.pkl', venv=env)
            except FileNotFoundError: 
                print("Vector normalize", cont_load + '.pkl', "doesn't exit")
                continue

            print("Algorithm:", alg + ',', "environment:", env_str)
            obs = env.reset()
            done = False
            total_reward = 0.0
            rewards = [total_reward]
            
            # Simulate a play
            while not done:
                if norm:
                    obs = vec_normalize.normalize_obs(obs)

                # Predict and execute action
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)

                # Keep reward
                total_reward += reward[0]
                rewards.append(total_reward)

            rewards_models.append((alg, rewards))

        if len(rewards_models) > 0:
            rewards_envs.append((env_str, rewards_models))
    
    # Plot results
    num_cols = 3
    num_rows = 1
    plt.figure(figsize=(6 * num_cols, 5 * num_rows))

    for env, rewards_models in rewards_envs:
        # Sort rewards to see all plots
        reverse = False
        if rewards_models[0][1][-1] > 0:
            reverse = True
        rewards_models.sort(key=lambda x: x[1][-1], reverse=reverse)

        # Select graph
        if 'CartPole' in env:
            i = 1
        elif 'Acrobot' in env:
            i = 3
        else:
            i = 2
        plt.subplot(num_rows, num_cols, i)

        last_rewards = []
        offset = 10

        for alg, rewards in rewards_models:
            if any(abs(reward - rewards[-1]) <= offset for reward in last_rewards):
                x = len(rewards) - 1 - offset
            else:
                last_rewards.append(rewards[-1])
                x = len(rewards) - 1

            if rewards[-1] < 0:
                y = rewards[-1] - 1
                va = 'top'
            else:
                y = rewards[-1] + 1 
                va = 'bottom'

            if 'Continuous' in env:
                last_reward = round(rewards[-1], 2)
                color = alg_callable[alg][2]
                label = alg + ' ' + 'cont'
            else:
                last_reward = int(rewards[-1])
                color = alg_callable[alg][1]
                label = alg

            plt.plot(range(len(rewards)), rewards, label=label, color=color)
            plt.text(x, y, f'{last_reward}', ha='right', va=va, color=color, fontsize=11)

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