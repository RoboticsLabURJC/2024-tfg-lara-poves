import gymnasium as gym
from stable_baselines3 import DQN, A2C, DDPG, TD3, SAC, PPO
import argparse
import matplotlib.pyplot as plt
import random
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
import math

alg_callable = {
    'DQN': (DQN, 'yellow'),
    'A2C': (A2C, 'green'),
    'PPO': (PPO, 'red'),
    'DDPG': (DDPG, 'orange'),
    'TD3': (TD3, 'blue'),
    'SAC': (SAC, 'violet')
}

possible_envs = [
    "CartPole-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "Acrobot-v1"
]

def main(args):
    seed = random.randint(0, 1000)
    all = False

    if 'all' in args.alg:
        args.alg = list(alg_callable.keys())
        all = True
    elif not isinstance(args.alg, list):
        args.alg = [args.alg]

    if 'all' in args.env:
        args.env = possible_envs
        all = True
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
            gym_dir = '/home/lpoves/2024-tfg-lara-poves/src/gym/'
            cont_load =  env_dir.lower() + '/' + alg + '-' + env_str

            # Load model
            try:
                model = alg_callable[alg][0].load(gym_dir + 'model/' + cont_load)
                model.set_random_seed(seed)
            except FileNotFoundError:
                if all and ((alg == 'DQN' and env_str == 'MountainCarContinuous-v0') or 
                            ((alg == 'DDPG' or alg == 'TD3' or alg == 'SAC') 
                            and env_str != 'MountainCarContinuous-v0')):
                    pass
                else:
                    print("Model", cont_load + '.zip', "doesn't exit")
                continue

            try:
                # Create env
                def make_env(**kwargs) -> gym.Env:
                    spec = gym.spec(env_str)
                    return spec.make(**kwargs)  
                env = make_vec_env(make_env, seed=seed)

                # Load vecnormalize
                norm = alg != 'DQN' and env_str != 'CartPole-v1' and alg != 'DDPG'
                if norm and alg != 'TD3' and alg != 'SAC':
                    vec_norm = VecNormalize.load(gym_dir + 'vecnormalize/' + cont_load + '.pkl', venv=env)
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
                    obs = vec_norm.normalize_obs(obs)

                # Predict and execute action
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)

                # Keep reward
                total_reward += reward[0]
                rewards.append(total_reward)

            rewards_models.append((alg, rewards))

        if len(rewards_models) > 0:
            rewards_envs.append((env_str, rewards_models))

    if len(rewards_envs) == 0:
        return
    
    # Plot results
    num_cols = 1 if len(rewards_envs) == 1 else 2
    num_rows = math.ceil(len(rewards_envs) / 2)
    plt.figure(figsize=(6 * num_cols, 5 * num_rows))

    # Obtain the same x limite for MountainCar envs
    mountain = [r for r in rewards_envs if 'Mountain' in r[0]]
    max_mountain = 0
    for m in mountain:
        for i in m[1]:
            len_mountain = len(i[1])
            if len_mountain > max_mountain:
                max_mountain = len_mountain

    subplot = 0
    for env, rewards_models in rewards_envs:
        # Sort rewards to see all plots
        reverse = False
        if rewards_models[0][1][-1] > 0:
            reverse = True
        rewards_models.sort(key=lambda x: x[1][-1], reverse=reverse)

        # Select graph
        subplot += 1
        plt.subplot(num_rows, num_cols, subplot)

        last_rewards = []
        for alg, rewards in rewards_models:
            found = False
            for i, (reward, count) in enumerate(last_rewards):
                if abs(reward - rewards[-1]) <= 10:
                    offset = len(rewards) / 9
                    if env == 'MountainCarContinuous-v0' and max_mountain > 200:
                        offset = int(-93 * 1.2)

                    x = len(rewards) - 1 - offset * count
                    last_rewards[i] = (reward, count + 1)
                    found = True
            
            if not found:
                last_rewards.append((rewards[-1], 1))
                x = len(rewards) - 1

            if rewards[-1] < 0:
                y = rewards[-1] - 1
                va = 'top'
            else:
                y = rewards[-1] + 1 
                va = 'bottom'

            if 'Continuous' in env:
                last_reward = round(rewards[-1], 2)
            else:
                last_reward = int(rewards[-1])

            plt.plot(range(len(rewards)), rewards, label=alg, color=alg_callable[alg][1])
            plt.text(x, y, f'{last_reward}', ha='right', va=va, color=alg_callable[alg][1],
                     fontsize=11)
            if 'Mountain' in env:
                plt.xlim(0, max_mountain)

        plt.xlabel('Steps')
        plt.ylabel('Total Reward')
        plt.title(env)
        plt.legend()

    #plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    possible_algs = list(alg_callable.keys())

    parser = argparse.ArgumentParser(
        description="Use a model on a specified Gym environment",
        usage="python3 %(prog)s --env {" + ",".join(possible_envs) + "} \
            [--alg {" + ",".join(possible_algs) + " | all}]"
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