import gymnasium as gym
from stable_baselines3 import DQN
import argparse
import os
import yaml

# Visualize train: tensorboard --logdir=log_dir

def main(args):
    # Mapping str with constructor
    alg_callable = {
        'DQN': DQN
    }

    log_dir = '/home/alumnos/lara/2024-tfg-lara-poves/src/gym/log/' + args.env.lower()
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    model_dir = '/home/alumnos/lara/2024-tfg-lara-poves/src/gym/model' + args.env.lower()
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Create env
    env = gym.make(args.env, render_mode="human")

    # Algorithm
    alg = alg_callable[args.alg]

    # Log parameters
    log_interval = args.log_interval
    log_name = args.alg + '-' + args.env

    # Get hyperparams
    config_path = "/home/alumnos/lara/2024-tfg-lara-poves/src/gym/config/" + args.alg.lower() + ".yml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    model_params = config[args.env]

    # Extract some parameters
    n_timesteps = model_params['n_timesteps']
    model_params.pop('n_timesteps', None)
    policy = model_params['policy']
    model_params.pop('policy', None)

    # Create, train and save the model
    model = alg(policy, env, verbose=1, tensorboard_log=log_dir, **model_params)
    model.set_random_seed(6)
    model.learn(total_timesteps=n_timesteps, log_interval=log_interval, tb_log_name=log_name, progress_bar=True)
    model.save(model_dir + '/' + args.alg + '-' + args.env)

if __name__ == "__main__":
    possible_envs = [
        "CartPole-v1",
        "MountainCar-v0",
        "Acrobot-v1",
        "atari"
        "LunarLander-v2",
    ]
    possible_algs = [
        "DQN"
    ]

    parser = argparse.ArgumentParser(
        description="Run a training on a specified Gym environment",
        usage="python3 %(prog)s --env {" + ",".join(possible_envs) + "} --alg {" + ",".join(possible_algs) + "} [--log_interval <log_interval>]"
    )
    parser.add_argument(
        '--env', 
        type=str, 
        required=True, 
        choices=possible_envs,
        help='The Gym environment ID. Possible values are: {' + ', '.join(possible_envs) + '}'
    )
    parser.add_argument(
        '--alg', 
        type=str, 
        required=True, 
        choices=possible_algs,
        help='The algorithm to use. Possible values are: {' + ', '.join(possible_algs) + '}'
    )
    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='Number of episodes before logging information. Default is 1.'
    )

    main(parser.parse_args())