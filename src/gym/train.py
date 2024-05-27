import gymnasium as gym
from stable_baselines3 import DQN, A2C, DDPG, TD3, SAC, PPO
import argparse
import os
import yaml
from stable_baselines3.common.vec_env import DummyVecEnv

# Visualize train: tensorboard --logdir=log_dir

SEED = 6

# Mapping str each constructor
alg_callable = {
    'DQN': DQN,
    'A2C': A2C,
    'DDPG': DDPG, 
    'TD3': TD3,
    'SAC': SAC,
    'PPO': PPO
}

def main(args):
    env_dir = args.env.split("-")[0]
    if "MountainCar" in args.env:
        env_dir = "MountainCar"

    log_dir = '/home/alumnos/lara/2024-tfg-lara-poves/src/gym/log/' + env_dir.lower()
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    model_dir = '/home/alumnos/lara/2024-tfg-lara-poves/src/gym/model/' + env_dir.lower()
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    alg = alg_callable[args.alg]
    log_name = args.alg + '-' + args.env + '-' + str(args.log_interval)

    # Get hyperparams
    config_path = "/home/alumnos/lara/2024-tfg-lara-poves/src/gym/config/" + args.alg.lower() + ".yml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    try:
        model_params = config[args.env]
    except KeyError:
        print("Algorithm", args.alg, "is not available for environment", args.env)
        return
    
    # Extract some parameters
    n_timesteps = model_params['n_timesteps']
    model_params.pop('n_timesteps', None)
    policy = model_params['policy']
    model_params.pop('policy', None)

    # Create env
    try:
        n_envs = model_params['n_envs']
        model_params.pop('n_envs', None)
        norm = model_params['normalize']
        model_params.pop('normalize', None)

        envs = [lambda: gym.make(args.env) for _ in range(n_envs)]
        env = DummyVecEnv(envs) # Vector of envs
    except KeyError:
        env = gym.make(args.env, render_mode="human")

    # Create, train and save the model
    model = alg(policy, env, verbose=1, seed=SEED, tensorboard_log=log_dir, **model_params)
    model.set_random_seed(SEED)
    model.learn(total_timesteps=n_timesteps, log_interval=args.log_interval, 
                tb_log_name=log_name, progress_bar=True)
    model.save(model_dir + '/' + args.alg + '-' + args.env)

if __name__ == "__main__":
    possible_envs = [
        "CartPole-v1",
        "MountainCar-v0",
        "MountainCarContinuous-v0",
        "Acrobot-v1"
    ]
    possible_algs = list(alg_callable.keys())

    parser = argparse.ArgumentParser(
        description="Run a training on a specified Gym environment",
        usage="python3 %(prog)s --env {" + ",".join(possible_envs) + \
            "} --alg {" + ",".join(possible_algs) + "} [--log_interval <log_interval>]"
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