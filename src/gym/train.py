import gymnasium as gym
from stable_baselines3 import DQN, A2C, DDPG, TD3, SAC, PPO
import argparse
import os
import yaml
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise
import numpy as np

# Visualize train: tensorboard --logdir=log_dir

SEED = 6

# Mapping str alg each constructor
alg_callable = {
    'DQN': DQN,
    'A2C': A2C,
    'DDPG': DDPG, 
    'TD3': TD3,
    'SAC': SAC,
    'PPO': PPO
}

# Mapping str noise each constructor
noise_callable = {
    'ornstein-uhlenbeck': OrnsteinUhlenbeckActionNoise,
    'normal': NormalActionNoise
}

def check_dir(dir:str, env:str):
    if not os.path.exists(dir):
        os.makedirs(dir)

    dir = dir + env.lower()
    if not os.path.exists(dir):
        os.makedirs(dir)

    return dir

class Linear_decay:
    def __init__(self, initial:float):
        self.initial = initial

    def linear_decay(self, progress:float):
        return self.initial * progress

def main(args):
    env_dir = args.env.split("-")[0]
    if "MountainCar" in args.env:
        env_dir = "MountainCar"

    gym_dir = '/home/alumnos/lara/2024-tfg-lara-poves/src/gym/'
    log_dir = check_dir(gym_dir + 'log/', env_dir)
    model_dir = check_dir(gym_dir + 'model/', env_dir)

    alg = alg_callable[args.alg]
    log_name = args.alg + '-' + args.env + '-' + str(args.log_interval)

    # Get hyperparams
    config_path = gym_dir + "config/" + args.alg.lower() + ".yml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    try:
        model_params = config[args.env]
    except KeyError:
        print("Algorithm", args.alg, "is not available for environment", args.env)
        return
    
    # Extract compulsory params
    n_timesteps = model_params['n_timesteps']
    model_params.pop('n_timesteps', None)
    policy = model_params['policy']
    model_params.pop('policy', None)

    # Extrat optional params
    try:
        n_envs = model_params['n_envs']
        model_params.pop('n_envs', None)
    except KeyError:
        n_envs = 1
    try:
        normalize = model_params['normalize']
        model_params.pop('normalize', None)
    except KeyError:
        normalize = False

    # Create env
    def make_env(**kwargs) -> gym.Env:
        spec = gym.spec(args.env)
        return spec.make(**kwargs)  
    env = make_vec_env(make_env, n_envs=n_envs, seed=SEED)
    env = VecNormalize(env, norm_obs=normalize, norm_reward=normalize)

    # For DDPG
    noise_type = None
    try:
        noise_type = model_params['noise_type']
        model_params.pop('noise_type', None)
    except KeyError:
        pass

    if noise_type != None:
        try:
            noise_std = model_params['noise_std']
            model_params.pop('noise_std', None)

            n_actions = env.action_space.shape[-1]
            action_noise = noise_callable[noise_type](sigma=noise_std, mean=np.zeros(n_actions))
            model_params['action_noise'] = action_noise # Add to model params
        except Exception:
            print("Can't include noise", noise_type)

    # Check if linear decay
    if isinstance(model_params['learning_rate'], str) and 'lin' in model_params['learning_rate']:
        init_val = float(model_params['learning_rate'].split('_')[-1])
        lr = Linear_decay(init_val)
        model_params['learning_rate'] = lr.linear_decay

    try:
        if isinstance(model_params['clip_range'], str) and 'lin' in model_params['clip_range']:
            init_val = float(model_params['clip_range'].split('_')[-1])
            cr = Linear_decay(init_val)
            model_params['clip_range'] = cr.linear_decay
    except KeyError:
        pass

    # Create, train and save the model
    model = alg(policy, env, verbose=1, seed=SEED, tensorboard_log=log_dir, **model_params)
    model.learn(total_timesteps=n_timesteps, log_interval=args.log_interval, 
                tb_log_name=log_name, progress_bar=True)
    cont_save = '/' + args.alg + '-' + args.env
    model.save(model_dir + cont_save)

    if normalize:
        # Save the running average, for testing the agent we need that normalization
        vec_normalize = model.get_vec_normalize_env()
        assert vec_normalize is not None
        vec_dir = check_dir(gym_dir + '/vecnormalize/', env_dir)
        vec_normalize.save(vec_dir + cont_save + ".pkl")

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
        '--log_interval',
        type=int,
        default=1,
        help='Number of episodes before logging information. Default is 1.'
    )

    main(parser.parse_args())
