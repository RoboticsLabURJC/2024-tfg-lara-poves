from environment import CarlaDiscreteBasic, CarlaContinuousBasic
import argparse
from stable_baselines3 import DQN, A2C, DDPG, TD3, SAC, PPO
import os
import yaml
from stable_baselines3.common.env_util import make_vec_env

SEED = 6

alg_callable = {
    'DQN': DQN,
    'A2C': A2C,
    'DDPG': DDPG, 
    'TD3': TD3,
    'SAC': SAC,
    'PPO': PPO
}

env_callable = {
    'CarlaDiscreteBasic': CarlaDiscreteBasic,
    'CarlaContinuousBasic': CarlaContinuousBasic
}

def check_dir(dir:str, env:str):
    if not os.path.exists(dir):
        os.makedirs(dir)

    dir = dir + env
    if not os.path.exists(dir):
        os.makedirs(dir)

    return dir

def main(args):
    # Get hyperparams
    config_path = '/home/lpoves/2024-tfg-lara-poves/src/deepRL/' + 'config/' + args.env + '.yml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    try:
        model_params = config[args.alg]
    except KeyError:
        print("Algorithm", args.alg, "is not available for environment", args.env)
        exit(1)

    # Extract compulsory params
    n_timesteps = model_params['n_timesteps']
    model_params.pop('n_timesteps', None)
    policy = model_params['policy']
    model_params.pop('policy', None)

    alg_class = alg_callable[args.alg]
    env_class = env_callable[args.env]

    dir = '/home/lpoves/2024-tfg-lara-poves/src/deepRL/'
    log_dir = check_dir(dir + 'log/', args.env)
    model_dir = check_dir(dir + 'model/', args.env)

    log_name = args.alg + '-' + args.env
    env = env_class(train=True, fixed_delta_seconds=args.delta, human=False, port=args.port, 
                    alg=args.alg, normalize=True, seed=SEED)
    
    if args.alg != 'DQN':
        env = make_vec_env(lambda: env_class(train=True, fixed_delta_seconds=args.delta, human=False,
                                             port=args.port, alg=args.alg, normalize=True, seed=SEED), n_envs=1)
    
    model = alg_class(policy, env, verbose=1, seed=SEED, tensorboard_log=log_dir, **model_params)
    if args.alg == 'DQN':
        env.set_model(model)
    model.learn(total_timesteps=n_timesteps, log_interval=1, tb_log_name=log_name, progress_bar=True)
    
    files = os.listdir(dir + 'model/' + args.env)
    num_files = len(files) + 1
    model.save(model_dir + '/' + args.alg + '-' + args.env + '_' + str(num_files))

if __name__ == "__main__":
    possible_envs = list(env_callable.keys())
    possible_algs = list(alg_callable.keys())

    parser = argparse.ArgumentParser(
        description="Run a training on a specified Gym environment",
        usage="python3 %(prog)s --env {" + ",".join(possible_envs) + \
            "} --alg {" + ",".join(possible_algs) + "} [--port <port_number>]" +\
            "[--delta <fixed_delta_seconds>]"
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
        '--port', 
        type=int, 
        required=False, 
        default=6016,
        help='Port for Carla'
    )
    parser.add_argument(
        '--delta', 
        type=float, 
        required=False, 
        default=0.1,
        help='Fixed delta second for CARLA simulator. By default 0.1s'
    )

    main(parser.parse_args())