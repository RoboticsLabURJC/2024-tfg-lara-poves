import environment
import argparse
from stable_baselines3 import DQN, A2C, DDPG, TD3, SAC, PPO
import os
import yaml
from stable_baselines3.common.env_util import make_vec_env
import sys
import warnings
import traceback
from stable_baselines3.common.callbacks import CheckpointCallback

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, src_path)

from configcarla import PATH

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
    'CarlaLaneDiscrete': environment.CarlaLaneDiscrete,
    'CarlaLaneContinuous': environment.CarlaLaneContinuous,
    'CarlaObstacle': environment.CarlaObstacle, 
    'CarlaPassing': environment.CarlaPassing,
    'CarlaOvertaken': environment.CarlaOvertaken
}

def check_dir(dir:str, env:str):
    if not os.path.exists(dir):
        os.makedirs(dir)

    dir = dir + env
    if not os.path.exists(dir):
        os.makedirs(dir)

    return dir

def main(args):
    alg_class = alg_callable[args.alg]
    env_class = env_callable[args.env]

    dir = PATH + '2024-tfg-lara-poves/src/deepRL/'
    log_dir = check_dir(dir + 'log/', args.env)
    model_dir = check_dir(dir + 'model/', args.env)
    log_name = args.alg + '-' + args.env

    if env_class == environment.CarlaPassing and args.delta < 0.1:
        warnings.warn(f"Fixed delta seconds should be ≤ 100ms (10 FPS) for the environment {args.env}",
                      UserWarning)

    if args.alg != 'DQN':
        env = make_vec_env(lambda: env_class(train=True, fixed_delta_seconds=args.delta, human=args.human,
                                             retrain=args.retrain, port=args.port, alg=args.alg, 
                                             normalize=True, seed=SEED, port_tm=args.port_tm), n_envs=1)
    else:
        env = env_class(train=True, fixed_delta_seconds=args.delta, human=args.human, port=args.port,
                        alg=args.alg, normalize=True, seed=SEED, retrain=args.retrain)

    # Get hyperparams
    config_path = PATH + '2024-tfg-lara-poves/src/deepRL/' + 'config/' + args.env + '.yml'
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

    if env_class == environment.CarlaOvertaken:
        model_params['policy_kwargs'] = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))

    if not args.retrain:
        model = alg_class(policy, env, verbose=args.verbose, seed=SEED, tensorboard_log=log_dir, **model_params)
    else:
        dir = '/home/lpoves/2024-tfg-lara-poves/src/deepRL/'
        if args.env == 'CarlaPassing':
            model_file = dir + 'model/CarlaBaseModels/' + args.alg + '-' + args.env + '_' + str(args.retrain)
        else:
            model_file = dir + 'model/CarlaBaseModels/' + args.alg + '-' + args.env

        try:
            model = alg_class.load(model_file, env=env, **model_params)
        except FileNotFoundError:
            print("Model", model_file, "doesn't exit")
            exit(1)

    if args.alg == 'DQN':
        env.set_model(model)

    # Save a checkpoint every 20_000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=20_000,
        save_path=model_dir,
        name_prefix=args.alg + '-' + args.env + '_' + str(args.num_file),
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    model = alg_class.load(dir + 'model/CarlaOvertaken/' + args.alg + '-' + args.env + '_7', env=env, tensorboard_log=log_dir, **model_params)

    model.learn(total_timesteps=n_timesteps, log_interval=args.log_interval, tb_log_name=log_name,
                progress_bar=True, callback=checkpoint_callback)
    model.save(model_dir + '/' + args.alg + '-' + args.env + '_' + str(args.num_file))
    env.close()

if __name__ == "__main__":
    possible_envs = list(env_callable.keys())
    possible_algs = list(alg_callable.keys())

    parser = argparse.ArgumentParser(
        description="Run a training on a specified Gym environment",
        usage="python3 %(prog)s --env {" + ",".join(possible_envs) + \
            "} --alg {" + ",".join(possible_algs) + "} [--port <port_number>] [--human <human>]" +\
            "[--delta <fixed_delta_seconds>] [--log_interval <log_interval>] [--verbose <verbose>]" +\
            "[--num_cir <num_cir>] [--retrain <retrain>] [--port_tm <port_tm>] [--num_file <num_file>]"
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
        default=0.05,
        help='Fixed delta second for CARLA simulator. By default 50ms.'
    )
    parser.add_argument(
        '--log_interval', 
        type=int, 
        required=False, 
        default=64,
        help='Logging interval for traing. By default 64.'
    )
    parser.add_argument(
        '--verbose', 
        type=int, 
        required=False, 
        default=0,
        choices=[0, 1, 2],
        help='Show basic (1) or detailed (2) training information, or hide it (0). Default is 0.'
    )
    parser.add_argument(
        '--human', 
        type=int, 
        required=False, 
        default=0,
        help='Display or not Pygame screen. By default 0 (0 = False).'
    )
    parser.add_argument(
        '--retrain', 
        type=int, 
        required=False, 
        default=0,
        help='If a model has to be reatrained, 1: retraining lane, 2: retraining obstacle. By default 0 (0 = False).'
    )
    parser.add_argument(
        '--port_tm', 
        type=int, 
        required=False, 
        default=3456,
        help='Port for the traffic manger. By default 3456.'
    )
    parser.add_argument(
        '--num_file', 
        type=int, 
        required=False, 
        default=1,
        help='Number which identifies the model.'
    )

    main(parser.parse_args())