import gymnasium as gym
from stable_baselines3 import DQN
import argparse

def main(args):
    model_dir = '/home/alumnos/lara/2024-tfg-lara-poves/src/gym/model/' + args.env.lower()
    model = DQN.load(model_dir + '/' + args.alg + '-' + args.env)
    env = gym.make(args.env, render_mode="human")

    '''
    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True) # _states: internal state of the model
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated: # the episode
            break
    '''

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
        description="Use a model on a specified Gym environment",
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
        '--n',
        type=int,
        default=1,
        help='Number of plays. Default is 1.'
    )

    main(parser.parse_args())