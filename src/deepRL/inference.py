import environment
import argparse
from stable_baselines3 import DQN, A2C, DDPG, TD3, SAC, PPO
import os
import csv
import numpy as np

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

    env = env_class(train=False, port=args.port, human=True, normalize=True, num_cir=args.num_cir,
                    lane_network=args.lane_network, target_vel=args.target_vel, retrain=args.scene,
                    port_tm=args.port_tm)

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
    writer_csv.writerow([environment.KEY_STEPS, environment.KEY_REWARD, environment.KEY_ACC_REWARD, environment.KEY_VEL,
                         environment.KEY_THROTTLE, environment.KEY_STEER, environment.KEY_DEV, environment.KEY_DISTANCE,
                         environment.KEY_LASER_RIGHT_FRONT, environment.KEY_LASER_RIGHT, environment.KEY_LASER_RIGHT_BACK])
    
    obs, info = env.reset()

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            if args.alg == 'DQN':
                throttle, steer = env.action_to_control[action.item()]
            else:
                throttle, steer = action

            obs, reward, terminated, truncated, info = env.step(action)

            try:
                dist_front = info[environment.KEY_LASER]
            except KeyError:
                dist_front = np.nan

            print(f"Throttle: {throttle:.6f}\t||\tSteer: {steer:.7f}\t||\tDist front: {dist_front:.2f}")

            try:
                dist_right_front = info[environment.KEY_LASER_RIGHT_FRONT]
                dist_right = info[environment.KEY_LASER_RIGHT]
                dist_right_back = info[environment.KEY_LASER_RIGHT_BACK]

                print(f"Dist right front: {dist_right_front:.2f}\t||\tDist right: {dist_right:.2f}" 
                      f"\t||\tDist right back: {dist_right_back:.2f}\n")
            except KeyError:
                dist_right_front = environment.MAX_DIST_LASER
                dist_right = environment.MAX_DIST_LASER
                dist_right_back = environment.MAX_DIST_LASER

            step += 1
            total_reward += reward
            writer_csv.writerow([step, reward, total_reward, info[environment.KEY_VEL], throttle, steer, 
                                 info[environment.KEY_DEV], dist_front, dist_right_front, dist_right,
                                 dist_right_back])        

            if terminated or truncated:
                break
    except KeyboardInterrupt:
        return
    
    finally:
        env.close()

if __name__ == "__main__":
    possible_envs = list(env_callable.keys())
    possible_algs = list(alg_callable.keys())

    parser = argparse.ArgumentParser(
        description="Execute an inference trial on a specified Gym environment",
        usage="python3 %(prog)s --env {" + ",".join(possible_envs) + \
            "} --alg {" + ",".join(possible_algs) + \
            "} --n <model_number> [--port <port_number>] [--num_cir <num_cir>] [--port_tm <port_tm]"
            " [--lane_network <lane_network>] [--target_vel <target_vel>] [--scene <scene>]"
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
    parser.add_argument(
        '--num_cir', 
        type=int, 
        required=False, 
        default=0,
        choices=[0, 1, 2, 3, 4, 5, 6],
        help='Number of the circuit for the enviroment. By default 0.'
    )
    parser.add_argument(
        '--lane_network', 
        type=int, 
        required=False, 
        default=0,
        help='Detect the lane with the neuronal network instead of ground truth. By default 0 (0 = False).'
    )
    parser.add_argument(
        '--port_tm', 
        type=int, 
        required=False, 
        default=3456,
        help='Port for the traffic manger. By default 3456.'
    )
    parser.add_argument(
        '--target_vel', 
        type=float, 
        required=False, 
        default=-1.0,
        help='Velocity of the front vehicle, By default random velocities.'
    )
    parser.add_argument(
        '--scene', 
        type=int, 
        required=False, 
        default=2,
        choices=[0, 1, 2],
        help='Types of model you want to test: 0 = lane, 1 = obstacle, 2 = passing. By default 2.'
    )

    main(parser.parse_args())