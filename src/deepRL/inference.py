import environment
import argparse
from stable_baselines3 import DQN, A2C, DDPG, TD3, SAC, PPO
import os
import csv

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
    'CarlaPassing': environment.CarlaPassing
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
                    lane_network=args.lane_network, target_vel=args.target_vel)

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
    writer_csv.writerow([environment.KEY_STEPS, environment.KEY_REWARD, environment.KEY_ACC_REWARD, 
                         environment.KEY_THROTTLE, environment.KEY_STEER, environment.KEY_DEV,
                         environment.KEY_VEL, environment.KEY_BRAKE, environment.KEY_DISTANCE,
                         environment.KEY_BACK])
    
    brake = -1.0
    obs, info = env.reset()

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            if args.alg == 'DQN':
                throttle, steer = env.action_to_control[action.item()]
            else:
                throttle, steer = action
                print(f"Throttle: {throttle:.6f} \t||\tSteer: {steer:.7f}", end="")

            obs, reward, terminated, truncated, info = env.step(action)

            try:
                dist = info[environment.KEY_LASER]
                print(f"\t||\tDist front: {dist:.6f}")
            except KeyError:
                dist = environment.MAX_DIST_LASER
                print()

            try:
                dist_back = info[environment.KEY_BACK]
                print("dist back:", dist_back) # quitar
            except KeyError:
                dist_back = environment.MAX_DIST_LASER

            step += 1
            total_reward += reward
            writer_csv.writerow([step, reward, total_reward, throttle, steer, info[environment.KEY_DEV], 
                                 info[environment.KEY_VEL], brake, dist, dist_back])        

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
            "} --n <model_number> [--port <port_number>] [--num_cir <num_cir>] [--port_tm <port_tm] [--lane_network <lane_network>]"
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
        choices=[0, 1, 2, 3, 4, 5],
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
        default=6,
        help='Velocity in m/s of the front vehicle. By default 6m/s.'
    )

    main(parser.parse_args())