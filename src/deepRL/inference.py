import pygame
import carla
import sys
import os
from stable_baselines3 import DQN, A2C, DDPG, TD3, SAC, PPO
import argparse

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, src_path)

import configcarla
from configcarla import SIZE_CAMERA

# Mapping str each constructor
alg_callable = {
    'DQN': DQN,
    'A2C': A2C,
    'DDPG': DDPG, 
    'TD3': TD3,
    'SAC': SAC,
    'PPO': PPO
}

def run_pid():
    screen = configcarla.setup_pygame(size=(SIZE_CAMERA * 2, SIZE_CAMERA), name='Follow lane: PID')
    world, _ = configcarla.setup_carla(name_world='Town04', port=2000, syn=False)

    map = world.get_map()

    print(map)

    # Add Ego Vehicle
    # Transform(Location(x=-341.586243, y=-75.615234, z=-0.004921), Rotation(pitch=0.102869, yaw=-30.943275, roll=-0.208588))

    transform = carla.Transform(carla.Location(x=-385, y=-3, z=0.5), carla.Rotation(yaw=-90.0))
    #transform = carla.Transform(carla.Location(x=17.047432, y=-18.00895, z=0.5), carla.Rotation(yaw=180.0))
    ego_vehicle = configcarla.add_one_vehicle(world=world, vehicle_type='vehicle.lincoln.mkz_2020',
                                              ego_vehicle=True, transform=transform)

    # Add sensors to Ego Vehicle
    sensors = configcarla.Vehicle_sensors(vehicle=ego_vehicle, world=world, screen=screen)

    driver_transform = carla.Transform(carla.Location(z=2.0, x=1.25), carla.Rotation(roll=90.0, pitch=-2.0))
    camera = sensors.add_camera_rgb(size_rect=(SIZE_CAMERA, SIZE_CAMERA), transform=driver_transform,
                                    seg=True, text='Driver view', init_extra=(SIZE_CAMERA, 0), 
                                    lane=True, canvas_seg=False)

    world_transform = carla.Transform(carla.Location(z=2.5, x=-4.75), carla.Rotation(roll=90.0))
    sensors.add_camera_rgb(size_rect=(SIZE_CAMERA, SIZE_CAMERA), init=(0, 0), 
                           transform=world_transform, text='World view')
    
    # Instance PID controller
    pid = configcarla.PID(ego_vehicle)
    spectator =  world.get_spectator()
    t = carla.Transform(transform.location, carla.Rotation(pitch=-90))
    spectator.set_transform(t)

    run = True
    try:
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

            sensors.update_data()
            #print(spectator.get_transform())
            #print(ego_vehicle.get_transform())
            
            # Control vehicle
            error_road = camera.get_deviation()
            pid.controll_vehicle(error_road)
            
    except KeyboardInterrupt:
        pass
    except AssertionError:
        pass

    sensors.destroy()
    pygame.quit()

def main(args):
    if args.env == "PID" and args.alg == "PID":
        run_pid()
        return
    elif args.env != "PID" and args.alg != "PID":
        print("env - model")
    else:
        print("If mode is 'PID', both parameters must be 'PID'")
        return

if __name__ == "__main__":
    possible_envs = [
        "PID",
        "CarlaDiscreteBasic",
    ]
    possible_algs = list(alg_callable.keys())
    possible_algs.append("PID")

    parser = argparse.ArgumentParser(
        description="Run a training on a specified Gym environment",
        usage="python3 %(prog)s --env {" + ",".join(possible_envs) + \
            "} --alg {" + ",".join(possible_algs)
    )
    parser.add_argument(
        '--env', 
        type=str, 
        required=True, 
        choices=possible_envs,
        help='The Gym environment ID or PID controll. Possible values are: {' + ', '.join(possible_envs) + '}'
    )
    parser.add_argument(
        '--alg', 
        type=str, 
        required=True, 
        choices=possible_algs,
        help='The algorithm to use or PID controll. Possible values are: {' + ', '.join(possible_algs) + '}'
    )

    main(parser.parse_args())