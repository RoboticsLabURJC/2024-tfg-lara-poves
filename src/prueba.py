import pygame
import carla
import configcarla
import argparse

# Screen
HEIGHT= 450
WIDTH = 450
DECREASE = 50
    
def main(args):
    # Setup 
    world, client = configcarla.setup_carla(name_world='Town04', fixed_delta_seconds=0.05, syn=True, 
                                            port=args.port)
    screen = configcarla.setup_pygame(size=(WIDTH * 3 - DECREASE * 2, HEIGHT * 2), name='Autopilot')

    # Add Ego Vehicle
    ego_transform = carla.Transform(carla.Location(x=347.65, y=-355, z=0.1), carla.Rotation(yaw=-131))
    ego_vehicle = configcarla.add_one_vehicle(world=world, ego_vehicle=True, transform=ego_transform,
                                              vehicle_type='vehicle.carlamotors.carlacola')

    # Add sensors to Ego Vehicle
    sensors = configcarla.Vehicle_sensors(vehicle=ego_vehicle, world=world, screen=screen)

    driver_transform = carla.Transform(carla.Location(z=2.0, x=3), carla.Rotation(pitch=-2.0))
    sensors.add_camera_rgb(size_rect=(WIDTH, HEIGHT), transform=driver_transform,
                           seg=False, init_extra=(0, 0), text='Driver view')
    
    lidar_transform = carla.Transform(carla.Location(x=-0.5, z=1.8), carla.Rotation(yaw=90.0))
    sensors.add_lidar(size_rect=((WIDTH - DECREASE) * 2, HEIGHT * 2), init=(WIDTH, 0), scale=38,
                      transform=lidar_transform, show_stats=True) 
    
    world_transform = carla.Transform(carla.Location(z=2.5, x=-4.75))
    sensors.add_camera_rgb(size_rect=(WIDTH, HEIGHT), init=(0, HEIGHT),
                           transform=world_transform, text='World view')

    # Add more vehicles
    configcarla.traffic_manager(client=client, vehicles=[ego_vehicle])
    
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
           
            world.tick()

            vel = carla.Vector3D(ego_vehicle.get_velocity()).length()
            print(ego_vehicle.get_velocity())
            target_vel = 2.0
            if vel > target_vel:
                ego_vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))

            sensors.update_data()

    except KeyboardInterrupt:
        return

    finally:
        sensors.destroy()
        pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execute an inference trial on a specified Gym environment",
        usage="python3 %(prog)s --port <port_number>"
    )
    parser.add_argument(
        '--port', 
        type=int, 
        required=False, 
        default=2000,
        help='Port for Carla'
    )

    main(parser.parse_args())