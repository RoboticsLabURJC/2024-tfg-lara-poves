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
    world, client = configcarla.setup_carla(name_world='Town05', fixed_delta_seconds=0.05, syn=True, 
                                            port=args.port)
    screen = configcarla.setup_pygame(size=(WIDTH * 3 - DECREASE * 2, HEIGHT * 2), name='Autopilot')

    # Add Ego Vehicle
    ego_transform = carla.Transform(carla.Location(x=151.5, y=-90.0, z=2.5), carla.Rotation(yaw=90.0))
    ego_vehicle = configcarla.add_one_vehicle(world=world, ego_vehicle=True, transform=ego_transform,
                                              vehicle_type='vehicle.lincoln.mkz_2020')

    # Add sensors to Ego Vehicle
    sensors = configcarla.Vehicle_sensors(vehicle=ego_vehicle, world=world, screen=screen)

    driver_transform = carla.Transform(carla.Location(z=2.0, x=1.25), carla.Rotation(roll=90.0, pitch=-2.0))
    sensors.add_camera_rgb(size_rect=(WIDTH, HEIGHT), transform=driver_transform,
                           seg=True, init_extra=(0, 0), text='Driver view')
    
    lidar_transform = carla.Transform(carla.Location(x=-0.5, z=1.8), carla.Rotation(yaw=90.0))
    sensors.add_lidar(size_rect=((WIDTH - DECREASE) * 2, HEIGHT * 2), init=(WIDTH, 0), scale=38,
                      transform=lidar_transform, show_stats=True) 
    
    world_transform = carla.Transform(carla.Location(z=2.5, x=-4.75), carla.Rotation(roll=90.0))
    sensors.add_camera_rgb(size_rect=(WIDTH, HEIGHT), init=(0, HEIGHT),
                           transform=world_transform, text='World view')
    
    # Add a car in front of Ego Vehicle
    ego_transform.location.y += 7.0
    front_vehicle = configcarla.add_one_vehicle(world=world, transform=ego_transform, 
                                                vehicle_type='vehicle.tesla.model3')

    # Add more vehicles
    vehicles = configcarla.add_vehicles_randomly(world=world, number=5)
    vehicles.append(ego_vehicle)
    vehicles.append(front_vehicle)
    tm = configcarla.traffic_manager(client=client, vehicles=vehicles)
    
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
           
            world.tick()
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