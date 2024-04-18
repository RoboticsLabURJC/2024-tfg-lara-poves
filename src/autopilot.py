import pygame
import carla
import configcarla

# Screen
HEIGHT= 460
WIDTH = 460
DECREASE = 50
    
def main():
    # Setup 
    world, client = configcarla.setup_carla(name_world='Town01', port=2000)
    screen = configcarla.setup_pygame(size=(WIDTH * 4 - DECREASE * 2, HEIGHT * 2), name='Autopilot')

    # Add Ego Vehicle
    ego_transform = carla.Transform(carla.Location(x=140, y=129, z=2.5), carla.Rotation(yaw=180))
    ego_vehicle = configcarla.add_one_vehicle(world=world, ego_vehicle=True, transform=ego_transform,
                                              vehicle_type='vehicle.lincoln.mkz_2020')
    sensors = configcarla.Vehicle_sensors(vehicle=ego_vehicle, world=world, screen=screen)

    # Add sensors to Ego Vehicle
    camera_transform = carla.Transform(carla.Location(z=2.5, x=0.5), carla.Rotation(pitch=-10.0, roll=90.0))
    sensors.add_camera_rgb(size_rect=(WIDTH, HEIGHT), init=(0, 0), transform=camera_transform,
                           seg=True, init_seg=(0, HEIGHT), text='Driver view')

    camera_transform.location.x = -4.0
    sensors.add_camera_rgb(size_rect=(WIDTH, HEIGHT), init=(WIDTH * 3 - DECREASE * 2, 0),
                           transform=camera_transform, text='World view')

    camera_back_transform = carla.Transform(carla.Location(z=2.5, x=-0.5), 
                                            carla.Rotation(roll=90.0, yaw=180.0))
    sensors.add_camera_rgb(size_rect=(WIDTH, HEIGHT), init=(WIDTH * 3 - DECREASE * 2, HEIGHT),
                           transform=camera_back_transform, text='Back view')

    lidar_transform = carla.Transform(carla.Location(x=-0.5, z=1.8), carla.Rotation(yaw=90.0))
    sensors.add_lidar(size_rect=((WIDTH - DECREASE) * 2, HEIGHT * 2), init=(WIDTH, 0), scale_lidar=40,
                      transform=lidar_transform)
    
    # Add a car in front of Ego Vehicle
    ego_transform.location.x -= 6.0
    front_vehicle = configcarla.add_one_vehicle(world=world, transform=ego_transform,
                                                vehicle_type='vehicle.tesla.model3')

    # Add more vehicles
    vehicles = configcarla.add_vehicles_randomly(world=world, number=15)
    vehicles.append(ego_vehicle)
    vehicles.append(front_vehicle)
    tm = configcarla.traffic_manager(client=client, vehicles=vehicles)
    
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
           
            sensors.update_data()
            world.tick()

    except KeyboardInterrupt:
        return

    finally:
        sensors.destroy()
        pygame.quit()

if __name__ == "__main__":
    main()