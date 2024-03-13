import pygame
import carla
import configcarla

# Screen
HEIGHT= 600
WIDTH = 600
    
def main():
    # Setup 
    world, client = configcarla.setup_carla(name_world='Town01')
    screen, clock = configcarla.setup_pygame(size=(WIDTH * 3, HEIGHT), name='Autopilot')

    # Add Ego Vehicle
    ego_transform = carla.Transform(carla.Location(x=140, y=129, z=2.5), carla.Rotation(yaw=180))
    ego_vehicle = configcarla.add_one_vehicle(world=world, ego_vehicle=True, 
                                              transform=ego_transform,
                                              vehicle_type='vehicle.lincoln.mkz_2020')
    sensors = configcarla.Vehicle_sensors(vehicle=ego_vehicle, world=world, screen=screen)

    # Add sensors to Ego Vehicle
    camera_transform = carla.Transform(carla.Location(z=2.5, x=0.5), 
                                       carla.Rotation(pitch=-10.0, roll=90.0))
    lidar_transform = carla.Transform(carla.Location(x=-0.5, z=1.8), carla.Rotation(yaw=90.0))
    
    sensors.add_sensor(sensor_type='sensor.camera.rgb', size_rect=(WIDTH, HEIGHT), 
                       init=(0, 0), transform=camera_transform)
    
    camera_transform.location.x = -4.0
    sensors.add_sensor(sensor_type='sensor.camera.rgb', size_rect=(WIDTH, HEIGHT),
                       init=(WIDTH, 0), transform=camera_transform)
    
    sensors.add_sensor(sensor_type='sensor.lidar.ray_cast', size_rect=(WIDTH, HEIGHT), 
                       init=(WIDTH * 2, 0), transform=lidar_transform, scale_lidar=25)
    
    # Add a car in front of Ego Vehicle
    ego_transform.location.x = 134.0

    near_vehicle = configcarla.add_one_vehicle(world=world, ego_vehicle=False, tag='*tesla*',
                                               transform=ego_transform)
    
    '''
    ego_transform.location.y -= 3.0
    ego_transform.location.x = 140
    a = configcarla.add_one_vehicle(world=world, ego_vehicle=False, tag='*tesla*',transform=ego_transform)
    '''

    # Add more vehicles
    vehicles = configcarla.add_vehicles_randomly(world=world, number=10)
    vehicles.append(ego_vehicle)
    vehicles.append(near_vehicle)
    #tm = configcarla.traffic_manager(client=client, vehicles=vehicles)
    
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            
            sensors.update_screen()
            clock.tick(60) # Frame rate

    except KeyboardInterrupt:
        return

    finally:
        sensors.destroy()
        pygame.quit()

if __name__ == "__main__":
    main()