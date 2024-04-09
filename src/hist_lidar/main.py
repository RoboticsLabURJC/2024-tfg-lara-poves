import pygame
import carla
import os
import sys

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_path)

import configcarla

# Screen
HEIGHT= 450
WIDTH = 450
    
def main():
    # Setup 
    world, client = configcarla.setup_carla(name_world='Town01')
    screen, clock = configcarla.setup_pygame(size=(WIDTH * 3, HEIGHT * 2), name='Autopilot')

    # Add Ego Vehicle
    ego_transform = carla.Transform(carla.Location(x=140, y=129, z=2.5), carla.Rotation(yaw=180))
    ego_vehicle = configcarla.add_one_vehicle(world=world, ego_vehicle=True, transform=ego_transform,
                                              vehicle_type='vehicle.lincoln.mkz_2020')
    sensors = configcarla.Vehicle_sensors(vehicle=ego_vehicle, world=world, screen=screen)

    # Add sensors to Ego Vehicle
    lidar_transform = carla.Transform(carla.Location(x=-0.5, z=1.8), carla.Rotation(yaw=90.0))
    camera_transform = carla.Transform(carla.Location(z=2.5, x=0.5), 
                                       carla.Rotation(pitch=-10.0, roll=90.0))
    
    sensors.add_camera_rgb(size_rect=(WIDTH, HEIGHT), init=(0, 0), transform=camera_transform)
    camera_transform.location.x = -4.0
    sensors.add_camera_rgb(size_rect=(WIDTH, HEIGHT), init=(0, HEIGHT), transform=camera_transform)
    sensors.add_lidar(size_rect=(WIDTH * 2, HEIGHT * 2), init=(WIDTH, 0), scale_lidar=40,
                      transform=lidar_transform)
    
    # Add a car in front of Ego Vehicle
    ego_transform.location.x -= 6.0
    front_vehicle = configcarla.add_one_vehicle(world=world, transform=ego_transform,
                                                vehicle_type='vehicle.tesla.model3')
    
    counter = 0
    try:
        while counter < 1:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                    pygame.image.save(screen, "img/hist_lidar0.png")
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_x:
                    front_vehicle.destroy()
                    counter += 1
            
            sensors.update_data()
            clock.tick(120) # Frame rate

    except KeyboardInterrupt:
        return

    finally:
        sensors.destroy()
        pygame.quit()

if __name__ == "__main__":
    main()