import pygame
import carla
from configcarla import setup_carla, add_one_vehicle, setup_pygame, Teleoperator, Vehicle_sensors

# Screen
HEIGHT= 600
WIDTH = 600
    
def main():
    # Setup CARLA and Pygame
    world, _ = setup_carla(name_world='Town03')
    screen, clock = setup_pygame(size=(WIDTH * 2, HEIGHT), name='Teleoperator')

    # Add Ego Vehicle
    vehicle_transform = carla.Transform(carla.Location(x=100.0, y=-6.0, z=2.5))
    ego_vehicle = add_one_vehicle(world=world, vehicle_type='vehicle.lincoln.mkz_2020',
                                  transform=vehicle_transform, ego_vehicle=True)

    # Create teleoperator
    teleop = Teleoperator(ego_vehicle)

    # Add cameras
    cameras = Vehicle_sensors(vehicle=ego_vehicle, world=world, screen=screen)
    camera_transform = carla.Transform(carla.Location(z=2.5, x=0.5), 
                                       carla.Rotation(pitch=-10.0, roll=90.0))
    
    cameras.add_camera_rgb(size_rect=(WIDTH, HEIGHT), init=(0, 0), transform=camera_transform)
    camera_transform.location.x = -4.0
    cameras.add_camera_rgb(size_rect=(WIDTH, HEIGHT), init=(WIDTH, 0), transform=camera_transform)

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                
                teleop.control()
            
            cameras.update_data()
            clock.tick(60) # Frame rate

    except KeyboardInterrupt:
        return

    finally:
        cameras.destroy()
        pygame.quit()

if __name__ == "__main__":
    main()
