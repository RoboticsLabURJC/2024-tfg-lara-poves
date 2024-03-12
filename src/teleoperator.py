import pygame
import carla
from configcarla import setup_carla, setup_pygame, Teleoperator, Vehicle_sensors

# Screen
HEIGHT= 600
WIDTH = 600
    
def main():
    # Setup CARLA and Pygame
    vehicle_transform = carla.Transform(carla.Location(x=100.0, y=-6.0, z=2.5))
    world, ego_vehicle, _ = setup_carla(name_world='Town03', transform=vehicle_transform)
    screen, clock = setup_pygame(size=(WIDTH * 2, HEIGHT), name='Teleoperator')

    # Create teleoperator
    teleop = Teleoperator(ego_vehicle)

    # Add cameras
    camera_transform = carla.Transform(carla.Location(z=2.5, x=0.5), 
                                       carla.Rotation(pitch=-10.0, roll=90.0))
    
    cameras = Vehicle_sensors(vehicle=ego_vehicle, world=world, screen=screen)
    cameras.add_sensor(sensor='sensor.camera.rgb', size=(WIDTH, HEIGHT), 
                       init=(0, 0), transform=camera_transform)
    
    camera_transform.location.x = -4.0
    cameras.add_sensor(sensor='sensor.camera.rgb', size=(WIDTH, HEIGHT), 
                       init=(WIDTH, 0), transform=camera_transform)

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                
                teleop.control()
            
            cameras.update_screen()
            clock.tick(60) # Frame rate

    except KeyboardInterrupt:
        return

    finally:
        cameras.destroy()
        pygame.quit()

if __name__ == "__main__":
    main()
