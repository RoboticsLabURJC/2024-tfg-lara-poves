import pygame
import carla
from configcarla import setup_carla, setup_pygame, Camera_stream, teleoperator

# Screen
HEIGHT= 600
WIDTH = 600
ELEVATION = 2.5
    
def main():
    # Setup CARLA and pygame
    vehicle_transform = carla.Transform(carla.Location(x=100.0, y=-6.0, z=ELEVATION))
    world, ego_vehicle, _ = setup_carla(name_world='Town03', transform=vehicle_transform)
    screen, clock = setup_pygame(width=WIDTH * 2, height=HEIGHT, name='Teleoperator')

    # Create cameras' screens
    sub_screen = pygame.Surface((WIDTH, HEIGHT))
    camera_transform = carla.Transform(carla.Location(z=ELEVATION, x=0.5), 
                                       carla.Rotation(pitch=-10.0, roll=90.0))
    
    driver = Camera_stream(vehicle=ego_vehicle, rect=sub_screen.get_rect(topleft=(0, 0)), 
                           world=world, transform=camera_transform)

    camera_transform.location.x = -4.0
    spectator = Camera_stream(vehicle=ego_vehicle, transform=camera_transform,
                              world=world, rect=sub_screen.get_rect(topleft=(WIDTH, 0)))

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    teleoperator(ego_vehicle)

            driver.show_camera(screen)
            spectator.show_camera(screen)
            pygame.display.flip()
            clock.tick(60) # Frame rate

    except KeyboardInterrupt:
        pygame.quit()

if __name__ == "__main__":
    main()
