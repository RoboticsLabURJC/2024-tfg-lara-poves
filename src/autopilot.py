import pygame
import carla
import configcarla
from configcarla import Camera_stream as CS

# Screen
HEIGHT= 600
WIDTH = 600
ELEVATION = 2.5
    
def main():
    # Setup CARLA and pygame
    world, ego_vehicle, client = configcarla.setup_carla(name_world='Town03')
    screen, clock = configcarla.setup_pygame(width=WIDTH * 2, height=HEIGHT, 
                                             name='Teleoperator')

    # Create cameras' screens
    sub_screen = pygame.Surface((WIDTH, HEIGHT))
    camera_transform = carla.Transform(carla.Location(z=ELEVATION, x=0.5), 
                                       carla.Rotation(pitch=-10.0, roll=90.0))
    
    driver = CS(vehicle=ego_vehicle, rect=sub_screen.get_rect(topleft=(0, 0)), 
                world=world, transform=camera_transform)

    camera_transform.location.x = -4.0
    spectator = CS(vehicle=ego_vehicle, transform=camera_transform, world=world, 
                   rect=sub_screen.get_rect(topleft=(WIDTH, 0)))
    
    # Add vehicles
    vehicles = configcarla.add_vehicles(world=world, number=10)
    vehicles.append(ego_vehicle)
    tm = configcarla.traffic_manager(client=client, vehicles=vehicles)

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            driver.show_camera(screen)
            spectator.show_camera(screen)
            pygame.display.flip()
            clock.tick(60) # Frame rate

    except KeyboardInterrupt:
        pygame.quit()

if __name__ == "__main__":
    main()

# lidar - 3 -> añadir lidar y visualizarlo
'''
autopilot -> al dar las curvas a veces pierde al carril y se choca contra las paredes
'''