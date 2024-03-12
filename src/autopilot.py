import pygame
import carla
import configcarla

# Screen
HEIGHT= 600
WIDTH = 600
    
def main():
    # Setup 
    world, ego_vehicle, client = configcarla.setup_carla(name_world='Town03')
    screen, clock = configcarla.setup_pygame(size=(WIDTH * 2, HEIGHT), name='Teleoperator')
    sensors = configcarla.Vehicle_sensors(vehicle=ego_vehicle, world=world, screen=screen)

    # Add cameras
    camera_transform = carla.Transform(carla.Location(z=2.5, x=0.5), 
                                       carla.Rotation(pitch=-10.0, roll=90.0))
    
    sensors.add_sensor(sensor='sensor.camera.rgb', size=(WIDTH, HEIGHT), 
                       init=(0, 0), transform=camera_transform)
    
    camera_transform.location.x = -4.0
    sensors.add_sensor(sensor='sensor.camera.rgb', size=(WIDTH, HEIGHT), 
                       init=(WIDTH, 0), transform=camera_transform)
    
    # Add LIDAR
    
    # Add vehicles
    vehicles = configcarla.add_vehicles(world=world, number=10)
    vehicles.append(ego_vehicle)
    tm = configcarla.traffic_manager(client=client, vehicles=vehicles)

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

# lidar - 3 -> añadir lidar y visualizarlo
'''
autopilot -> al dar las curvas a veces pierde al carril y se choca contra las paredes
'''