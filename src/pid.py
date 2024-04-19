import pygame
import carla
import configcarla
from PIL import Image

HEIGHT= 490
WIDTH = 500

def main():
    world, client = configcarla.setup_carla(name_world='Town05', port=2000)
    screen = configcarla.setup_pygame(size=(WIDTH * 2, HEIGHT * 2), name='PID')

    # Load layered map for Town 01 with minimum layout plus buildings and parked vehicles
    world = client.load_world('Town05_Opt', carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)

    # Toggle all buildings off
    world.unload_map_layer(carla.MapLayer.Buildings)

    # Surface to show segmentation mask road
    sub_screen_mask = pygame.Surface((WIDTH, HEIGHT))
    rect_mask = sub_screen_mask.get_rect(topleft=(WIDTH, HEIGHT))

    # Add Ego Vehicle
    vehicle_transform = carla.Transform(carla.Location(x=180.0, y=-6.0, z=2.5))
    ego_vehicle = configcarla.add_one_vehicle(world=world, vehicle_type='vehicle.lincoln.mkz_2020', ego_vehicle=True)
    sensors = configcarla.Vehicle_sensors(vehicle=ego_vehicle, world=world, screen=screen)

    # Add sensors to Ego Vehicle
    driver_transform = carla.Transform(carla.Location(z=2.5, x=0.75), carla.Rotation(pitch=-0.0, roll=90.0))
    camera = sensors.add_camera_rgb(size_rect=(WIDTH, HEIGHT), init=(0, 0), transform=driver_transform,
                                    seg=True, init_seg=(WIDTH, 0), text='Driver view')
    
    world_transform = carla.Transform(carla.Location(z=2.5, x=-4.0), carla.Rotation(pitch=-10.0, roll=90.0))
    sensors.add_camera_rgb(size_rect=(WIDTH, HEIGHT), init=(0, HEIGHT), transform=world_transform, 
                           text='World view')
    
    # Instance PID controller
    pid = configcarla.PID(ego_vehicle)

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
           
            # Upadate data of sensors
            sensors.update_data(flip=False)
            error_road = camera.get_deviation_road(rect_mask=rect_mask)

            # Control vehicle
            pid.controll_vehicle(error_road)

            # Update pygame and carla
            pygame.display.flip()
            world.tick()

    except KeyboardInterrupt:
        return

    finally:
        sensors.destroy()
        pygame.quit()

if __name__ == "__main__":
    main()