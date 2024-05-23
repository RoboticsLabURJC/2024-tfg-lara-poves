import pygame
import carla
import configcarla
from configcarla import SIZE_CAMERA

Z = 0.5

def main(client:carla.Client, screen:pygame.Surface, town:str, transform:carla.Transform):
    world, client = configcarla.setup_carla(name_world=town, port=2000, delta_seconds=0.05, client=client)

    # Add Ego Vehicle
    ego_vehicle = configcarla.add_one_vehicle(world=world, vehicle_type='vehicle.lincoln.mkz_2020',
                                              ego_vehicle=True, transform=transform)

    # Add sensors to Ego Vehicle
    sensors = configcarla.Vehicle_sensors(vehicle=ego_vehicle, world=world, screen=screen)

    driver_transform = carla.Transform(carla.Location(z=2.0, x=1.25), carla.Rotation(roll=90.0, pitch=-2.0))
    camera = sensors.add_camera_rgb(size_rect=(SIZE_CAMERA, SIZE_CAMERA), transform=driver_transform,
                                    seg=True, text='Driver view', init_extra=(SIZE_CAMERA, 0), 
                                    lane=True, canvas_seg=False)
    
    world_transform = carla.Transform(carla.Location(z=2.5, x=-4.75), carla.Rotation(roll=90.0))
    sensors.add_camera_rgb(size_rect=(SIZE_CAMERA, SIZE_CAMERA), init=(0, 0), 
                           transform=world_transform, text='World view')
    
    # Instance PID controller
    pid = configcarla.PID(ego_vehicle)

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
           
            world.tick()
            sensors.update_data()
            
            # Control vehicle
            error_road = camera.get_deviation()
            pid.controll_vehicle(error_road)

    except KeyboardInterrupt:
        return None

    except AssertionError as e:
        print("ERROR:", e)
        return client

    finally:
        sensors.destroy()

if __name__ == "__main__":
    scenes = [
        ('Town05', carla.Transform(carla.Location(x=50.0, y=-145.7, z=Z))),
        ('Town05', carla.Transform(carla.Location(x=151.5, y=7.0, z=Z), carla.Rotation(yaw=90.0))),
        ('Town04', carla.Transform(carla.Location(x=198.5, y=-163, z=0.5), carla.Rotation(yaw=90.0)))
    ]

    client = None
    size = (SIZE_CAMERA * 2, SIZE_CAMERA)
    screen = configcarla.setup_pygame(size=size, name='Follow lane')

    for scene in scenes:
        client = main(screen=screen, client=client, town=scene[0], transform=scene[1])
        if client == None:
            break

        screen.blit(pygame.Surface(size), (0, 0))
        pygame.display.flip()

    pygame.quit()