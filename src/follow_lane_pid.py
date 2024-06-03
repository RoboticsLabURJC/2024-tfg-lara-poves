import pygame
import carla
import configcarla
from configcarla import SIZE_CAMERA

def main():
    world, _ = configcarla.setup_carla(name_world='Town05', port=2000, syn=False)
    screen = configcarla.setup_pygame(size=(SIZE_CAMERA * 2, SIZE_CAMERA), name='Follow lane')

    # Add Ego Vehicle
    transform = carla.Transform(carla.Location(x=50.0, y=-145.7, z=0.5))
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
    
    jump = False
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
           
            sensors.update_data()
            t = ego_vehicle.get_transform()

            if not jump and t.location.y > -18:
                t.location.y = 15
                ego_vehicle.set_transform(t)
                jump = True
            elif t.location.x < 43:
                print("Finish route")
                return
            
            # Control vehicle
            error_road = camera.get_deviation()
            pid.controll_vehicle(error_road)

    except KeyboardInterrupt:
        return

    except AssertionError as e:
        print("ERROR:", e)
        return

    finally:
        sensors.destroy()
        pygame.quit()

if __name__ == "__main__":
    main()