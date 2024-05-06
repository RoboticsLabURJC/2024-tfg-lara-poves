import pygame
import carla
import configcarla
from configcarla import SIZE_CAMERA

def main():
    # Scenes
    town_s1 = 'Town05'
    ego_transform_s1 = carla.Transform(carla.Location(x=151.5, y=5.0, z=1.0), carla.Rotation(yaw=90.0))

    town_s2 = 'Town05'
    ego_transform_s2 = carla.Transform(carla.Location(x=47.0, y=-146.0, z=1.0), carla.Rotation(yaw=0.0))

    world, _ = configcarla.setup_carla(name_world=town_s1, port=2000, delta_seconds=0.05)
    screen = configcarla.setup_pygame(size=(SIZE_CAMERA * 2, SIZE_CAMERA), name='Follow lane - PID')

    # Add Ego Vehicle
    ego_vehicle = configcarla.add_one_vehicle(world=world, vehicle_type='vehicle.lincoln.mkz_2020',
                                              ego_vehicle=True, transform=ego_transform_s1)

    # Add sensors to Ego Vehicle
    sensors = configcarla.Vehicle_sensors(vehicle=ego_vehicle, world=world, screen=screen)

    driver_transform = carla.Transform(carla.Location(z=2.0, x=1.25), carla.Rotation(roll=90.0, pitch=-2.0))
    camera = sensors.add_camera_rgb(size_rect=(SIZE_CAMERA, SIZE_CAMERA), transform=driver_transform,
                                    seg=True, text='Driver view', init_extra=(SIZE_CAMERA, 0), lane=True)
    camera.set_threshold_lane(0.2)
    
    world_transform = carla.Transform(carla.Location(z=2.5, x=-4.75), carla.Rotation(roll=90.0))
    sensors.add_camera_rgb(size_rect=(SIZE_CAMERA, SIZE_CAMERA), init=(0, 0), transform=world_transform, 
                           text='World view')
    
    # Instance PID controller
    pid = configcarla.PID(ego_vehicle)

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
           
            sensors.update_data()
            error_road = camera.get_deviation()
            
            # Control vehicle
            pid.controll_vehicle(error_road)
          
            world.tick()

    except KeyboardInterrupt:
        return

    finally:
        sensors.destroy()
        pygame.quit()

if __name__ == "__main__":
    main()