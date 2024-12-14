import pygame
import carla
import configcarla
from configcarla import SIZE_CAMERA
import argparse

def main(args):
    world, _ = configcarla.setup_carla(name_world='Town04', port=args.port, syn=False)
    screen = configcarla.setup_pygame(size=(SIZE_CAMERA * 2, SIZE_CAMERA), name='Follow lane')

    # Add Ego Vehicle
    transform = carla.Transform(carla.Location(x=-25.0, y=-252, z=0.1), carla.Rotation(yaw=125.0))
    ego_vehicle = configcarla.add_one_vehicle(world=world, vehicle_type='vehicle.lincoln.mkz_2020',
                                              ego_vehicle=True, transform=transform)

    # Add sensors to Ego Vehicle
    sensors = configcarla.Vehicle_sensors(vehicle=ego_vehicle, world=world, screen=screen)

    driver_transform = carla.Transform(carla.Location(x=0.5, z=1.7292))
    camera = sensors.add_camera_rgb(size_rect=(SIZE_CAMERA, SIZE_CAMERA), transform=driver_transform,
                                    seg=True, text='Driver view', init_extra=(SIZE_CAMERA, 0), 
                                    lane=True, canvas_seg=False, check_area_lane=True)
    
    world_transform = carla.Transform(carla.Location(z=2.5, x=-4.75))
    sensors.add_camera_rgb(size_rect=(SIZE_CAMERA, SIZE_CAMERA), init=(0, 0), 
                           transform=world_transform, text='World view')
    
    # Instance PID controller
    pid = configcarla.PID(ego_vehicle)
    
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
           
            sensors.update_data()
            t = ego_vehicle.get_transform()

            if t.location.y > -24.5:
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
    parser = argparse.ArgumentParser(
        description="Execute an inference trial on a specified Gym environment",
        usage="python3 %(prog)s --port <port_number>"
    )
    parser.add_argument(
        '--port', 
        type=int, 
        required=False, 
        default=2000,
        help='Port for Carla'
    )

    main(parser.parse_args())
