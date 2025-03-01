import pygame
import carla
import argparse
import time
import csv
import os
import sys

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, src_path)

from configcarla import SIZE_CAMERA, PATH
import configcarla

def main(args):
    if args.new == 1:
        transform = carla.Transform(carla.Location(x=-10, y=300, z=0.1), carla.Rotation(yaw=292))
        town = 'Town04'
    elif args.new == 2:
        transform = carla.Transform(carla.Location(x=114, y=207.3, z=1.7))
        town = 'Town03'
    else:
        transform = carla.Transform(carla.Location(x=-25.0, y=-252, z=0.1), carla.Rotation(yaw=125.0))
        town = 'Town04'

    world, _ = configcarla.setup_carla(name_world=town, port=args.port, syn=False)
    screen = configcarla.setup_pygame(size=(SIZE_CAMERA * 2, SIZE_CAMERA), name='Follow lane')

    # Add Ego Vehicle
    ego_vehicle = configcarla.add_one_vehicle(world=world, vehicle_type='vehicle.lincoln.mkz_2020',
                                              ego_vehicle=True, transform=transform)

    # Add sensors to Ego Vehicle
    sensors = configcarla.Vehicle_sensors(vehicle=ego_vehicle, world=world, screen=screen)

    driver_transform = carla.Transform(carla.Location(x=0.5, z=1.7292))
    camera = sensors.add_camera_rgb(size_rect=(SIZE_CAMERA, SIZE_CAMERA), transform=driver_transform,
                                    seg=False, text='Driver view', init_extra=(SIZE_CAMERA, 0), 
                                    lane=True, canvas_seg=False, check_area_lane=True)
    
    world_transform = carla.Transform(carla.Location(z=2.5, x=-4.75))
    sensors.add_camera_rgb(size_rect=(SIZE_CAMERA, SIZE_CAMERA), init=(0, 0), 
                           transform=world_transform, text='World view')
    
    # Instance PID controller
    pid = configcarla.PID(ego_vehicle)

    if args.record:
        # Episode file
        dir_csv = PATH + '2024-tfg-lara-poves/src/pid/csv/' 
        if not os.path.exists(dir_csv):
            os.makedirs(dir_csv)
        files = os.listdir(dir_csv)
        num_files = len(files) + 1
        csv_file = open(dir_csv + 'data_' + str(num_files) + '.csv', mode='w', newline='')
        writer_csv = csv.writer(csv_file)
        writer_csv.writerow(['velocity', 'deviation', 'throttle', 'steer'])

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
           
            sensors.update_data()
            t = ego_vehicle.get_transform()

            if args.new == 1:
                finish = abs(t.location.x - 96) <= 3 and abs(t.location.y + 362) <= 3
            elif args.new == 2:
                finish = abs(t.location.x - 165) <= 3 and abs(t.location.y + 208) <= 3
            else:
                finish = t.location.y > -24.5

            if finish:
                print("Finish route")
                return
            
            # Control vehicle
            error_road = camera.get_deviation()
            #t = time.time_ns()
            control = pid.controll_vehicle(error_road)
            #print("PID:", time.time_ns() - t)

            if args.record:
                vel = ego_vehicle.get_velocity()
                writer_csv.writerow([carla.Vector3D(vel).length(), error_road, control.throttle, control.steer])

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
        usage="python3 %(prog)s --port <port_number> --record <record> --new <new>"
    )
    parser.add_argument(
        '--port', 
        type=int, 
        required=False, 
        default=2000,
        help='Port for Carla'
    )
    parser.add_argument(
        '--record', 
        type=int, 
        required=False, 
        default=0,
        help='Save the velocity and the deviation in a csv.'
    )
    parser.add_argument(
        '--new', 
        type=int, 
        required=False, 
        default=0,
        help='Test it in a new scenario.'
    )

    main(parser.parse_args())
