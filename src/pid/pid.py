import pygame
import carla
import os
import sys
import csv
import shutil

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, src_path)

import configcarla
from configcarla import SIZE_CAMERA

def main(save_data):
    world, _ = configcarla.setup_carla(name_world='Town05', port=2000, delta_seconds=0.05)
    screen = configcarla.setup_pygame(size=(SIZE_CAMERA * 2, SIZE_CAMERA), name='PID')

    # Add Ego Vehicle
    ego_transform = carla.Transform(carla.Location(x=151.5, y=-65.0, z=2.5), carla.Rotation(yaw=90.0))
    ego_vehicle = configcarla.add_one_vehicle(world=world, vehicle_type='vehicle.lincoln.mkz_2020',
                                              ego_vehicle=True, transform=ego_transform)

    # Add sensors to Ego Vehicle
    sensors = configcarla.Vehicle_sensors(vehicle=ego_vehicle, world=world, screen=screen)

    driver_transform = carla.Transform(carla.Location(z=2.0, x=1.25), carla.Rotation(roll=90.0, pitch=-2.0))
    camera = sensors.add_camera_rgb(size_rect=(SIZE_CAMERA, SIZE_CAMERA), transform=driver_transform,
                                    seg=True, text='Driver view', init_extra=(SIZE_CAMERA, 0), lane=True)
    
    world_transform = carla.Transform(carla.Location(z=2.5, x=-4.75), carla.Rotation(roll=90.0))
    sensors.add_camera_rgb(size_rect=(SIZE_CAMERA, SIZE_CAMERA), init=(0, 0), transform=world_transform, 
                           text='World view')
    
    # Instance PID controller
    pid = configcarla.PID(ego_vehicle)

    # Open csv file
    if save_data:
        path_csv = '/home/alumnos/lara/2024-tfg-lara-poves/src/pid/pid_data.csv'

        # To be able to compare the new PID values with the old ones
        shutil.copy(path_csv, '/home/alumnos/lara/2024-tfg-lara-poves/src/pid/pid_data_prev.csv')

        csv_file = open(path_csv, 'w')
        csv_desktop = csv.writer(csv_file)

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
           
            sensors.update_data()
            error_road = camera.get_deviation()
            
            # Save control error in a csv
            if save_data:
                csv_desktop.writerow([error_road])

            # Control vehicle
            pid.controll_vehicle(error_road)
          
            world.tick()

    except KeyboardInterrupt:
        return

    finally:
        sensors.destroy()
        pygame.quit()

if __name__ == "__main__":
    save_data = len(sys.argv) > 1 and sys.argv[1] == "w"
    main(save_data)