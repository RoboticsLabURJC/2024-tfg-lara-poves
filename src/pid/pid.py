import pygame
import carla
import os
import sys
import csv
import shutil

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, src_path)

import configcarla

HEIGHT= 512
WIDTH = 512

def main(save_data):
    world, _ = configcarla.setup_carla(name_world='Town05', port=2000)
    screen = configcarla.setup_pygame(size=(WIDTH * 3, HEIGHT), name='PID')

    # Surface to show segmentation mask road
    sub_screen_mask = pygame.Surface((WIDTH, HEIGHT))
    rect_mask = sub_screen_mask.get_rect(topleft=(WIDTH * 2, 0))

    # Add Ego Vehicle
    ego_transform = carla.Transform(carla.Location(x=153.0, y=-58.0, z=2.5), carla.Rotation(yaw=90.0))
    ego_vehicle = configcarla.add_one_vehicle(world=world, vehicle_type='vehicle.lincoln.mkz_2020',
                                              ego_vehicle=True, transform=ego_transform)
    sensors = configcarla.Vehicle_sensors(vehicle=ego_vehicle, world=world, screen=screen)

    # Add sensors to Ego Vehicle
    driver_transform = carla.Transform(carla.Location(z=2.0, x=1.25), carla.Rotation(roll=90.0, pitch=-2.0))
    camera = sensors.add_camera_rgb(size_rect=(WIDTH, HEIGHT), transform=driver_transform,
                                    seg=True, text='Driver view', init_seg=(WIDTH, 0))
    
    world_transform = carla.Transform(carla.Location(z=2.5, x=-4.75), carla.Rotation(roll=90.0))
    sensors.add_camera_rgb(size_rect=(WIDTH, HEIGHT), init=(0, 0), transform=world_transform, 
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
           
            # Upadate data of sensors
            sensors.update_data(flip=False)
            error_road = camera.get_deviation_road(rect_mask=rect_mask)

            # Save control error in a csv
            if save_data:
                csv_desktop.writerow([error_road])

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
    save_data = len(sys.argv) > 1 and sys.argv[1] == "w"
    main(save_data)