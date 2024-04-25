import pygame
import carla
import os
import sys
import csv
import argparse

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, src_path)

import configcarla
from configcarla import FRONT, DIST

# Screen
HEIGHT= 400
WIDTH = 450

TITLE = 0
TRANSFORM = 1
VEHICLE = 2

def add_config_vehicles(config, world):
    vehicles = []
    for i in range(len(config[TRANSFORM])):
        v = configcarla.add_one_vehicle(transform=config[TRANSFORM][i], world=world, 
                                        vehicle_type=config[VEHICLE][i])
        vehicles.append(v)

    return vehicles
    
def main(mode):
    # Setup 
    world, _ = configcarla.setup_carla(name_world='Town01')
    screen = configcarla.setup_pygame(size=(WIDTH * 3, HEIGHT * 2), name='Histogram Front-Front')

    # Add Ego Vehicle
    ego_transform = carla.Transform(carla.Location(x=140, y=129, z=2.5), carla.Rotation(yaw=180))
    ego_vehicle = configcarla.add_one_vehicle(world=world, ego_vehicle=True, transform=ego_transform,
                                              vehicle_type='vehicle.lincoln.mkz_2020')

    # Add sensors to Ego Vehicle
    sensors = configcarla.Vehicle_sensors(vehicle=ego_vehicle, world=world, screen=screen)

    camera_transform = carla.Transform(carla.Location(z=2.5, x=0.5), carla.Rotation(pitch=-10.0, roll=90.0))
    sensors.add_camera_rgb(size_rect=(WIDTH, HEIGHT), init=(0, 0), text='Driver view',
                           transform=camera_transform)
    
    camera_transform.location.x = -4.0
    sensors.add_camera_rgb(size_rect=(WIDTH, HEIGHT), init=(0, HEIGHT), text='World View',
                           transform=camera_transform)

    lidar_transform = carla.Transform(carla.Location(x=-0.5, z=1.8), carla.Rotation(yaw=90.0))
    lidar = sensors.add_lidar(size_rect=(WIDTH * 2, HEIGHT * 2), init=(WIDTH, 0), scale_lidar=35,
                              transform=lidar_transform)
    
    # Possible configurations of vehicles
    x = ego_transform.location.x
    y = ego_transform.location.y
    z = ego_transform.location.z
    front_transform = carla.Transform(carla.Location(x=x-6.0, y=y, z=z), ego_transform.rotation)
    side_transform = carla.Transform(carla.Location(x=x-12.0, y=y+2.5, z=z), ego_transform.rotation)

    configuration = [
        ('Empty', [], []),
        ('Car', [front_transform], ['vehicle.tesla.model3']),
        ('Motorbike', [front_transform], ['vehicle.yamaha.yzf']),
        ('Car + Truck', [front_transform, side_transform], 
         ['vehicle.tesla.model3', 'vehicle.carlamotors.firetruck'])
    ]

    # Add first config
    index = 0
    front_vehicles = add_config_vehicles(config=configuration[index], world=world)

    # Open csv file
    if mode != 'n':
        csv_file= open('/home/alumnos/lara/2024-tfg-lara-poves/src/histograms/hist_data.csv', mode, newline='') 
        csv_desktop = csv.writer(csv_file)
    
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                    dist = lidar.get_meas_zones()[DIST][FRONT]
                    dist.insert(0, configuration[index][TITLE])
                    print("Save data:", configuration[index][TITLE])

                    if mode != 'n':
                        csv_desktop.writerow(dist)

                elif event.type == pygame.KEYDOWN and event.key == pygame.K_x:
                    for v in front_vehicles:
                        v.destroy()

                    index += 1
                    if index >= len(configuration):
                        index = 0

                    print("Set configuration: ", configuration[index][TITLE])
                    front_vehicles = add_config_vehicles(config=configuration[index], world=world)
            
            sensors.update_data()
            world.tick() # Frame rate

    except KeyboardInterrupt:
        return

    finally:
        sensors.destroy()
        for v in front_vehicles:
            v.destroy()

        if mode != 'n':
            csv_file.close()

        pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specified opening mode for csv file")
    current_path = os.path.abspath(__file__)
    parser.add_argument("mode", choices=["a", "w", "n"])
    args = parser.parse_args()    
    main(args.mode)
