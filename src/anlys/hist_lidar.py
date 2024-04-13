import pygame
import carla
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, src_path)

import configcarla
from configcarla import FRONT, DIST

# Screen
HEIGHT= 400
WIDTH = 400

def add_config_vehicles(config, world):
    vehicles = []
    for i in range(len(config[0])):
        v = configcarla.add_one_vehicle(transform=config[0][i], world=world, vehicle_type=config[1][i])
        vehicles.append(v)

    return vehicles

def show_hist(dist, bin_edges):
    plt.hist(dist, bins=bin_edges, edgecolor='black')
    plt.ylim(0, 65)
    plt.show()
    
def main():
    # Setup 
    world, client = configcarla.setup_carla(name_world='Town01')
    screen, clock = configcarla.setup_pygame(size=(WIDTH * 3, HEIGHT * 2), name='Histogram Front-Front')

    # Add Ego Vehicle
    ego_transform = carla.Transform(carla.Location(x=140, y=129, z=2.5), carla.Rotation(yaw=180))
    ego_vehicle = configcarla.add_one_vehicle(world=world, ego_vehicle=True, transform=ego_transform,
                                              vehicle_type='vehicle.lincoln.mkz_2020')
    sensors = configcarla.Vehicle_sensors(vehicle=ego_vehicle, world=world, screen=screen)

    # Add sensors to Ego Vehicle
    lidar_transform = carla.Transform(carla.Location(x=-0.5, z=1.8), carla.Rotation(yaw=90.0))
    camera_transform = carla.Transform(carla.Location(z=2.5, x=0.5), 
                                       carla.Rotation(pitch=-10.0, roll=90.0))
    
    sensors.add_camera_rgb(size_rect=(WIDTH, HEIGHT), init=(0, 0), transform=camera_transform)
    camera_transform.location.x = -4.0
    sensors.add_camera_rgb(size_rect=(WIDTH, HEIGHT), init=(0, HEIGHT), transform=camera_transform)
    lidar = sensors.add_lidar(size_rect=(WIDTH * 2, HEIGHT * 2), init=(WIDTH, 0), scale_lidar=35,
                      transform=lidar_transform)
    
    # Possible configurations of vehicles
    x = ego_transform.location.x
    y = ego_transform.location.y
    z = ego_transform.location.z
    front_transform = carla.Transform(carla.Location(x=x-6.0, y=y, z=z), ego_transform.rotation)
    side_transform = carla.Transform(carla.Location(x=x-12.0, y=y+2.5, z=z), ego_transform.rotation)

    config_vehicles = [
        ([], []),
        ([front_transform], ['vehicle.tesla.model3']),
        ([front_transform, side_transform], ['vehicle.tesla.model3', 'vehicle.carlamotors.firetruck']),
        ([front_transform], ['vehicle.yamaha.yzf'])
    ]

    # Add first config
    index = 0
    front_vehicles = add_config_vehicles(config=config_vehicles[index], world=world)
    
    # Histogram
    bin_edges = np.linspace(0, 10, num=11)
    plt.xlabel('Distance (m)')
    plt.ylabel('Frecuency')
    plt.title('Histogram Front-Front')
        
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                    dist = lidar.get_meas_zones()[DIST][FRONT]
                    plt.hist(dist, bins=bin_edges, edgecolor='black')
                    plt.ylim(0, 65)
                    plt.show()

                elif event.type == pygame.KEYDOWN and event.key == pygame.K_x:
                    for v in front_vehicles:
                        v.destroy()

                    index += 1
                    if index >= len(config_vehicles):
                        index = 0

                    print(f"Changing configutation {index}...")
                    front_vehicles = add_config_vehicles(config=config_vehicles[index], world=world)
            
            sensors.update_data()
            clock.tick(120) # Frame rate

    except KeyboardInterrupt:
        return

    finally:
        sensors.destroy()
        pygame.quit()

if __name__ == "__main__":
    main()
