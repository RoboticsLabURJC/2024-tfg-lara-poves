import pygame
import carla
import sys
import os

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, src_path)

import configcarla
from configcarla import SIZE_CAMERA

Z = 0.5

def main():
    screen = configcarla.setup_pygame(size=(SIZE_CAMERA * 2, SIZE_CAMERA), name='DQN')
    world, _ = configcarla.setup_carla(name_world='Town05', port=2000, syn=False)

    # Add Ego Vehicle
    transform = carla.Transform(carla.Location(x=50.0, y=-145.7, z=Z))
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
           
            sensors.update_data(flip=False)
            cm = camera.get_lane_cm()
            area = camera.get_lane_area()
            points_left, points_right = camera.get_lane_points(show=True)

            print("cm:", cm, type(cm), cm.shape)
            print("area:", area, type(area))
            print("points left:", points_left, type(points_left), points_left.shape)
            print("points right:", points_right, type(points_right), points_right.shape)
            pygame.display.flip()
            
            # Control vehicle
            error_road = camera.get_deviation()
            pid.controll_vehicle(error_road)

    except KeyboardInterrupt:
        return None

    finally:
        sensors.destroy()
        pygame.quit()

if __name__ == "__main__":
    main()