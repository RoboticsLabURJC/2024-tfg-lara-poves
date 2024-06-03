import pygame
import carla
import configcarla
    
def main():
    # Setup CARLA and Pygame
    world, _ = configcarla.setup_carla(name_world='Town03', syn=False)
    screen = configcarla.setup_pygame(size=(configcarla.SIZE_CAMERA * 2, configcarla.SIZE_CAMERA), 
                                      name='Teleoperator')

    # Add Ego Vehicle
    vehicle_transform = carla.Transform(carla.Location(x=80.0, y=10.0, z=2.5))
    ego_vehicle = configcarla.add_one_vehicle(world=world, vehicle_type='vehicle.lincoln.mkz_2020',
                                              transform=vehicle_transform, ego_vehicle=True)

    # Create teleoperator
    teleop = configcarla.Teleoperator(ego_vehicle, throttle=0.6)

    # Add cameras
    cameras = configcarla.Vehicle_sensors(vehicle=ego_vehicle, world=world, screen=screen)

    camera_transform = carla.Transform(carla.Location(z=2.5, x=0.5), carla.Rotation(pitch=-10.0, roll=90.0))
    cameras.add_camera_rgb(size_rect=(configcarla.SIZE_CAMERA, configcarla.SIZE_CAMERA), init=(0, 0), 
                           text='Driver view', transform=camera_transform)
    
    camera_transform.location.x = -4.0
    cameras.add_camera_rgb(size_rect=(configcarla.SIZE_CAMERA, configcarla.SIZE_CAMERA),
                           init=(configcarla.SIZE_CAMERA, 0),
                           text='World View', transform=camera_transform)

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            
            cameras.update_data()

            v = ego_vehicle.get_velocity()
            v = carla.Vector3D(v).length()
            print("v = ", v)

            diff = 10 - v
            print("diff = ", diff)

            control = carla.VehicleControl()
            if diff > 0:
                control.throttle = min(diff, 1)
            else:
                control.brake = min(-diff, 1.0)
            
            ego_vehicle.apply_control(control)

    except KeyboardInterrupt:
        return

    finally:
        cameras.destroy()
        pygame.quit()

if __name__ == "__main__":
    main()
