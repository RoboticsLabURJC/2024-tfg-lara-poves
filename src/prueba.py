import pygame
import carla
import configcarla
from configcarla import SIZE_CAMERA
import argparse

def main(args):
    world, client = configcarla.setup_carla(name_world='Town04', port=args.port, syn=False)
    screen = configcarla.setup_pygame(size=(SIZE_CAMERA * 2, SIZE_CAMERA), name='Follow lane')

    # Add Ego Vehicle
    transform = carla.Transform(carla.Location(x=-8.76, y=60.8, z=0.1), carla.Rotation(yaw=89.7))
    
    ego_vehicle = configcarla.add_one_vehicle(world=world, vehicle_type='vehicle.lincoln.mkz_2020',
                                              ego_vehicle=True, transform=transform)

    # Add sensors to Ego Vehicle
    sensors = configcarla.Vehicle_sensors(vehicle=ego_vehicle, world=world, screen=screen)

    driver_transform = carla.Transform(carla.Location(x=0.5, z=1.7292))
    camera = sensors.add_camera_rgb(size_rect=(SIZE_CAMERA, SIZE_CAMERA), transform=driver_transform,
                                    seg=False, text='Driver view', init_extra=(SIZE_CAMERA, 0), 
                                    lane=True, canvas_seg=False)
    
    world_transform = carla.Transform(carla.Location(z=2.5, x=-4.75))
    sensors.add_camera_rgb(size_rect=(SIZE_CAMERA, SIZE_CAMERA), init=(0, 0), 
                           transform=world_transform, text='World view')
    



    tm = configcarla.traffic_manager(client=client, vehicles=[ego_vehicle], speed=80)
    tm.ignore_lights_percentage(ego_vehicle, 100)
    

    finish_ep = False
    percentage = 80

    vel_front = 0
    counter = 0
    mapa = world.get_map()

    location = ego_vehicle.get_location()

    # Obtener el waypoint más cercano a la posición del vehículo
    waypoint = mapa.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
    
    # Obtener el road_id del waypoint
    road_id = waypoint.road_id
    change = False

    try:
        while not finish_ep:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                
            counter += 1
            try:
                sensors.update_data()
                t = ego_vehicle.get_transform()
                vel_front += carla.Vector3D(ego_vehicle.get_velocity()).length()
                target_vel = 1

                if counter % 5 == 0:
                    vel_front /= 5
                    if vel_front > target_vel and abs(vel_front - target_vel) > 1:
                        percentage += 1
                        tm.global_percentage_speed_difference(percentage)
                        print("reduzco vel, vel front:", vel_front, "target vel:", target_vel, "%:", percentage)
                    elif target_vel > vel_front and abs(target_vel - vel_front) > 1: 
                        percentage -= 1
                        tm.global_percentage_speed_difference(percentage)
                        print("aumento vel, vel front:", vel_front, "target vel:", target_vel, "%:", percentage)
                    vel_front = 0

                finish_ep =  abs(t.location.x + 442) <= 3 and abs(t.location.y - 30) <= 3
 
            except AssertionError:
                print("exp")
                pass

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
