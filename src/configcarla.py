import carla
import pygame
import numpy as np
import math
import random
from typing import Tuple, List

class Sensor:
    def __init__(self, sensor:carla.Sensor):
        self.data = None
        self.sensor = sensor
        self.sensor.listen(lambda data: self._update_data(data))

    def _update_data(self, data):
        self.data = data

    def show_image(self, screen:pygame.Surface):
        return

class Camera(Sensor):      
    def __init__(self, size:Tuple[int, int], init:Tuple[int, int], sensor:carla.Sensor):
        super().__init__(sensor=sensor)

        sub_screen = pygame.Surface(size)
        self.rect = sub_screen.get_rect(topleft=init)

    def show_image(self, screen: pygame.Surface):
        if self.data != None:
            array = np.frombuffer(self.data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.data.height, self.data.width, 4))

            # Swap blue and red channels
            array = array[:, :, (2, 1, 0)]

            # Reserve mirror effect
            image_surface = pygame.surfarray.make_surface(array)
            flipped_surface = pygame.transform.flip(image_surface, True, False)

            screen_surface = pygame.transform.scale(flipped_surface, self.rect.size)
            screen.blit(screen_surface, self.rect)
        
class Lidar(Sensor):    
    def __init__(self, size:Tuple[int, int], init:Tuple[int, int], sensor:carla.Sensor, scale:int):
        super().__init__(sensor=sensor)

        self.sub_screen = pygame.Surface(size)
        self.rect = self.sub_screen.get_rect(topleft=init)
        self.scale = scale

        self.min_thickness = 1
        self.max_thickness = 4
        self.color_min = (0, 0, 255)
        self.color_max = (255, 0, 0)
        self.color_screen = (0, 0, 0)
        self.size = size

    def show_image(self, screen: pygame.Surface):
        if self.data != None:
            lidar_data = np.copy(np.frombuffer(self.data.raw_data, dtype=np.dtype('f4')))
            lidar_data = np.reshape(lidar_data, (int(lidar_data.shape[0] / 4), 4))

            z_min = np.min(lidar_data[:, 2])
            z_max = np.max(lidar_data[:, 2])

            i_min = np.min(lidar_data[:, 3])
            i_max = np.max(lidar_data[:, 3])

            self.sub_screen.fill(self.color_screen)
            screen.blit(self.sub_screen, self.rect)

            for x, y, z, i in lidar_data:
                color = self._interpolate_color(num=i, min=i_min, max=i_max)
                thickness = self._interpolate_thickness(num=z, min=z_min, max=z_max)

                center = (int(x * self.scale + self.size[0] / 2), 
                          int(y * self.scale + self.size[1] / 2))
                pygame.draw.circle(self.sub_screen, color, center, thickness)

            screen.blit(self.sub_screen, self.rect)

    def _interpolate_thickness(self, num:float, min:float, max:float):
        norm = (num - min) / (max - min)
        thickness = int(self.min_thickness + (self.max_thickness - self.min_thickness) * norm)

        return thickness

    def _interpolate_color(self, num:float, min:float, max:float):
        norm = (num - min) / (max - min)
        
        r = int(self.color_min[0] + (self.color_max[0] - self.color_min[0]) * norm)
        g = int(self.color_min[1] + (self.color_max[1] - self.color_min[1]) * norm)
        b = int(self.color_min[2] + (self.color_max[2] - self.color_min[2]) * norm)

        return (r, g, b)

class Vehicle_sensors:
    def __init__(self, vehicle: carla.Vehicle, world: carla.World, screen: pygame.Surface):
        self.vehicle = vehicle
        self.world = world
        self.screen = screen
        self.sensors = []

    def add_sensor(self, sensor_type:str, size_rect:Tuple[int, int], init:Tuple[int, int]=(0, 0), 
                   transform:carla.Transform=carla.Transform(), scale_lidar:int=20):
        try:
            sensor_bp = self.world.get_blueprint_library().find(sensor_type)
        except IndexError:
            print("Sensor", sensor_type, "doesn't exist!")
            return None
                
        sensor = self.world.spawn_actor(sensor_bp, transform, attach_to=self.vehicle)
        if 'camera' in sensor_type:
            sensor_class = Camera(size=size_rect, init=init, sensor=sensor)
        elif 'lidar' in sensor_type:
            sensor_class = Lidar(size=size_rect, init=init, sensor=sensor, scale=scale_lidar)
        else:
            sensor_class = Sensor(sensor=sensor)

        self.sensors.append(sensor_class)
        return sensor_class

    def update_screen(self):
        for sensor in self.sensors:
            sensor.show_image(self.screen)

        pygame.display.flip()

    def destroy(self):
        for sensor in self.sensors:
            sensor.sensor.destroy()

        self.vehicle.destroy()

class Teleoperator:
    def __init__(self, vehicle:carla.Vehicle, steer:float=0.3, throttle:float=0.6, brake:float=1.0):
        self.vehicle = vehicle
        self.steer = max(0.0, min(1.0, steer))
        self.throttle = max(0.0, min(1.0, throttle))
        self.brake = max(0.0, min(1.0, brake))
        
    def control(self):
        control = carla.VehicleControl()
        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT]:
            control.steer = -self.steer
        if keys[pygame.K_RIGHT]:
            control.steer =  self.steer
        if keys[pygame.K_UP]:
            control.throttle = self.throttle
        if keys[pygame.K_DOWN]:
            control.brake = self.brake

        self.vehicle.apply_control(control)

    def set_steer(self, steer:float):
        self.steer = max(0.0, min(1.0, steer))

    def set_throttle(self, throttle:float):
        self.throttle = max(0.0, min(1.0, throttle))

    def set_brake(self, brake:float):
        self.brake = max(0.0, min(1.0, brake))

def setup_carla(port:int=2000, name_world:str='Town01'):
    client = carla.Client('localhost', port)
    world = client.get_world()
    client.load_world(name_world)

    return world, client

def add_one_vehicle(world:carla.World, ego_vehicle:bool, vehicle_type:str=None, 
                    tag:str='*vehicle*', transform:carla.Transform=None): 
    if transform == None:
        spawn_points = world.get_map().get_spawn_points()
        transform = random.choice(spawn_points)

    if vehicle_type == None:
        vehicle_bp = world.get_blueprint_library().filter(tag)
        try:
            vehicle_bp = random.choice(vehicle_bp)
        except IndexError:
            print("No vehicle of type", tag, "found!")
            return None
    else:
        try:
            vehicle_bp = world.get_blueprint_library().find(vehicle_type)
        except IndexError:
            print("Vehicle", vehicle_type, "doesn't exist!")
            return None

    if ego_vehicle:
        vehicle_bp.set_attribute('role_name', 'hero')

    vehicle = world.spawn_actor(vehicle_bp, transform)
    return vehicle

def center_spectator(world:carla.World, transform:carla.Transform,
                     scale:float=5.5, height:float=3.0, pitch:float=-10.0):
    yaw = math.radians(transform.rotation.yaw)
    spectator =  world.get_spectator()

    transform.location.z = height
    transform.location.x -= scale * math.cos(yaw)
    transform.location.y -= scale * math.sin(yaw)
    transform.rotation.pitch = pitch

    spectator.set_transform(transform)
    return spectator

def setup_pygame(size:Tuple[int, int], name:str):
    pygame.init()
    clock = pygame.time.Clock()

    screen = pygame.display.set_mode(size)
    pygame.display.set_caption(name)

    return screen, clock

def add_vehicles_randomly(world:carla.World, number:int):
    vehicle_bp = world.get_blueprint_library().filter('*vehicle*')
    spawn_points = world.get_map().get_spawn_points()

    vehicles = []
    for _ in range(number):
        v = world.try_spawn_actor(random.choice(vehicle_bp), random.choice(spawn_points))
        if v is not None:
            vehicles.append(v)

    return vehicles

def traffic_manager(client:carla.Client, vehicles:List[carla.Vehicle], port:int=5000, 
                    dist:float=3.0, speed_lower:float=10.0):
    tm = client.get_trafficmanager(port)
    tm_port = tm.get_port()

    for v in vehicles:
        v.set_autopilot(True, tm_port)
        tm.auto_lane_change(v, False) 

    speed_lower = max(0.0, min(100.0, speed_lower))
    tm.set_global_distance_to_leading_vehicle(dist)
    tm.global_percentage_speed_difference(speed_lower)

    return tm
