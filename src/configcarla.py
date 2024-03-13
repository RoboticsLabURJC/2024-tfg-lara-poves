import carla
import pygame
import numpy as np
import math
import random
from typing import Tuple, List

class Sensor:
    def __init__(self, size:Tuple[int, int], init:Tuple[int, int], sensor:carla.Sensor, 
                 scale:int, color_draw:Tuple[int, int, int], color_screen:Tuple[int, int, int]):
        self.data = None
        self.sensor = sensor
        self.scale = scale
        self.color_draw = color_draw
        self.color_screen = color_screen

        self.init = (init[0] + size[0] / 2, init[1] + size[1] / 2)
        self.sub_screen = pygame.Surface(size)
        self.rect = self.sub_screen.get_rect(topleft=init)

        self.sensor.listen(lambda data: self._update_data(data))

    def _update_data(self, data):
        self.data = data
    
    def show_image(self, screen:pygame.Surface):
        if self.data == None:
            return
        
        if isinstance(self.data, carla.Image):
            array = np.frombuffer(self.data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.data.height, self.data.width, 4))

            # Swap blue and red channels
            array = array[:, :, (2, 1, 0)]

            # Reserve mirror effect
            image_surface = pygame.surfarray.make_surface(array)
            flipped_surface = pygame.transform.flip(image_surface, True, False)

            screen_surface = pygame.transform.scale(flipped_surface, self.rect.size)
            screen.blit(screen_surface, self.rect)

        elif isinstance(self.data, carla.LidarMeasurement):
            lidar_data = np.copy(np.frombuffer(self.data.raw_data, dtype=np.dtype('f4')))
            lidar_data = np.reshape(lidar_data, (int(lidar_data.shape[0] / 4), 4))
            points = lidar_data[:, :-1]

            self.sub_screen.fill(self.color_screen)
            screen.blit(self.sub_screen, self.rect)

            for x, y, _ in points:
                center = (int(x * self.scale + self.init[0]), int(y * self.scale + self.init[1]))
                pygame.draw.circle(screen, self.color_draw, center, 1)

class Vehicle_sensors:
    def __init__(self, vehicle: carla.Vehicle, world: carla.World, screen: pygame.Surface):
        self.vehicle = vehicle
        self.world = world
        self.screen = screen
        self.sensors = []

    def add_sensor(self, sensor:str, size_rect:Tuple[int, int], init:Tuple[int, int]=(0, 0), 
                   transform:carla.Transform=carla.Transform(), scale_lidar:int=20,
                   color_screen:Tuple[int, int, int]=(0, 0, 0), 
                   color_draw:Tuple[int, int, int]=(0, 255, 0)):
        try:
            sensor_bp = self.world.get_blueprint_library().find(sensor)
        except IndexError:
            print("Sensor", sensor, "doesn't exist!")
            return None
                
        sensor = self.world.spawn_actor(sensor_bp, transform, attach_to=self.vehicle)
        sensor_class = Sensor(size=size_rect, init=init, sensor=sensor, scale=scale_lidar, 
                              color_screen=color_screen, color_draw=color_draw)
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
        vehicle_bp = random.choice(vehicle_bp)
    else:
        try:
            vehicle_bp = world.get_blueprint_library().find(vehicle_type)
        except IndexError:
            print("Vehicle", vehicle_type, "doesn't exist!")
            return None, transform

    if ego_vehicle:
        vehicle_bp.set_attribute('role_name', 'hero')

    vehicle = world.spawn_actor(vehicle_bp, transform)
    return vehicle, transform

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
