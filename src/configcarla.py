import carla
import pygame
import numpy as np
import math
import random
from typing import Tuple, List
from queue import Queue

LEFT = 0
FRONT = 1
RIGHT = 2

MIN = 0
MEAN = 1
MEDIAN = 2
STD = 3

def _get_angle_range(angle:float):
    if angle > 180.0:
        angle -= 180.0 * 2
    elif angle < -180.0:
        angle += 180.0 * 2

    return angle

def _same_sign(angle1, angle2):
    return (angle1 >= 0 and angle2 >= 0) or (angle1 <= 0 and angle2 <= 0)

class Sensor():
    def __init__(self, sensor:carla.Sensor):
        self.sensor = sensor
        self.queue = Queue()
        self.sensor.listen(lambda data: self._update_data(data))

    def _update_data(self, data):
        self.queue.put(data)

    def get_last_data(self):
        if not self.queue.empty():
            data = self.queue.get(False) # Non-blocking call 
            return data
        return None
    
    def process_data(self):
        pass

class CameraRGB(Sensor):      
    def __init__(self, size:Tuple[int, int], init:Tuple[int, int], 
                 sensor:carla.Sensor, screen:pygame.Surface):
        super().__init__(sensor=sensor)

        sub_screen = pygame.Surface(size)
        self.rect = sub_screen.get_rect(topleft=init)
        self.screen = screen

    def process_data(self):
        image = self.get_last_data()
        
        if image != None:
            image_data = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            image_data = np.reshape(image_data, (image.height, image.width, 4))

            # Swap blue and red channels
            image_data = image_data[:, :, (2, 1, 0)]

            # Reserve mirror effect
            image_surface = pygame.surfarray.make_surface(image_data)
            flipped_surface = pygame.transform.flip(image_surface, True, False)

            screen_surface = pygame.transform.scale(flipped_surface, self.rect.size)
            self.screen.blit(screen_surface, self.rect)
        
class Lidar(Sensor): 
    def __init__(self, size:Tuple[int, int], init:Tuple[int, int], sensor:carla.Sensor,
                 scale:int, front_angle:int, yaw:float, screen:pygame.Surface):
        super().__init__(sensor=sensor)

        self.sub_screen = pygame.Surface(size)
        self.rect = self.sub_screen.get_rect(topleft=init)
        self.scale = scale
        self.size_screen = size
        self.center_screen = (size[0] // 2, size[1] // 2 + self.scale)
        self.screen = screen

        # Visualize lidar
        self.min_thickness = 2
        self.max_thickness = math.ceil(scale / 10) + self.min_thickness
        self.color_min = (0, 0, 255)
        self.color_max = (255, 0, 0)

        # Select front zone
        self.yaw = abs(yaw)
        self.front_angle = abs(front_angle)
        if self.front_angle > 360:
            self.front_angle = 360
        
        # Divide front zone
        angle1 = _get_angle_range(-self.front_angle / 2 - yaw)
        angle4 = _get_angle_range(self.front_angle / 2 - yaw)
        angle2 = _get_angle_range(angle1 + self.front_angle / 3)
        angle3 = _get_angle_range(angle4 - self.front_angle / 3)        
        self.angles = [angle1, angle2, angle3, angle4]

        # Detect obstacles
        self.stat_zones = np.full((3, 4), 100.0) 
        self.std_min = 0.0

        # Write data
        self.color_text = (255, 255, 255)
        self.size_text = self.scale // 2
        self.x_text = (self.max_thickness, self.center_screen[0], size[0] - self.max_thickness) 
        self.image = self._get_back_image()

    def _write_text(self, text:str, img:pygame.Surface, point:Tuple[int, int],
                    side:int=FRONT, bold:bool=False):
        font = pygame.font.Font(pygame.font.match_font('tlwgtypo'), self.size_text)
        if bold:
            font.set_bold(True)

        text = font.render(text, True, self.color_text)
        text_rect = text.get_rect()

        if side == LEFT:
            text_rect.topleft = point 
        elif side == RIGHT:
            text_rect.topright = point
        else:
            text_rect.center = point

        img.blit(text, text_rect)

    def _get_back_image(self):
        image = pygame.Surface(self.size_screen)
        mult_line = 10 * self.scale
        mult_zone = mult_line - self.scale

        text_column = ['front-left', 'front-front', 'front-right']
        text_zone = ['FL', 'FF', 'FR']

        for i in range(len(self.angles)):
            x_line = self.center_screen[0] + mult_line * math.cos(math.radians(self.angles[i]))
            y_line = self.center_screen[1] + mult_line * math.sin(math.radians(self.angles[i]))
            pygame.draw.line(image, (70, 70, 70), self.center_screen, (x_line, y_line), 2)

            if i < len(self.angles) - 1:
                dif = _get_angle_range(self.angles[i] - self.angles[i+1])
                angle = self.angles[i + 1] + dif / 2

                x_zone = self.center_screen[0] + mult_zone * math.cos(math.radians(angle))
                y_zone = self.center_screen[1] + mult_zone * math.sin(math.radians(angle))

                self._write_text(text=text_zone[i], img=image, point=(x_zone, y_zone), bold=True)
                self._write_text(text=text_column[i], img=image, side=i,
                                 point=(self.x_text[i], self.size_text))

        return image
    
    def _interpolate_thickness(self, num:float, min:float, max:float):
        norm = (num - min) / (max - min)
        thickness = self.min_thickness + (self.max_thickness - self.min_thickness) * norm

        return thickness

    def _interpolate_color(self, num:float, min:float, max:float):
        norm = (num - min) / (max - min)
        
        r = int(self.color_min[0] + (self.color_max[0] - self.color_min[0]) * norm)
        g = int(self.color_min[1] + (self.color_max[1] - self.color_min[1]) * norm)
        b = int(self.color_min[2] + (self.color_max[2] - self.color_min[2]) * norm)

        return (r, g, b)
    
    def _update_stats(self, dist:List[float], y:List[float], zone:int):
        if len(dist) != 0:
            self.stat_zones[zone][MIN] = np.min(dist)
            self.stat_zones[zone][MEAN] = np.mean(dist)
            self.stat_zones[zone][MEDIAN] = np.median(dist)
        if len(y) != 0:  
            '''
            aqui si es 0 no se deberia mantener si se
            filtra por intensidad
            '''
            self.stat_zones[zone][STD] = np.std(y)

        y = self.size_text * 2
        stats_text = [
            "Min = {:.2f}".format(self.stat_zones[zone][MIN]),
            "Mean = {:.2f}".format(self.stat_zones[zone][MEAN]),
            "Median = {:.2f}".format(self.stat_zones[zone][MEDIAN]),
            "Std = {:.2f}".format(self.stat_zones[zone][STD])
        ]
        
        for text in stats_text:
            self._write_text(text=text, img=self.sub_screen, 
                             point=(self.x_text[zone], y), side=zone)
            y += self.size_text

    def _in_zone(self, zone:int, angle:float):
        if self.angles[zone] <= self.angles[zone + 1]:
            return self.angles[zone] <= angle <= self.angles[zone + 1]
        else:
            return self.angles[zone] <= angle or angle <= self.angles[zone + 1]

    def _get_zone(self, x:float, y:float):
        angle = np.arctan2(y, x) * 180 / np.pi 

        for zone in range(3):
            if self._in_zone(zone, angle):
                return zone

        return 10

    def process_data(self):
        if not self.queue.empty():
            lidar = self.queue.get(False) 
            lidar_data = np.copy(np.frombuffer(lidar.raw_data, dtype=np.dtype('f4')))
            lidar_data = np.reshape(lidar_data, (int(lidar_data.shape[0] / 4), 4))

            z_min = np.min(lidar_data[:, 2])
            z_max = np.max(lidar_data[:, 2])

            i_min = np.min(lidar_data[:, 3])
            i_max = np.max(lidar_data[:, 3])

            self.sub_screen.blit(self.image, (0, 0))

            zones_y = [[], [], []]
            zones_dist = [[], [], []]

            for x, y, z, i in lidar_data:
                zone = self._get_zone(x=x, y=y)
                if zone < len(zones_y): # quitar la zona roja
                    zones_dist[zone].append(math.sqrt(x ** 2 + y ** 2))
                    zones_y[zone].append(y) # limitar por z

                thickness = self._interpolate_thickness(num=z, min=z_min, max=z_max)
                color = self._interpolate_color(num=i, min=i_min, max=i_max)

                center = (int(x * self.scale + self.center_screen[0]), 
                        int(y * self.scale + self.center_screen[1]))
                pygame.draw.circle(self.sub_screen, color, center, thickness)

            for i in range(len(zones_y)):
                self._update_stats(dist=zones_dist[i], y=zones_y[i], zone=i)

            self.screen.blit(self.sub_screen, self.rect)

    def obstacle_front_right(self):
        return self.std_min >= self.std_zones[RIGHT]

    def obstacle_front_left(self):
        return self.std_min >= self.std_zones[LEFT]
    
    def obstacle_front_front(self):
        return self.std_min >= self.std_zones[FRONT]

class Vehicle_sensors:
    def __init__(self, vehicle:carla.Vehicle, world:carla.World, screen:pygame.Surface):
        self.vehicle = vehicle
        self.world = world
        self.screen = screen
        self.sensors = []

    def _put_sensor(self, sensor_type:str, transform:carla.Transform):
        try:
            sensor_bp = self.world.get_blueprint_library().find(sensor_type)
        except IndexError:
            print("Sensor", sensor_type, "doesn't exist!")
            return None
                
        sensor = self.world.spawn_actor(sensor_bp, transform, attach_to=self.vehicle)
        return sensor

    def add_sensor(self, sensor_type:str, transform:carla.Transform=carla.Transform()):
        sensor = self._put_sensor(sensor_type=sensor_type, transform=transform)
        sensor_class = Sensor(sensor=sensor)
        self.sensors.append(sensor_class)
        return sensor_class
    
    def add_camera_rgb(self, size_rect:Tuple[int, int], init:Tuple[int, int]=(0, 0), 
                       transform:carla.Transform=carla.Transform()):
        sensor = self._put_sensor(sensor_type='sensor.camera.rgb', transform=transform)
        camera = CameraRGB(size=size_rect, init=init, sensor=sensor, screen=self.screen)
        self.sensors.append(camera)
        return camera
    
    def add_lidar(self, size_rect:Tuple[int, int], init:Tuple[int, int]=(0, 0), scale_lidar:int=25,
                  transform:carla.Transform=carla.Transform(), front_angle:int=150):
        sensor = self._put_sensor(sensor_type='sensor.lidar.ray_cast', transform=transform)
        lidar = Lidar(size=size_rect, init=init, sensor=sensor, front_angle=front_angle,
                      scale=scale_lidar, yaw=transform.rotation.yaw, screen=self.screen)
        
        self.sensors.append(lidar)
        return lidar

    def update_data(self):
        for sensor in self.sensors:
            sensor.process_data()

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
