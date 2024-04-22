import carla
import pygame
import numpy as np
import math
import random
from queue import LifoQueue
import sys
import os
from PIL import Image, ImageDraw
import time

sys.path.insert(0, '/home/alumnos/lara/efficientvit-urjc/urjc')
sys.path.insert(0, '/home/alumnos/lara/efficientvit-urjc')
os.chdir('/home/alumnos/lara/efficientvit-urjc/urjc')

import EfficientVit as EV

NUM_ZONES = 3
LEFT = 0
FRONT = 1
RIGHT = 2

NUM_STATS = 4
MEAN = 0
MEDIAN = 1
STD = 2
MIN = 3

DIST = 0
Z = 1

ROAD = 0

def get_angle_range(angle:float):
    if angle > 180.0:
        angle -= 180.0 * 2
    elif angle < -180.0:
        angle += 180.0 * 2

    return angle

def write_text(text:str, img:pygame.Surface, point:tuple[int, int], bold:bool=False, 
               side:int=FRONT, size:int=50, color:tuple[int, int, int]=(255, 255, 255)):
    font = pygame.font.Font(pygame.font.match_font('tlwgtypo'), size)
    if bold:
        font.set_bold(True)
        
    text = font.render(text, True, color)
    text_rect = text.get_rect()

    if side == LEFT:
        text_rect.topleft = point 
    elif side == RIGHT:
        text_rect.topright = point
    else:
        text_rect.center = point

    img.blit(text, text_rect)

class Sensor:
    def __init__(self, sensor:carla.Sensor):
        self.sensor = sensor
        self.queue = LifoQueue()
        self.sensor.listen(lambda data: self.__callback_data(data))
        self.data = None

    def __callback_data(self, data):
        self.queue.put(data)

    def update_data(self):
        self.data = self.get_last_data()

        if self.data != None:
            return self.data.frame
        return 0

    def get_last_data(self):
        if not self.queue.empty():
            data = self.queue.get(False) # Non-blocking call 
            return data
        return None
    
    def process_data(self):
        pass

class CameraRGB(Sensor):      
    def __init__(self, size:tuple[int, int], init:tuple[int, int], sensor:carla.Sensor,
                 screen:pygame.Surface, seg:bool, init_seg:tuple[int, int], text:str):
        super().__init__(sensor=sensor)

        self.screen = screen
        self.text = text
        self.size_text = 20

        self.mask = []
        self.seg = seg
        if seg:
            self.seg_model = EV.EfficientVit(cuda_device="cuda:3", model="l2")

        self.sub_screen = pygame.Surface(size)
        self.rect_org = self.sub_screen.get_rect(topleft=init)
        self.rect_seg = self.sub_screen.get_rect(topleft=init_seg)

    def process_data(self):
        image = self.data
        if image == None:
            return

        image_data = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        image_data = np.reshape(image_data, (image.height, image.width, 4))

        # Swap blue and red channels
        image_data = image_data[:, :, (2, 1, 0)]

        # Reserve mirror effect
        image_surface = pygame.surfarray.make_surface(image_data)
        flipped_surface = pygame.transform.flip(image_surface, True, False)
        screen_surface = pygame.transform.scale(flipped_surface, self.rect_org.size)

        if self.text != None:
            write_text(text=self.text, img=screen_surface, color=(0, 0, 0), side=RIGHT, bold=True,
                       size=self.size_text, point=(self.rect_org.size[0], 0))
            
        self.screen.blit(screen_surface, self.rect_org)

        if self.seg:
            image_seg = Image.fromarray(image_data)

            # Create a canvas with the segmentation output
            pred = self.seg_model.predict(image_seg)
            canvas, self.mask = self.seg_model.get_canvas(np.array(image_seg), pred)
            canvas = Image.fromarray(canvas)

            surface_seg = pygame.image.fromstring(canvas.tobytes(), canvas.size, canvas.mode)
            surface_seg = pygame.transform.rotate(surface_seg, -90)
            surface_seg = pygame.transform.scale(surface_seg, self.rect_org.size)

            self.screen.blit(surface_seg, self.rect_seg)
    
    def get_deviation_road(self, rect_mask:pygame.Rect):
        if len(self.mask) == 0:
            return 0
        
        height_text = 10
        vehicle_color = (255, 0, 0)
        mask_color = (128, 64, 128)
        center_color = (0, 255, 0)

        height, width = self.mask.shape
        road_pixels = np.argwhere(self.mask == ROAD)

        image = Image.new('RGB', (width, height), color=0)
        draw = ImageDraw.Draw(image)
        for i, j in road_pixels:
            draw.point((j, i), fill=mask_color)

        if len(road_pixels) > 0:
            center_of_mass = np.mean(road_pixels, axis=0)
            y, x = center_of_mass
            deviation = y - height / 2
            dev_write = int(deviation)
        else:
            deviation = dev_write = np.nan

        if rect_mask != None:
            # Transform to pygame surface
            image = image.resize((rect_mask.height, rect_mask.width))
            image_data = image.tobytes()
            surface = pygame.image.fromstring(image_data, image.size, image.mode)

            if not np.isnan(deviation):
                # Scale center of mass
                x = int(x * rect_mask.height / width)
                y = int(y * rect_mask.width / height)

                # Draw center mass
                pygame.draw.line(surface, center_color, (0, y), (rect_mask.height, y), 1)
                pygame.draw.circle(surface, center_color, (x, y), 9)

            # Draw vehicle    
            pygame.draw.line(surface, vehicle_color, (0, int(rect_mask.width / 2)), 
                            (rect_mask.height, int(rect_mask.width / 2)), 1)

            # Rotation
            surface = pygame.transform.rotate(surface, -90)

            # Write text post rotation
            write_text(text="Deviation = "+str(abs(dev_write))+"(in pixels)", img=surface, point=(0, 0), 
                       side=LEFT, size=self.size_text, color=(255, 255, 255), bold=True)
            write_text(text="vehicle", img=surface, point=(rect_mask.width / 2 + 2, height_text), 
                       side=LEFT, size=self.size_text, color=vehicle_color)
            
            if not np.isnan(deviation):
                write_text(text="center of mass", point=(rect_mask.width - y + 2, height_text * 4),
                           side=LEFT, img=surface, size=self.size_text, color=center_color)

            self.screen.blit(surface, rect_mask)
            
        return deviation
        
class Lidar(Sensor): 
    def __init__(self, size:tuple[int, int], init:tuple[int, int], sensor:carla.Sensor, scale:int,
                 front_angle:int, yaw:float, screen:pygame.Surface, show_stats:bool=True):
        super().__init__(sensor=sensor)

        self.sub_screen = pygame.Surface(size)
        self.rect = self.sub_screen.get_rect(topleft=init)
        self.scale = scale
        self.size_screen = size
        self.screen = screen

        # Visualize lidar
        self.min_thickness = 2
        self.max_thickness = math.ceil(scale / 10) + self.min_thickness * 2
        self.color_min = (0, 0, 255)
        self.color_max = (255, 0, 0)

        # Calculate stats
        self.i_threshold = 0.987
        self.z_threshold = -1.6
        self.stat_zones = np.full((NUM_ZONES, NUM_STATS), 100.0) 
        self.meas_zones = None

        # Write stats
        self.show_stats = show_stats
        y = size[1] / 2
        if show_stats:
            y += self.scale * 1.5
        self.center_screen = (int(size[0] / 2), int(y))

        # Update per second
        self.time = -2
        self.stat_text = None

        # Write text
        self.size_text = min(int(self.scale / 1.5), 20)
        self.x_text = (self.max_thickness, self.center_screen[0], size[0] - self.max_thickness)

        # Select front zone
        self.front_angle = abs(front_angle)
        if self.front_angle > 360:
            self.front_angle = 360
        
        # Divide front zone
        angle1 = get_angle_range(-self.front_angle / 2 - yaw)
        angle2 = get_angle_range(self.front_angle / 2 - yaw)
        angle1_add = get_angle_range(angle1 + self.front_angle / 3)
        angle2_sub = get_angle_range(angle2 - self.front_angle / 3)        
        self.angles = [angle1, angle1_add, angle2_sub, angle2]

        self.image = self.__get_back_image()

    def __get_back_image(self):
        image = pygame.Surface(self.size_screen)
        mult = 10 * self.scale
        text = ['FL', 'FF', 'FR']
        text_zone = ['Front-Left', 'Front-Front', 'Front-Right']

        for i in range(len(self.angles)):
            x_line = self.center_screen[0] + mult * math.cos(math.radians(self.angles[i]))
            y_line = self.center_screen[1] + mult * math.sin(math.radians(self.angles[i]))
            pygame.draw.line(image, (70, 70, 70), self.center_screen, (x_line, y_line), 2)

            if i < NUM_ZONES:
                angle = self.angles[i] + self.front_angle / 6
                x_zone = self.center_screen[0] + mult * math.cos(math.radians(angle))
                y_zone = max(self.center_screen[1] + mult * math.sin(math.radians(angle)), 
                             self.size_text * (NUM_STATS + 2)) # Make sure to write behind the text
                
                write_text(text=text[i], img=image, point=(x_zone, y_zone), bold=True, size=self.size_text)

                if self.show_stats:
                    write_text(text=text_zone[i], img=image, side=i, color=(255, 165, 180), bold=True,
                            point=(self.x_text[i], self.size_text), size=self.size_text)
        return image
    
    def __interpolate_thickness(self, num:float, min:float, max:float):
        norm = (num - min) / (max - min)
        thickness = self.min_thickness + (self.max_thickness - self.min_thickness) * norm

        return thickness

    def __interpolate_color(self, num:float, min:float, max:float):
        norm = (num - min) / (max - min)
        
        r = int(self.color_min[0] + (self.color_max[0] - self.color_min[0]) * norm)
        g = int(self.color_min[1] + (self.color_max[1] - self.color_min[1]) * norm)
        b = int(self.color_min[2] + (self.color_max[2] - self.color_min[2]) * norm)

        return (r, g, b)
    
    def __update_stats(self):
        for zone in range(NUM_ZONES):
            if len(self.meas_zones[DIST][zone]) != 0:
                # Filter distances by z
                filter = np.array(self.meas_zones[Z][zone]) > self.z_threshold
                filtered_dist = np.array(self.meas_zones[DIST][zone])[filter]

                if len(filtered_dist) == 0:
                    self.stat_zones[zone][MIN] = np.nan
                else:
                    self.stat_zones[zone][MIN] = np.min(filtered_dist)

                self.stat_zones[zone][MEAN] = np.mean(self.meas_zones[DIST][zone])
                self.stat_zones[zone][MEDIAN] = np.median(self.meas_zones[DIST][zone])
                self.stat_zones[zone][STD] = np.std(self.meas_zones[DIST][zone])
            else:
                for i in range(NUM_STATS):
                    self.stat_zones[zone][i] = np.nan

            if self.show_stats:
                if time.time() - self.time > 1:
                    self.time = time.time()
                    self.stats_text = [
                        "Mean = {:.2f}".format(self.stat_zones[zone][MEAN]),
                        "Median = {:.2f}".format(self.stat_zones[zone][MEDIAN]),
                        "Std = {:.2f}".format(self.stat_zones[zone][STD]),
                        "Min(z>{:.1f}) = {:.2f}".format(self.z_threshold, self.stat_zones[zone][MIN])
                    ]

                # Write stats
                y = self.size_text * 2
                for text in self.stats_text:
                    write_text(text=text, point=(self.x_text[zone], y), img=self.sub_screen, 
                               side=zone, size=self.size_text)
                    y += self.size_text

    def __in_zone(self, zone:int, angle:float):
        if self.angles[zone] <= self.angles[zone + 1]:
            return self.angles[zone] <= angle <= self.angles[zone + 1]
        else:
            return self.angles[zone] <= angle or angle <= self.angles[zone + 1]

    def __get_zone(self, x:float, y:float):
        angle = np.arctan2(y, x) * 180 / np.pi 

        for zone in range(NUM_ZONES):
            if self.__in_zone(zone, angle):
                return zone

        return NUM_ZONES

    def process_data(self):
        lidar = self.data
        if lidar == None:
            return
        
        lidar_data = np.copy(np.frombuffer(lidar.raw_data, dtype=np.dtype('f4')))
        lidar_data = np.reshape(lidar_data, (int(lidar_data.shape[0] / 4), 4))

        z_min = np.min(lidar_data[:, 2])
        z_max = np.max(lidar_data[:, 2])

        i_min = np.min(lidar_data[:, 3])
        i_max = np.max(lidar_data[:, 3])

        self.sub_screen.blit(self.image, (0, 0))

        dist_zones = [[] for _ in range(NUM_ZONES)]
        z_zones = [[] for _ in range(NUM_ZONES)]

        for x, y, z, i in lidar_data:
            zone = self.__get_zone(x=x, y=y)
            if zone < NUM_ZONES and i < self.i_threshold:
                dist_zones[zone].append(math.sqrt(x ** 2 + y ** 2))
                z_zones[zone].append(z)

            thickness = self.__interpolate_thickness(num=z, min=z_min, max=z_max)
            color = self.__interpolate_color(num=i, min=i_min, max=i_max)
            center = (int(x * self.scale + self.center_screen[0]),
                      int(y * self.scale + self.center_screen[1]))

            pygame.draw.circle(self.sub_screen, color, center, thickness)

        self.meas_zones = [dist_zones, z_zones]
        self.__update_stats()            
        self.screen.blit(self.sub_screen, self.rect)
    
    def set_i_threshold(self, i:float):
        self.i_threshold = i

    def get_i_threshold(self):
        return self.i_threshold
    
    def set_z_threshold(self, z:float):
        self.z_threshold = z

    def get_z_threshold(self):
        return self.z_threshold
    
    def get_stat_zones(self):
        return self.stat_zones
    
    def get_meas_zones(self):
        return self.meas_zones

class Vehicle_sensors:
    def __init__(self, vehicle:carla.Vehicle, world:carla.World, screen:pygame.Surface):
        self.vehicle = vehicle
        self.world = world
        self.screen = screen
        self.sensors = []

        self.time_frame = -1.0
        self.count_frame = -1
        self.write_frame = 0

    def __put_sensor(self, sensor_type:str, transform:carla.Transform, lidar:bool=False):
        try:
            sensor_bp = self.world.get_blueprint_library().find(sensor_type)
        except IndexError:
            print("Sensor", sensor_type, "doesn't exist!")
            return None
        
        if lidar:
            sensor_bp.set_attribute('rotation_frequency', '20')
                
        return self.world.spawn_actor(sensor_bp, transform, attach_to=self.vehicle)

    def add_sensor(self, sensor_type:str, transform:carla.Transform=carla.Transform()):
        sensor = self._put_sensor(sensor_type=sensor_type, transform=transform)
        sensor_class = Sensor(sensor=sensor)
        self.sensors.append(sensor_class)
        return sensor_class
    
    def add_camera_rgb(self, size_rect:tuple[int, int], init:tuple[int, int]=(0, 0), seg:bool=False,
                       transform:carla.Transform=carla.Transform(), init_seg:tuple[int, int]=(0, 0),
                       text:str=None):
        sensor = self.__put_sensor(sensor_type='sensor.camera.rgb', transform=transform)
        camera = CameraRGB(size=size_rect, init=init, sensor=sensor, screen=self.screen, 
                           seg=seg, init_seg=init_seg, text=text)
        self.sensors.append(camera)
        return camera
    
    def add_lidar(self, size_rect:tuple[int, int], init:tuple[int, int]=(0, 0), scale_lidar:int=25,
                  transform:carla.Transform=carla.Transform(), front_angle:int=150, show_stats:bool=True):
        sensor = self.__put_sensor(sensor_type='sensor.lidar.ray_cast', transform=transform, lidar=True)
        lidar = Lidar(size=size_rect, init=init, sensor=sensor, front_angle=front_angle, scale=scale_lidar,
                      yaw=transform.rotation.yaw, screen=self.screen, show_stats=show_stats)
        
        self.sensors.append(lidar)
        return lidar

    def update_data(self, flip:bool=True):
        # Pick data in the same frame
        for sensor in self.sensors:
            frame = sensor.update_data()

        for sensor in self.sensors:
            sensor.process_data()

        if time.time() - self.time_frame > 1: 
            self.time_frame = time.time()
            self.write_frame = frame - self.count_frame
            self.count_frame = frame

        write_text(text="FPS: "+str(self.write_frame), img=self.screen, color=(0, 0, 0),
                   bold=True, point=(2, 0), size=23, side=LEFT)
        
        if flip:
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

    def get_steer(self):
        return self.steer
    
    def get_throttle(self):
        return self.throttle
    
    def get_brake(self):
        return self.brake

class PID:
    def __init__(self, vehicle:carla.Vehicle):
        self.throttle = 0.4
        self.vehicle = vehicle

        # Max error is 300
        self.kp = -1/300
        self.kd = 0
        self.ki = 0

        self.error = 0
        self.prev_error = 0
        self.total_error = 0

    def controll_vehicle(self, error:float):
        self.prev_error = self.error
        self.error = error
        self.total_error += error

        control = carla.VehicleControl()
        control.throttle = self.throttle

        if error == np.nan:
            w = 0
        else:
            w = self.kp * self.error
        print(w) 
        control.steer = w
        self.vehicle.apply_control(control)

def setup_carla(port:int=2000, name_world:str='Town01', delta_seconds=0.1):
    client = carla.Client('localhost', port)
    world = client.get_world()
    world = client.load_world(name_world)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = delta_seconds
    world.apply_settings(settings)

    return world, client

def add_one_vehicle(world:carla.World, ego_vehicle:bool=False, vehicle_type:str=None, 
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

def setup_pygame(size:tuple[int, int], name:str):
    pygame.init()
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption(name)

    return screen

def add_vehicles_randomly(world:carla.World, number:int):
    vehicle_bp = world.get_blueprint_library().filter('*vehicle*')
    spawn_points = world.get_map().get_spawn_points()

    vehicles = []
    for _ in range(number):
        v = world.try_spawn_actor(random.choice(vehicle_bp), random.choice(spawn_points))
        if v is not None:
            vehicles.append(v)

    return vehicles

def traffic_manager(client:carla.Client, vehicles:list[carla.Vehicle], port:int=5000):
    tm = client.get_trafficmanager(port)
    tm_port = tm.get_port()

    for v in vehicles:
        v.set_autopilot(True, tm_port)
        tm.auto_lane_change(v, False) 

    return tm