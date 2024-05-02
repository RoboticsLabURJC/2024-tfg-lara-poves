import carla
import pygame
import numpy as np
import math
import random
from queue import LifoQueue
import sys
import os
from PIL import Image
import time
import torch
import threading
import cv2

sys.path.insert(0, '/home/alumnos/lara/efficientvit-urjc/urjc')
sys.path.insert(0, '/home/alumnos/lara/efficientvit-urjc')
os.chdir('/home/alumnos/lara/efficientvit-urjc/urjc')

import EfficientVit as EV

SEG_TO_NANOSEG = 1000000000

SIZE_CAMERA = 512

# Type
CAMERA = 1
LIDAR = 2

# Side
NUM_ZONES = 3
LEFT = 0
FRONT = 1
RIGHT = 2

# Stats
NUM_STATS = 4
MEAN = 0
MEDIAN = 1
STD = 2
MIN = 3

# Measurements
DIST = 0
Z = 1

# Segmentation
ROAD = 0

# Global variables
lock_sensor = threading.Lock()

def get_angle_range(angle:float):
    if angle > 180.0:
        angle -= 180.0 * 2
    elif angle < -180.0:
        angle += 180.0 * 2

    return angle

def write_text(text:str, img:pygame.Surface, point:tuple[int, int], bold:bool=False, side:int=FRONT, 
               size:int=50, color:tuple[int, int, int]=(255, 255, 255), background:tuple[int, int, int]=None):
    font = pygame.font.Font(pygame.font.match_font('tlwgtypo'), size)
    if bold:
        font.set_bold(True)
        
    text = font.render(text, True, color)
    text__rect = text.get_rect()

    if side == LEFT:
        text__rect.topleft = point 
    elif side == RIGHT:
        text__rect.topright = point
    else:
        text__rect.center = point

    if background != None:
        pygame.draw.rect(img, background, text__rect)

    img.blit(text, text__rect)

class Sensor:
    def __init__(self, sensor:carla.Sensor):
        self.sensor = sensor
        self.queue = LifoQueue()
        self.sensor.listen(lambda data: self.__callback_data(data))

    def __callback_data(self, data):
        self.queue.put(data)

    def get_last_data(self):
        if not self.queue.empty():
            data = self.queue.get(False) # Non-blocking call 
            return data
        return None

    def process_data(self):
        pass

class CameraRGB(Sensor):      
    def __init__(self, size:tuple[int, int], init:tuple[int, int], sensor:carla.Sensor, text:str,
                 screen:pygame.Surface, seg:bool, init_extra:tuple[int, int], lane:bool):
        super().__init__(sensor=sensor)

        self.__screen = screen
        self.text = text
        self.__size_text = 20
        self.__deviation = 0

        self.__seg = seg
        if seg:
            self.__seg_model = EV.EfficientVit()

        self.__lane = lane
        if lane:
            file = '/home/alumnos/lara/carla_lane_detector/examples/fastai_torch_lane_detector_model.pth'
            self.__lane_model = torch.load(file)
            self.__lane_model.eval()

            self.__threshold_lane = 0.25

        self.__rect_org = init
        self.__rect_extra = init_extra

        if init_extra != None or init != None:
            assert size != None, "size is required!"
            sub_screen = pygame.Surface(size)

            if init != None:
                self.__rect_org = sub_screen.get_rect(topleft=init)

            if init_extra != None:
                self.__rect_extra = sub_screen.get_rect(topleft=init_extra)

    def __masks_lane(self, image:list, limits_lane:list):    
        # Resize for the network
        image_lane = np.zeros((SIZE_CAMERA, SIZE_CAMERA * 2, 3), dtype=np.uint8)
        image_lane[:SIZE_CAMERA, :SIZE_CAMERA, :] = image # Copy the image in the top left corner

        with torch.no_grad():
            image_tensor = image_lane.transpose(2,0,1).astype('float32')/255
            x_tensor = torch.from_numpy(image_tensor).to("cuda").unsqueeze(0)
            model_output = torch.softmax(self.__lane_model.forward(x_tensor), dim=1 ).cpu().numpy()
        
        _, left_mask, right_mask = model_output[0]
        left_mask = left_mask[:, :512]
        right_mask = right_mask[:, :512]

        limits_lane[:, 0] = SIZE_CAMERA
        for y in range(image.shape[0]):
            x_left_index = np.argwhere(left_mask[y] > self.__threshold_lane)
            if len(x_left_index) != 0:
                limits_lane[y, 0] = x_left_index[0][0]
            else:
                continue

            x_right_index = np.argwhere(right_mask[y] > self.__threshold_lane)
            if len(x_right_index) != 0:
                limits_lane[y, 1] = x_right_index[-1][0]

    def __process_seg(self, data:list):
        image_data = cv2.rotate(data, cv2.ROTATE_90_CLOCKWISE)
        image_seg = Image.fromarray(image_data)

        if self.__lane:
            limits_lane = np.zeros((SIZE_CAMERA, 2), dtype=int)
            thread = threading.Thread(target=self.__masks_lane, args=(image_data, limits_lane))
            thread.start()

        # Prediction
        pred = self.__seg_model.predict(image_seg)
        canvas, mask = self.__seg_model.get_canvas(np.array(image_seg), pred)

        if self.__lane:
            thread.join()
            
            road_pixels = []
            y_cm = count = 0

            for y in range(canvas.shape[0]):
                if limits_lane[y, 0] > limits_lane[y, 1]:
                    continue

                road_pixels.extend(range(limits_lane[y, 0], limits_lane[y, 1] + 1))

                if self.__rect_extra != None:
                    canvas[y, limits_lane[y, 0] : limits_lane[y, 1] + 1] = [255, 240, 255]
                    y_cm += y
                    count += 1

            # Calculate center of mass
            if len(road_pixels) != 0:
                x_cm = np.mean(road_pixels, axis=0)
            else:
                x_cm = np.nan

            if np.isnan(x_cm):
                self.__deviation = 0
            else:
                self.__deviation = x_cm - SIZE_CAMERA / 2
                y_cm = int(y_cm / count)
                x_cm = int(x_cm)
                middle = int(SIZE_CAMERA / 2)
                color_cm = (0, 255, 0)

                # Draw center of mass and vehicle
                cv2.line(canvas, (x_cm, 0), (x_cm, SIZE_CAMERA), color_cm, 2)
                cv2.line(canvas, (middle, 0), (middle, SIZE_CAMERA), (255, 0, 0), 2)
                cv2.circle(canvas, (x_cm, y_cm), 9, color_cm, thickness=-1)

        if self.__rect_extra != None:
            # Convert to pygame syrface
            canvas = Image.fromarray(canvas)
            surface_seg = pygame.image.fromstring(canvas.tobytes(), canvas.size, canvas.mode)
            surface_seg = pygame.transform.scale(surface_seg, self.__rect_extra.size)

            # Write text
            write_text(text="Segmented "+self.text, img=surface_seg, color=(0, 0, 0), side=RIGHT,
                       bold=True, size=self.__size_text, point=(self.__rect_extra.size[0], 0))
            if self.__lane:
                write_text(text="Deviation = "+str(int(abs(self.__deviation)))+" (pixels)", img=surface_seg,
                           bold=True, color=(0, 0, 0), side=LEFT, size=self.__size_text, 
                           point=(0, self.__rect_extra.size[0] - self.__size_text))

            lock_sensor.acquire()
            self.__screen.blit(surface_seg, self.__rect_extra)
            lock_sensor.release()

    def get_deviation(self):
        return self.__deviation

    def process_data(self):
        image = self.get_last_data()
        if image == None:
            return 

        image_data = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        image_data = np.reshape(image_data, (image.height, image.width, 4))

        # Swap blue and red channels
        image_data = image_data[:, :, (2, 1, 0)]

        if self.__seg:
            self.__process_seg(image_data)

        if self.__rect_org != None:
            # Reserve mirror effect
            image_surface = pygame.surfarray.make_surface(image_data)
            flipped_surface = pygame.transform.flip(image_surface, True, False)
            screen_surface = pygame.transform.scale(flipped_surface, self.__rect_org.size)

            if self.text != None:
                write_text(text=self.text, img=screen_surface, color=(0, 0, 0), side=RIGHT, bold=True,
                           size=self.__size_text, point=(self.__rect_org.size[0], 0))

            lock_sensor.acquire()
            self.__screen.blit(screen_surface, self.__rect_org)
            lock_sensor.release()
        
class Lidar(Sensor): 
    def __init__(self, size:tuple[int, int], init:tuple[int, int], sensor:carla.Sensor, scale:int,
                 front_angle:int, yaw:float, screen:pygame.Surface, show_stats:bool=True):
        super().__init__(sensor=sensor)

        self.__rect = init
        self.show_stats = show_stats

        if init != None:
            assert size != None, "size is required!"
            self.__sub_screen = pygame.Surface(size)
            self.__rect = self.__sub_screen.get_rect(topleft=init)

            self.__scale = scale
            self.__size_screen = size
            self.__screen = screen

            # Visualize lidar
            self.__min_thickness = 2
            self.__max_thickness = math.ceil(scale / 10) + self.__min_thickness * 3
            self.__color_min = (0, 0, 255)
            self.__color_max = (255, 0, 0)

            # Write stats
            y = size[1] / 2
            if show_stats:
                y += scale * 1.5
            self.__center_screen = (int(size[0] / 2), int(y))

            # Write text
            self.__size_text = min(int(scale / 1.5), 20)
            self.__x_text = (self.__max_thickness, self.__center_screen[0], size[0] - self.__max_thickness)

        # Calculate stats
        self.__i_threshold = 0.987
        self.__z_threshold = -1.6
        self.__stat_zones = np.full((NUM_ZONES, NUM_STATS), 100.0) 
        self.__meas_zones = None

        # Update per second
        self.__time = -2

        # Select front zone
        self.__front_angle = abs(front_angle)
        if self.__front_angle > 360:
            self.__front_angle = 360
        
        # Divide front zone
        angle1 = get_angle_range(-self.__front_angle / 2 - yaw)
        angle2 = get_angle_range(self.__front_angle / 2 - yaw)
        angle1_add = get_angle_range(angle1 + self.__front_angle / 3)
        angle2_sub = get_angle_range(angle2 - self.__front_angle / 3)        
        self.__angles = [angle1, angle1_add, angle2_sub, angle2]

        if init != None:
            self.__image = self.__get_back_image()

    def __get_back_image(self):
        image = pygame.Surface(self.__size_screen)
        mult = 10 * self.__scale
        text = ['FL', 'FF', 'FR']
        text_zone = ['Front-Left', 'Front-Front', 'Front-Right']

        for i in range(len(self.__angles)):
            x_line = self.__center_screen[0] + mult * math.cos(math.radians(self.__angles[i]))
            y_line = self.__center_screen[1] + mult * math.sin(math.radians(self.__angles[i]))
            pygame.draw.line(image, (70, 70, 70), self.__center_screen, (x_line, y_line), 2)

            if i < NUM_ZONES:
                angle = self.__angles[i] + self.__front_angle / 6
                x_zone = self.__center_screen[0] + mult * math.cos(math.radians(angle))
                y_zone = max(self.__center_screen[1] + mult * math.sin(math.radians(angle)), 
                             self.__size_text * (NUM_STATS + 2)) # Make sure to write behind the text
                
                write_text(text=text[i], img=image, point=(x_zone, y_zone), bold=True, size=self.__size_text)

                if self.show_stats:
                    write_text(text=text_zone[i], img=image, side=i, color=(255, 165, 180), bold=True,
                               point=(self.__x_text[i], self.__size_text), size=self.__size_text)
        return image
    
    def __interpolate_thickness(self, num:float):
        min = -2.3
        max = 1.8
        norm = (num - min) / (max - min)
        if norm < 0:
            norm = 0

        return self.__min_thickness + (self.__max_thickness - self.__min_thickness) * norm

    def __interpolate_color(self, num:float):
        min = 0.96
        max = 1
        norm = (num - min) / (max - min)
        if norm < 0:
            norm = 0
        
        r = int(self.__color_min[0] + (self.__color_max[0] - self.__color_min[0]) * norm)
        g = int(self.__color_min[1] + (self.__color_max[1] - self.__color_min[1]) * norm)
        b = int(self.__color_min[2] + (self.__color_max[2] - self.__color_min[2]) * norm)

        return (r, g, b)
    
    def __update_stats(self):
        for zone in range(NUM_ZONES):
            if len(self.__meas_zones[DIST][zone]) != 0:
                # Filter distances by z
                filter = np.array(self.__meas_zones[Z][zone]) > self.__z_threshold
                filtered_dist = np.array(self.__meas_zones[DIST][zone])[filter]

                if len(filtered_dist) == 0:
                    self.__stat_zones[zone][MIN] = np.nan
                else:
                    self.__stat_zones[zone][MIN] = np.min(filtered_dist)

                self.__stat_zones[zone][MEAN] = np.mean(self.__meas_zones[DIST][zone])
                self.__stat_zones[zone][MEDIAN] = np.median(self.__meas_zones[DIST][zone])
                self.__stat_zones[zone][STD] = np.std(self.__meas_zones[DIST][zone])
            else:
                for i in range(NUM_STATS):
                    self.__stat_zones[zone][i] = np.nan

            if self.show_stats and self.__rect != None:
                if time.time_ns() - self.__time > SEG_TO_NANOSEG:
                    self.__time = time.time_ns()
                    self.__stats_text = [
                        "Mean = {:.2f}".format(self.__stat_zones[zone][MEAN]),
                        "Median = {:.2f}".format(self.__stat_zones[zone][MEDIAN]),
                        "Std = {:.2f}".format(self.__stat_zones[zone][STD]),
                        "Min(z>{:.1f}) = {:.2f}".format(self.__z_threshold, self.__stat_zones[zone][MIN])
                    ]

                # Write stats
                y = self.__size_text * 2
                for text in self.__stats_text:
                    write_text(text=text, point=(self.__x_text[zone], y), img=self.__sub_screen, 
                               side=zone, size=self.__size_text)
                    y += self.__size_text

    def __in_zone(self, zone:int, angle:float):
        if self.__angles[zone] <= self.__angles[zone + 1]:
            return self.__angles[zone] <= angle <= self.__angles[zone + 1]
        else:
            return self.__angles[zone] <= angle or angle <= self.__angles[zone + 1]

    def __get_zone(self, x:float, y:float):
        angle = np.arctan2(y, x) * 180 / np.pi 

        for zone in range(NUM_ZONES):
            if self.__in_zone(zone, angle):
                return zone

        return NUM_ZONES

    def process_data(self):
        lidar = self.get_last_data()
        if lidar == None:
            return 
        
        lidar_data = np.copy(np.frombuffer(lidar.raw_data, dtype=np.dtype('f4')))
        lidar_data = np.reshape(lidar_data, (int(lidar_data.shape[0] / 4), 4))

        dist_zones = [[] for _ in range(NUM_ZONES)]
        z_zones = [[] for _ in range(NUM_ZONES)]

        if self.__rect != None:
            self.__sub_screen.blit(self.__image, (0, 0))

        for x, y, z, i in lidar_data:
            zone = self.__get_zone(x=x, y=y)
            if zone < NUM_ZONES and i < self.__i_threshold:
                dist_zones[zone].append(math.sqrt(x ** 2 + y ** 2))
                z_zones[zone].append(z)

            if self.__rect != None:
                thickness = self.__interpolate_thickness(num=z)
                color = self.__interpolate_color(num=i)

                center = (int(x * self.__scale + self.__center_screen[0]),
                          int(y * self.__scale + self.__center_screen[1]))
                pygame.draw.circle(self.__sub_screen, color, center, thickness)

        self.__meas_zones = [dist_zones, z_zones]
        self.__update_stats()  

        lock_sensor.acquire()
        self.__screen.blit(self.__sub_screen, self.__rect)
        lock_sensor.release()
    
    def set_i_threshold(self, i:float):
        self.__i_threshold = i

    def get_i_threshold(self):
        return self.__i_threshold
    
    def set_z_threshold(self, z:float):
        self.__z_threshold = z

    def get_z_threshold(self):
        return self.__z_threshold
    
    def get_stat_zones(self):
        return self.__stat_zones
    
    def get_meas_zones(self):
        return self.__meas_zones

class Vehicle_sensors:
    def __init__(self, vehicle:carla.Vehicle, world:carla.World, screen:pygame.Surface):
        self.__vehicle = vehicle
        self.__world = world
        self.__screen = screen
        self.sensors = []

        self.__time_frame = -1.0
        self.__count_frame = 0
        self.__write_frame = 0

    def __put_sensor(self, sensor_type:str, transform:carla.Transform, type:int=0):
        try:
            sensor_bp = self.__world.get_blueprint_library().find(sensor_type)
        except IndexError:
            print("Sensor", sensor_type, "doesn't exist!")
            return None
        
        if type == LIDAR:
            sensor_bp.set_attribute('rotation_frequency', '20')
        elif type == CAMERA:
            sensor_bp.set_attribute('image_size_x', str(SIZE_CAMERA))
            sensor_bp.set_attribute('image_size_y', str(SIZE_CAMERA))
                
        return self.__world.spawn_actor(sensor_bp, transform, attach_to=self.__vehicle)

    def add_sensor(self, sensor_type:str, transform:carla.Transform=carla.Transform()):
        sensor = self.__put_sensor(sensor_type=sensor_type, transform=transform)
        sensor_class = Sensor(sensor=sensor)
        self.sensors.append(sensor_class)
        return sensor_class
    
    def add_camera_rgb(self, size_rect:tuple[int, int]=None, init:tuple[int, int]=None, seg:bool=False,
                       transform:carla.Transform=carla.Transform(), init_extra:tuple[int, int]=None,
                       text:str=None, lane:bool=False):
        sensor = self.__put_sensor(sensor_type='sensor.camera.rgb', transform=transform, type=CAMERA)
        camera = CameraRGB(size=size_rect, init=init, sensor=sensor, screen=self.__screen, 
                           seg=seg, init_extra=init_extra, text=text, lane=lane)
        self.sensors.append(camera)
        return camera
    
    def add_lidar(self, size_rect:tuple[int, int]=None, init:tuple[int, int]=None, scale:int=25,
                  transform:carla.Transform=carla.Transform(), front_angle:int=150, show_stats:bool=True):
        sensor = self.__put_sensor(sensor_type='sensor.lidar.ray_cast', transform=transform, type=LIDAR)
        lidar = Lidar(size=size_rect, init=init, sensor=sensor, front_angle=front_angle, scale=scale,
                      yaw=transform.rotation.yaw, screen=self.__screen, show_stats=show_stats)
        
        self.sensors.append(lidar)
        return lidar

    def update_data(self, flip:bool=True):
        threads = []
        for i, sensor in enumerate(self.sensors):
            threads.append(threading.Thread(target=sensor.process_data()))
            threads[i].start()

        if time.time_ns() - self.__time_frame > SEG_TO_NANOSEG: 
            self.__write_frame = self.__count_frame
            self.__count_frame = 0
            self.__time_frame = time.time_ns()

        for t in threads:
            t.join()

        self.__count_frame += 1
        write_text(text="FPS: "+str(self.__write_frame), img=self.__screen, color=(0, 0, 0),
                   bold=True, point=(2, 0), size=23, side=LEFT)

        if flip:
            pygame.display.flip()

    def destroy(self):
        for sensor in self.sensors:
            sensor.sensor.destroy()

        self.__vehicle.destroy()

class Teleoperator:
    def __init__(self, vehicle:carla.Vehicle, steer:float=0.3, throttle:float=0.6, brake:float=1.0):
        self.__vehicle = vehicle
        self.__steer = max(0.0, min(1.0, steer))
        self.__throttle = max(0.0, min(1.0, throttle))
        self.__brake = max(0.0, min(1.0, brake))
        
    def control(self):
        control = carla.VehicleControl()
        keys = pygame.key.get_pressed()

        if keys[pygame.K_a]:
            control.steer = -self.__steer
        if keys[pygame.K_d]:
            control.steer =  self.__steer
        if keys[pygame.K_w]:
            control.throttle = self.__throttle
        if keys[pygame.K_s]:
            control.brake = self.__brake

        self.__vehicle.apply_control(control)

    def set_steer(self, steer:float):
        self.__steer = max(0.0, min(1.0, steer))

    def set_throttle(self, throttle:float):
        self.__throttle = max(0.0, min(1.0, throttle))

    def set_brake(self, brake:float):
        self.__brake = max(0.0, min(1.0, brake))

    def get_steer(self):
        return self.__steer
    
    def get_throttle(self):
        return self.__throttle
    
    def get_brake(self):
        return self.__brake

class PID:
    def __init__(self, vehicle:carla.Vehicle):
        self.__throttle = 0.4
        self.__vehicle = vehicle

        self.__kp = 1 / (SIZE_CAMERA / 2)
        self.__kd = 0
        self.__ki = 0

        self.__error = 0
        self.__prev_error = 0
        self.__total_error = 0

    def controll_vehicle(self, error:float):
        self.__prev_error = self.__error
        self.__error = error
        self.__total_error += error

        control = carla.VehicleControl()
        control.throttle = self.__throttle

        if error == np.nan:
            w = 0
        else:
            w = self.__kp * self.__error + self.__kd * self.__prev_error + self.__ki * self.__total_error
        print(w) 
        control.steer = w
        self.__vehicle.apply_control(control)

def setup_carla(port:int=2000, name_world:str='Town01', delta_seconds=0.05):
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