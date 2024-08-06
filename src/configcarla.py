import carla
import pygame
import numpy as np
import math
import random
import sys
import os
from PIL import Image
import time
import torch
import cv2
from abc import ABC, abstractmethod

PATH = '/home/lpoves/'

sys.path.insert(0, PATH + 'efficientvit-urjc/urjc')
sys.path.insert(0, PATH + '/efficientvit-urjc')
os.chdir(PATH + '/efficientvit-urjc/urjc')

import EfficientVit as EV

SEG_TO_NANOSEG = 1_000_000_000
SIZE_CAMERA = 512
SIZE_MEM = 5

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

# Lane
LEFT_LANE = 0
RIGHT_LANE = 1

# Segmentation
ROAD = 0

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
    text_rect = text.get_rect()

    if side == LEFT:
        text_rect.topleft = point 
    elif side == RIGHT:
        text_rect.topright = point
    else:
        text_rect.center = point

    if background != None:
        pygame.draw.rect(img, background, text_rect)

    img.blit(text, text_rect)

class Sensor(ABC):
    def __init__(self, sensor):
        self.sensor = sensor
        self.sensor.listen(lambda data: self._callback_data(data))
        self.data = None

    def _callback_data(self, data):
        self.data = data

    def get_last_data(self):
        return self.data

    @abstractmethod
    def process_data(self):
        pass

class CameraRGB(Sensor):      
    def __init__(self, size:tuple[int, int], init:tuple[int, int], sensor:carla.Sensor,
                 text:str, screen:pygame.Surface, seg:bool, init_extra:tuple[int, int], 
                 lane:bool, canvas_seg:bool):
        super().__init__(sensor=sensor)

        self._screen = screen
        self.text = text
        self._size_text = 20
        self._deviation = 0
        self._road_percentage = 0
        self._error_lane = False
        self._angle_error = 0

        self._seg = seg
        self._canvas_seg = canvas_seg
        if seg:
            self._seg_model = EV.EfficientVit()
        else:
            self._canvas_seg = False

        self._lane = lane
        if lane:
            file = PATH + '2024-tfg-lara-poves/best_model_torch.pth'
            self._lane_model = torch.load(file)
            self._lane_model.eval()

            # Constants
            self._threshold_road_per = 90.0
            self._threshold_lane_mask = 0.05
            self._ymin_lane = 275 
            self._angle_lane = 7
            self._mem_max = 4

            # Initialize
            self._coefficients = np.zeros((SIZE_MEM, 2, 3), dtype=float)
            self._count_coef = [0, 0] # Not use until having 5 measurements
            self._count_mem_road = 0
            self._count_mem_lane = [0, 0]

        self._cm = np.zeros((2,), dtype=np.int32)
        self._area = np.int32(0)
        self._surface_seg = None

        self._rect_org = init
        self._rect_extra = init_extra

        if init_extra != None or init != None:
            assert size != None, "size is required!"
            sub_screen = pygame.Surface(size)

            if init != None:
                self._rect_org = sub_screen.get_rect(topleft=init)

            if init_extra != None:
                self._rect_extra = sub_screen.get_rect(topleft=init_extra)

    def _mask_lane(self, mask:list, index:int, canvas):
        if index == LEFT_LANE:
            lane = "left"
        else:
            lane = "right"

        if self._count_mem_lane[index] >= self._mem_max:
            self._error_lane = True
            assert False, "Line " + lane + " not found"

        mask = mask[:, int(SIZE_CAMERA / 2):int(SIZE_CAMERA + SIZE_CAMERA / 2)]
        index_mask = np.where(mask > self._threshold_lane_mask)
        if len(index_mask[0]) < 10:
            self._count_mem_lane[index] += 1
            return False
        else:
            self._count_mem_lane[index] = 0

        coefficients = np.polyfit(index_mask[0], index_mask[1], 1)

        # Previus linear regression
        if self._count_coef[index] >= SIZE_MEM:
            coefficients = self._coefficients[-1, index, 0:2]
        else:
            coefficients = np.polyfit(index_mask[0], index_mask[1], 1)

        # Remove outliers
        th = 60
        x_coef = []
        y_coef = []
        for y, x in zip(index_mask[0], index_mask[1]):
            x_1 = coefficients[0] * y + coefficients[1] + th
            x_2 = coefficients[0] * y + coefficients[1] - th

            if x_1 <= x <= x_2 or x_2 <= x <= x_1:
                x_coef.append(x)
                y_coef.append(y)

        # Memory lane
        if len(x_coef) < 10:
            self._count_mem_lane[index] += 1
            return False
        else:
            self._count_mem_lane[index] = 0

        # Linear regression
        coefficients = np.polyfit(y_coef, x_coef, 1)
     
        # Check measure
        mean = np.mean(self._coefficients[:, index, 2])
        angle = math.degrees(math.atan(coefficients[0])) % 180
        diff = abs(mean - angle)
        
        if diff > self._angle_lane and self._count_coef[index] >= SIZE_MEM:
            self._count_mem_lane[index] += 1
            # It might get both bad measures at the same time, but that doesn't mean it has lost the lane
            return  diff < 20 # Only return False for really bad measures
        elif self._count_coef[index] >= SIZE_MEM:
            self._count_mem_lane[index] = 0

        # Update memory
        for i in range(len(self._coefficients) - 1):
            self._coefficients[i, index, :] = self._coefficients[i + 1, index, :]
        self._coefficients[-1, index, 0:2] = coefficients
        self._coefficients[-1, index, 2] = angle

        # Update count 
        self._count_coef[index] += 1

        return True

    def _detect_lane(self, data:list, canvas:list, mask:list): 
        # Resize for the network, copy the image in the middle
        image_lane = np.zeros((SIZE_CAMERA, SIZE_CAMERA * 2, 3), dtype=np.uint8)
        image_lane[:SIZE_CAMERA, int(SIZE_CAMERA / 2):int(SIZE_CAMERA + SIZE_CAMERA / 2), :] = data

        with torch.no_grad():
            image_tensor = image_lane.transpose(2,0,1).astype('float32')/255
            x_tensor = torch.from_numpy(image_tensor).to("cuda").unsqueeze(0)
            model_output = torch.softmax(self._lane_model.forward(x_tensor), dim=1 ).cpu().numpy()

        _, left_mask, right_mask = model_output[0]
        see_line_left = self._mask_lane(mask=left_mask, index=LEFT_LANE, canvas=canvas)
        see_line_right = self._mask_lane(mask=right_mask, index=RIGHT_LANE, canvas=canvas)

        # Coefficients
        coef_left = self._coefficients[-1, LEFT_LANE, 0:2]
        coef_right = self._coefficients[-1, RIGHT_LANE, 0:2]

        if see_line_left == False and see_line_right == False:
            self._error_lane = True
            assert False, "Lane not found"

        count_x = count_y = 0
        count_total = count_road = 0

        for y in range(self._ymin_lane, SIZE_CAMERA):
            x_left = max(int(y * coef_left[0] + coef_left[1]), 0)
            x_right = min(int(y * coef_right[0] + coef_right[1]) + 1, SIZE_CAMERA - 1)

            if x_left < x_right:
                # Center of mass
                count_x += sum(range(x_left, x_right))
                count_y += y * (x_right - x_left)

                # Draw lane
                canvas[y, x_left:x_right] = [255, 240, 255]

                # Road percentage
                count_total += x_right - x_left
                if self._seg:
                    region_mask = mask[y, x_left:x_right]
                    count_road += np.count_nonzero(region_mask == ROAD)

        if count_total > 0:
            # Calculate center of mass
            x_cm = int(count_x / count_total)
            y_cm = int(count_y / count_total) 
            middle = int(SIZE_CAMERA / 2)
            self._deviation = x_cm - SIZE_CAMERA / 2 
            self._cm = np.array([x_cm, y_cm], dtype=np.int32)
            self._area = count_total

            # Calculate road porcentage
            if self._seg:
                self._road_percentage = count_road / count_total * 100
                if self._road_percentage < self._threshold_road_per:
                    self._count_mem_road += 1

                    if self._count_mem_road > self._mem_max:
                        self._error_lane = True
                        assert False, "Low percentage of lane"
                else:
                    self._count_mem_road = 0

            # Draw center of mass and vehicle
            cv2.line(canvas, (x_cm, 0), (x_cm, SIZE_CAMERA), (0, 255, 0), 2)
            cv2.line(canvas, (middle, 0), (middle, SIZE_CAMERA), (255, 0, 0), 2)
            cv2.circle(canvas, (x_cm, y_cm), 9, (0, 255, 0), -1)
        else:
            self._deviation = SIZE_CAMERA / 2
            self._road_percentage = 0
            self._error_lane = True
            assert False, "Area zero"

    def _process_seg(self, data:list):
        image_data = cv2.rotate(data, cv2.ROTATE_90_CLOCKWISE)
        image_seg = Image.fromarray(image_data)

        # Prediction
        if self._seg:
            pred = self._seg_model.predict(image_seg)

        if self._canvas_seg and self._seg:
            canvas, mask = self._seg_model.get_canvas(image_data, pred)
        else:
            canvas = image_data
            if self._seg:
                mask = pred

                if (SIZE_CAMERA, SIZE_CAMERA) != mask.shape:
                    mask = cv2.resize(mask, dsize=(SIZE_CAMERA, SIZE_CAMERA), interpolation=cv2.INTER_NEAREST)
            else:
                mask = None
            
        if self._lane:
            image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            self._detect_lane(data=image_data, canvas=canvas, mask=mask)

        if self._rect_extra != None:
            # Convert to pygame syrface
            canvas = Image.fromarray(canvas)
            surface_seg = pygame.image.fromstring(canvas.tobytes(), canvas.size, canvas.mode)
            surface_seg = pygame.transform.scale(surface_seg, self._rect_extra.size)

            # Write text
            text = ""
            if self._canvas_seg and self._seg:
                text = "Segmented "

            write_text(text=text+self.text, img=surface_seg, color=(0, 0, 0), side=RIGHT,
                       bold=True, size=self._size_text, point=(self._rect_extra.size[0], 0))
            
            if self._lane:
                write_text(text="Deviation = "+str(int(abs(self._deviation)))+" (pixels)",
                           img=surface_seg, color=(0, 0, 0), side=LEFT, size=self._size_text, 
                           point=(0, self._rect_extra.size[0] - self._size_text), bold=True)
                if self._seg:
                    write_text(text=f"{self._road_percentage:.2f}% road", side=RIGHT, bold=True,
                               img=surface_seg, color=(0, 0, 0), size=self._size_text, 
                               point=(SIZE_CAMERA, self._rect_extra.size[0] - self._size_text))
                
            self._surface_seg = surface_seg
            self._screen.blit(surface_seg, self._rect_extra)

    def process_data(self):
        image = self.data
        self._error_lane = False
        if image == None:
            return 

        image_data = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        image_data = np.reshape(image_data, (image.height, image.width, 4))

        # Swap blue and red channels
        image_data = image_data[:, :, (2, 1, 0)]

        if self._seg or self._lane:
            self._process_seg(image_data)

        if self._rect_org != None:
            # Reserve mirror effect
            image_surface = pygame.surfarray.make_surface(image_data)
            flipped_surface = pygame.transform.flip(image_surface, True, False)
            screen_surface = pygame.transform.scale(flipped_surface, self._rect_org.size)

            if self.text != None:
                write_text(text=self.text, img=screen_surface, color=(0, 0, 0), side=RIGHT, bold=True,
                           size=self._size_text, point=(self._rect_org.size[0], 0))
                
            self._screen.blit(screen_surface, self._rect_org)

    def get_deviation(self):
        return self._deviation
    
    def get_road_percentage(self):
        return self._road_percentage
    
    def get_lane_cm(self):
        if self._error_lane:
            # Move cm to the nearest corner
            if self._cm[0] < SIZE_CAMERA / 2:
                x_cm = 0
            else:
                x_cm = SIZE_CAMERA - 1
            if self._cm[1] < SIZE_CAMERA / 2:
                y_cm = 0
            else:
                y_cm = SIZE_CAMERA - 1

            self._cm = np.array([x_cm, y_cm], dtype=np.int32)

        return self._cm
    
    def get_lane_area(self):
        if self._error_lane:
            self._area = 0

        return self._area
    
    def get_lane_points(self, num_points:int=5, show:bool=False):
        if not self._lane or self._error_lane:
            return [np.full((num_points, 2), SIZE_CAMERA / 2, dtype=np.int32)] * 2
        
        lane_points = []
        for side in range(2):
            y_points = np.linspace(self._ymin_lane, SIZE_CAMERA - 1, num_points)
            points = np.zeros((num_points, 2), dtype=np.int32)

            for i, y in enumerate(y_points):
                coef = self._coefficients[-1, side, 0:2]
                points[i, 0] = int(y * coef[0] + coef[1]) # x
                points[i, 1] = y

                if points[i, 0] < 0:
                    points[i, 0] = 0
                elif points[i, 0] > SIZE_CAMERA - 1:
                    points[i, 0] = SIZE_CAMERA - 1

                if show and self._rect_extra != None and self._surface_seg != None:
                    if side == LEFT_LANE:
                        color = (0, 0, 200)
                    else:
                        color = (200, 0, 0)

                    pygame.draw.circle(self._surface_seg, color, points[i], 6, 0)
                    self._screen.blit(self._surface_seg, self._rect_extra)

            lane_points.append(points)

        return lane_points
        
class Lidar(Sensor): 
    def __init__(self, size:tuple[int, int], init:tuple[int, int], sensor:carla.Sensor, scale:int,
                 front_angle:int, yaw:float, screen:pygame.Surface, show_stats:bool=True):
        super().__init__(sensor=sensor)

        self._rect = init
        self.show_stats = show_stats

        if init != None:
            assert size != None, "size is required!"
            self._sub_screen = pygame.Surface(size)
            self._rect = self._sub_screen.get_rect(topleft=init)

            self._scale = scale
            self._size_screen = size
            self._screen = screen

            # Visualize lidar
            self._min_thickness = 2
            self._max_thickness = math.ceil(scale / 10) + self._min_thickness * 3
            self._color_min = (0, 0, 255)
            self._color_max = (255, 0, 0)

            # Write stats
            y = size[1] / 2
            if show_stats:
                y += scale * 1.5
            self._center_screen = (int(size[0] / 2), int(y))

            # Write text
            self._size_text = min(int(scale / 1.5), 20)
            self._x_text = (self._max_thickness, self._center_screen[0], size[0] - self._max_thickness)

        # Calculate stats
        self._i_threshold = 0.987
        self._z_threshold = -1.6
        self._stat_zones = np.full((NUM_ZONES, NUM_STATS), 100.0) 

        # Update per second
        self._time = -2

        # Select front zone
        self._front_angle = abs(front_angle)
        if self._front_angle > 360:
            self._front_angle = 360
        
        # Divide front zone
        angle1 = get_angle_range(-self._front_angle / 2 - yaw)
        angle2 = get_angle_range(self._front_angle / 2 - yaw)
        angle1_add = get_angle_range(angle1 + self._front_angle / 3)
        angle2_sub = get_angle_range(angle2 - self._front_angle / 3)        
        self._angles = [angle1, angle1_add, angle2_sub, angle2]

        if init != None:
            self._image = self._get_back_image()

    def _get_back_image(self):
        image = pygame.Surface(self._size_screen)
        mult = 10 * self._scale
        text = ['FL', 'FF', 'FR']
        text_zone = ['Front-Left', 'Front-Front', 'Front-Right']

        for i in range(len(self._angles)):
            x_line = self._center_screen[0] + mult * math.cos(math.radians(self._angles[i]))
            y_line = self._center_screen[1] + mult * math.sin(math.radians(self._angles[i]))
            pygame.draw.line(image, (70, 70, 70), self._center_screen, (x_line, y_line), 2)

            if i < NUM_ZONES:
                angle = self._angles[i] + self._front_angle / 6
                x_zone = self._center_screen[0] + mult * math.cos(math.radians(angle))
                y_zone = max(self._center_screen[1] + mult * math.sin(math.radians(angle)), 
                             self._size_text * (NUM_STATS + 2)) # Make sure to write behind the text
                
                write_text(text=text[i], img=image, point=(x_zone, y_zone), bold=True, size=self._size_text)

                if self.show_stats:
                    write_text(text=text_zone[i], img=image, side=i, color=(255, 165, 180), bold=True,
                               point=(self._x_text[i], self._size_text), size=self._size_text)
        return image
    
    def _interpolate_thickness(self, num:float):
        min = -2.3
        max = 1.8
        norm = (num - min) / (max - min)
        if norm < 0:
            norm = 0

        return self._min_thickness + (self._max_thickness - self._min_thickness) * norm

    def _interpolate_color(self, num:float):
        min = 0.96
        max = 1
        norm = (num - min) / (max - min)
        if norm < 0:
            norm = 0
        
        r = int(self._color_min[0] + (self._color_max[0] - self._color_min[0]) * norm)
        g = int(self._color_min[1] + (self._color_max[1] - self._color_min[1]) * norm)
        b = int(self._color_min[2] + (self._color_max[2] - self._color_min[2]) * norm)

        return (r, g, b)
    
    def _update_stats(self, meas_zones):
        for zone in range(NUM_ZONES):
            if len(meas_zones[DIST][zone]) != 0:
                # Filter distances by z
                filter = np.array(meas_zones[Z][zone]) > self._z_threshold
                filtered_dist = np.array(meas_zones[DIST][zone])[filter]

                if len(filtered_dist) == 0:
                    self._stat_zones[zone][MIN] = np.nan
                else:
                    self._stat_zones[zone][MIN] = np.min(filtered_dist)

                self._stat_zones[zone][MEAN] = np.mean(meas_zones[DIST][zone])
                self._stat_zones[zone][MEDIAN] = np.median(meas_zones[DIST][zone])
                self._stat_zones[zone][STD] = np.std(meas_zones[DIST][zone])
            else:
                for i in range(NUM_STATS):
                    self._stat_zones[zone][i] = np.nan

            if self.show_stats and self._rect != None:
                if time.time_ns() - self._time > SEG_TO_NANOSEG:
                    self._time = time.time_ns()
                    self._stats_text = [
                        "Mean = {:.2f}".format(self._stat_zones[zone][MEAN]),
                        "Median = {:.2f}".format(self._stat_zones[zone][MEDIAN]),
                        "Std = {:.2f}".format(self._stat_zones[zone][STD]),
                        "Min(z>{:.1f}) = {:.2f}".format(self._z_threshold, self._stat_zones[zone][MIN])
                    ]

                # Write stats
                y = self._size_text * 2
                for text in self._stats_text:
                    write_text(text=text, point=(self._x_text[zone], y), img=self._sub_screen, 
                               side=zone, size=self._size_text)
                    y += self._size_text

    def _in_zone(self, zone:int, angle:float):
        if self._angles[zone] <= self._angles[zone + 1]:
            return self._angles[zone] <= angle <= self._angles[zone + 1]
        else:
            return self._angles[zone] <= angle or angle <= self._angles[zone + 1]

    def _get_zone(self, x:float, y:float):
        angle = np.arctan2(y, x) * 180 / np.pi 

        for zone in range(NUM_ZONES):
            if self._in_zone(zone, angle):
                return zone

        return NUM_ZONES

    def process_data(self):
        lidar = self.data
        if lidar == None:
            return 
        
        lidar_data = np.copy(np.frombuffer(lidar.raw_data, dtype=np.dtype('f4')))
        lidar_data = np.reshape(lidar_data, (int(lidar_data.shape[0] / 4), 4))

        dist_zones = [[] for _ in range(NUM_ZONES)]
        z_zones = [[] for _ in range(NUM_ZONES)]

        if self._rect != None:
            self._sub_screen.blit(self._image, (0, 0))

        for x, y, z, i in lidar_data:
            zone = self._get_zone(x=x, y=y)
            if zone < NUM_ZONES and i < self._i_threshold:
                dist_zones[zone].append(math.sqrt(x ** 2 + y ** 2))
                z_zones[zone].append(z)

            if self._rect != None:
                thickness = self._interpolate_thickness(num=z)
                color = self._interpolate_color(num=i)

                center = (int(x * self._scale + self._center_screen[0]),
                          int(y * self._scale + self._center_screen[1]))
                pygame.draw.circle(self._sub_screen, color, center, thickness)

        self._update_stats(meas_zones=[dist_zones, z_zones])  

        if self._rect != None:
            self._screen.blit(self._sub_screen, self._rect)
    
    def set_i_threshold(self, i:float):
        self._i_threshold = i

    def get_i_threshold(self):
        return self._i_threshold
    
    def set_z_threshold(self, z:float):
        self._z_threshold = z

    def get_z_threshold(self):
        return self._z_threshold
    
    def get_stat_zones(self):
        return self._stat_zones

class Collision(Sensor):
    def __init__(self, sensor):
        super().__init__(sensor=sensor)

    def process_data(self):
        if self.data != None:
            other_actor = self.data.other_actor
            assert False, f"The vehicle crashed with {other_actor.type_id}"

class Vehicle_sensors:
    def __init__(self, vehicle:carla.Vehicle, world:carla.World, screen:pygame.Surface=None, 
                 color_text:tuple[int, int, int]=(0, 0, 0)):
        self._vehicle = vehicle
        self._world = world
        self._screen = screen
        self.sensors = []

        self.color_text = color_text
        self._time_frame = -1.0
        self._count_frame = 0
        self._write_frame = 0

    def _put_sensor(self, sensor_type:str, transform:carla.Transform=carla.Transform(), type:int=0):
        try:
            sensor_bp = self._world.get_blueprint_library().find(sensor_type)
        except IndexError:
            print("Sensor", sensor_type, "doesn't exist!")
            return None
        
        if type == LIDAR:
            sensor_bp.set_attribute('rotation_frequency', '20')
        elif type == CAMERA:
            sensor_bp.set_attribute('image_size_x', str(SIZE_CAMERA))
            sensor_bp.set_attribute('image_size_y', str(SIZE_CAMERA))
                
        return self._world.spawn_actor(sensor_bp, transform, attach_to=self._vehicle)
    
    def add_camera_rgb(self, size_rect:tuple[int, int]=None, init:tuple[int, int]=None, seg:bool=False,
                       transform:carla.Transform=carla.Transform(), init_extra:tuple[int, int]=None,
                       text:str='', lane:bool=False, canvas_seg:bool=True):
        if self._screen == None:
            init = None
            init_extra = None

        sensor = self._put_sensor(sensor_type='sensor.camera.rgb', transform=transform, type=CAMERA)
        camera = CameraRGB(size=size_rect, init=init, sensor=sensor, screen=self._screen, seg=seg,
                           init_extra=init_extra, text=text, lane=lane, canvas_seg=canvas_seg)
        self.sensors.append(camera)
        return camera
    
    def add_lidar(self, size_rect:tuple[int, int]=None, init:tuple[int, int]=None, scale:int=25,
                  transform:carla.Transform=carla.Transform(), front_angle:int=150, show_stats:bool=True):
        if self._screen == None:
            init = None

        sensor = self._put_sensor(sensor_type='sensor.lidar.ray_cast', transform=transform, type=LIDAR)
        lidar = Lidar(size=size_rect, init=init, sensor=sensor, front_angle=front_angle, scale=scale,
                      yaw=transform.rotation.yaw, screen=self._screen, show_stats=show_stats)
        
        self.sensors.append(lidar)
        return lidar
    
    def add_collision(self):
        sensor = self._put_sensor(sensor_type='sensor.other.collision')
        sensor_collision = Collision(sensor=sensor)
        self.sensors.append(sensor_collision)
        return sensor_collision

    def update_data(self, flip:bool=True):
        for sensor in self.sensors:
            sensor.process_data()

        if self._screen != None:
            if time.time_ns() - self._time_frame > SEG_TO_NANOSEG: 
                self._write_frame = self._count_frame
                self._count_frame = 0
                self._time_frame = time.time_ns()

            self._count_frame += 1
            write_text(text="FPS: "+str(self._write_frame), img=self._screen, color=self.color_text,
                       bold=True, point=(2, 0), size=23, side=LEFT)
            
            v = self._vehicle.get_velocity()
            v = carla.Vector3D(v).length()
            write_text(text=f"Vel: {v:.2f} m/s", img=self._screen, color=self.color_text,
                       bold=True, point=(2, 20), size=20, side=LEFT)

            if flip:
                pygame.display.flip()

    def destroy(self):
        for sensor in self.sensors:
            sensor.sensor.destroy()

        self._vehicle.destroy()

class Teleoperator:
    def __init__(self, vehicle:carla.Vehicle, steer:float=0.3, throttle:float=0.6, brake:float=1.0):
        self._vehicle = vehicle
        self._steer = max(0.0, min(1.0, steer))
        self._throttle = max(0.0, min(1.0, throttle))
        self._brake = max(0.0, min(1.0, brake))
        
    def control(self):
        control = carla.VehicleControl()
        keys = pygame.key.get_pressed()

        if keys[pygame.K_a]:
            control.steer = -self._steer
        if keys[pygame.K_d]:
            control.steer = self._steer
        if keys[pygame.K_w]:
            control.throttle = self._throttle
        if keys[pygame.K_s]:
            control.brake = self._brake

        self._vehicle.apply_control(control)

    def set_steer(self, steer:float):
        self._steer = max(0.0, min(1.0, steer))

    def set_throttle(self, throttle:float):
        self._throttle = max(0.0, min(1.0, throttle))

    def set_brake(self, brake:float):
        self._brake = max(0.0, min(1.0, brake))

    def get_steer(self):
        return self._steer
    
    def get_throttle(self):
        return self._throttle
    
    def get_brake(self):
        return self._brake

class PID:
    def __init__(self, vehicle:carla.Vehicle):
        self._vehicle = vehicle
        self._kp = 1 / (SIZE_CAMERA / 2)
        self._kd = -self._kp / 1.7
        self._throttle = 0.6

        self._error = 0
        self._prev_error = 0

    def controll_vehicle(self, error:float):
        control = carla.VehicleControl()
        self._prev_error = self._error
        self._error = error

        # Different sign
        if (self._error > 0 and self._prev_error < 0) or (self._error < 0 and self._prev_error > 0):
            self._prev_error = 0        

        v = self._vehicle.get_velocity()
        v = carla.Vector3D(v).length()
        if v > 10:
            control.brake = self._throttle / 2
        elif v > 7:
            control.brake = self._throttle / 5
        control.throttle = self._throttle

        if error > 20:
           error *= 1.15

        control.steer = self._kp * error + self._kd * self._prev_error
        self._vehicle.apply_control(control)

def setup_carla(port:int=2000, name_world:str='Town01', fixed_delta_seconds:float=0.0, 
                client:carla.Client=None, syn:bool=False):
    if client == None:
        client = carla.Client('localhost', port)
    world = client.get_world()
    world = client.load_world(name_world)

    settings = world.get_settings()
    settings.fixed_delta_seconds = fixed_delta_seconds
    if syn:
        settings.synchronous_mode = True
    else:
        settings.synchronous_mode = False
    world.apply_settings(settings)
    client.reload_world(False) # Reload world keeping settings

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
    tm.set_synchronous_mode(True)
    tm_port = tm.get_port()

    for v in vehicles:
        v.set_autopilot(True, tm_port)
        tm.auto_lane_change(v, False) 

    return tm