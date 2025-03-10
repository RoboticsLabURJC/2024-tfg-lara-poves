import carla
import pygame
import numpy as np
import math
import random
import sys
import os
from PIL import Image
import time
import cv2
from abc import ABC, abstractmethod
import torch

PATH = '/home/lpoves/'

sys.path.insert(0, PATH + 'carla_lane_detector/ground_truth')
from camera_geometry import (
    get_intrinsic_matrix,
    project_polyline,
    check_inside_image,
    create_lane_lines,
    get_matrix_global,
    CameraGeometry,
)

sys.path.insert(0, PATH + 'efficientvit-urjc/urjc')
sys.path.insert(0, PATH + 'efficientvit-urjc')
os.chdir(PATH + '/efficientvit-urjc/urjc')

import EfficientVit as EV

SEG_TO_NANOSEG = 1_000_000_000
SIZE_CAMERA = 512
SIZE_MEM = 5
FOV = 110

MAX_NUM_LANES = 4

# Type
CAMERA = 1
LIDAR = 2

# Side
NUM_ZONES = 5
LEFT_FRONT = 0
FRONT = 1
RIGHT_FRONT = 2
RIGHT = 3
RIGHT_BACK = 4

# Stats
NUM_STATS = 4
MEAN = 0
MEDIAN = 1
STD = 2
MIN = 3

# Measurements
DIST = 0
Z = 1
X = 2
Y = 3

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

    if side == LEFT_FRONT:
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
        self.update_screen = True

    def _callback_data(self, data):
        self.data = data

    def get_last_data(self):
        return self.data

    @abstractmethod
    def process_data(self):
        pass

class CameraRGB(Sensor):      
    def __init__(self, size:tuple[int, int], init:tuple[int, int], sensor:carla.Sensor,
                 text:str, screen:pygame.Surface, seg:bool, init_extra:tuple[int, int], lane:bool,
                 canvas_seg:bool, transform:carla.Transform, vehicle:carla.Vehicle, world:carla.World,
                 lane_network:bool, check_area_lane:bool, check_other_lane:bool):
        super().__init__(sensor=sensor)

        self._screen = screen 
        self._transform = transform
        self._vehicle = vehicle 
        self._world = world  
        self.init = init
        self.init_extra = init_extra
        self.text = text
        self.size_text = 20
        self.size = size
        self._lane_network = lane_network
        self._lane = lane
        self._seg = seg
        self._mask = []
        self._check_area_lane = self._seg and self._lane and check_area_lane
        self._check_other_lane = self._seg and self._lane and check_other_lane
        self._road_left = 0
        self._road_right = 0

        # Lane detection
        self._deviation = 0
        self._road_percentage = 0
        self._cm = np.zeros((2,), dtype=np.int32)
        self._area = np.int32(0)
        self._lane_left = []
        self._lane_right = []
        self._extra_surface = None
        self._num_lanes = [False] * (1 + (MAX_NUM_LANES - 1) * 2)

        if self._lane_network:
            file = PATH + '2024-tfg-lara-poves/best_model_torch.pth'
            self._lane_model = torch.load(file)
            self._lane_model.eval()

            # Constants
            self._threshold_lane_mask = 0.05
            self._ymin_lane = 275 
            self._angle_lane = 7
            self._mem_max = 4

            # Initialize
            self._coefficients = np.zeros((SIZE_MEM, 2, 3), dtype=float)
            self._count_coef = [0, 0] # Not use until having 5 measurements
            self._count_mem_road = 0
            self._count_no_lane = 0
            self._count_mem_lane = [0, 0]

        self._canvas_seg = canvas_seg
        if seg:
            self._seg_model = EV.EfficientVit()
        else:
            self._canvas_seg = False

        if lane:
            self._trafo_matrix_vehicle_to_cam = np.array(transform.get_inverse_matrix())
            self._K = get_intrinsic_matrix(FOV, SIZE_CAMERA, SIZE_CAMERA)

        self._threshold_road_per = 90.0
        self._threshold_area_road = 5000

    def _points_lane(self, boundary:np.ndarray, trafo_matrix_global_to_camera:np.ndarray, side:int):
        projected_boundary = project_polyline(boundary, trafo_matrix_global_to_camera,self._K).astype(np.int32)

        if not check_inside_image(projected_boundary, SIZE_CAMERA, SIZE_CAMERA) or len(projected_boundary) <= 1:
            return []

        # Draw the line lane
        black_surface = pygame.Surface((SIZE_CAMERA, SIZE_CAMERA))
        black_surface.fill((0, 0, 0))
        rect = pygame.draw.lines(black_surface, (255, 0, 0), False, projected_boundary, 4)

        # Get pixels of line lane
        step = 1 if side == LEFT_LANE else -1
        start = rect.left if side == LEFT_LANE else rect.right - 1
        end = rect.right if side == LEFT_LANE else rect.left - 1

        pixels = []
        for y in range(rect.top, rect.bottom):
            for x in range(start, end, step):
                color = black_surface.get_at((x, y))
                if color[0] == 255:
                    pixels.append((x, y))
                    break

        if side == LEFT_LANE:
            x = 0
        else:
            x = SIZE_CAMERA - 1

        for y in range(rect.bottom, SIZE_CAMERA):
            pixels.append((x, y))

        return pixels
    
    def _mask_lane(self, mask:list, index:int):
        if index == LEFT_LANE:
            lane = "left"
        else:
            lane = "right"

        if self._count_mem_lane[index] >= self._mem_max:
            self._lane_left = [] # Mark error
            assert False, "Line " + lane + " not found"

        mask = mask[:, int(SIZE_CAMERA / 2):int(SIZE_CAMERA + SIZE_CAMERA / 2)]
        index_mask = np.where(mask > self._threshold_lane_mask)
        if len(index_mask[0]) < 10:
            self._count_mem_lane[index] += 1
            return False
        else:
            self._count_mem_lane[index] = 0

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
        if len(x_coef) < 20:
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

    def _detect_lane(self, img:np.ndarray, mask:np.ndarray):
        if not self._lane_network:
            #t = time.time_ns()
            trafo_matrix_global_to_camera = get_matrix_global(self._vehicle, self._trafo_matrix_vehicle_to_cam)
            waypoint = self._world.get_map().get_waypoint(
                self._vehicle.get_transform().location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,                
            )
            
            # Get points of the lane
            _, left_boundary, right_boundary, _ = create_lane_lines(waypoint, self._vehicle)
            #print("Ground Truth - Predicción:", time.time_ns() - t)
            #t = time.time_ns()

            self._lane_left = self._points_lane(left_boundary, trafo_matrix_global_to_camera, LEFT_LANE)
            self._lane_right = self._points_lane(right_boundary, trafo_matrix_global_to_camera, RIGHT_LANE)
            assert len(self._lane_left) > 20 and len(self._lane_right) > 20, "Lane lost"

            # Start in same height
            size_left = len(self._lane_left)
            size_right = len(self._lane_right)
            if size_right > size_left:
                del self._lane_right[:size_right-size_left]
            elif size_left > size_right:
                del self._lane_left[:size_left-size_right]

            limitis_for = (0, len(self._lane_left))
        else:
            # Mark no errors
            self._lane_left = [1]
            self._lane_right = [1]

            # Resize for the network, copy the image in the middle
            image_lane = np.zeros((SIZE_CAMERA, SIZE_CAMERA * 2, 3), dtype=np.uint8)
            image_lane[:SIZE_CAMERA, int(SIZE_CAMERA / 2):int(SIZE_CAMERA + SIZE_CAMERA / 2), :] = img

            #t = time.time_ns()
            with torch.no_grad():
                image_tensor = image_lane.transpose(2,0,1).astype('float32')/255
                x_tensor = torch.from_numpy(image_tensor).to("cuda").unsqueeze(0)
                model_output = torch.softmax(self._lane_model.forward(x_tensor), dim=1 ).cpu().numpy()
            #print("Carril DL - Predicción:", time.time_ns() - t)

            t = time.time_ns()
            _, left_mask, right_mask = model_output[0]
            see_line_left = self._mask_lane(mask=left_mask, index=LEFT_LANE)
            see_line_right = self._mask_lane(mask=right_mask, index=RIGHT_LANE)

            # Coefficients
            coef_left = self._coefficients[-1, LEFT_LANE, 0:2]
            coef_right = self._coefficients[-1, RIGHT_LANE, 0:2]

            if see_line_left == False and see_line_right == False:
                self._count_no_lane += 1

                if self._count_no_lane >= self._mem_max / 2:
                    self._lane_left = []
                    assert False, "Lane not found"
            else:
                self._count_no_lane += 0

            limitis_for = (self._ymin_lane, SIZE_CAMERA)

        # Draw the lane 
        count_x = count_y = 0
        count_total = count_road = 0
        count_road_left = count_road_right = count_total_left = count_total_right = 0
        for i in range(limitis_for[0], limitis_for[1]):
            if not self._lane_network:
                x_left, y = self._lane_left[i]
                x_right, _ = self._lane_right[i]
            else:
                y = i
                x_left = max(int(y * coef_left[0] + coef_left[1]), 0)
                x_right = min(int(y * coef_right[0] + coef_right[1]) + 1, SIZE_CAMERA - 1)

            if x_left < x_right:
                img[y, x_left:x_right] = (255, 240, 255)
                width_lane = x_right - x_left
                
                if self._check_other_lane:
                    width_lane_offset = width_lane + int(width_lane / 2)

                    bound_left = np.clip(x_left - width_lane_offset, 0, x_left)
                    region_mask = mask[y, bound_left:x_left]
                    count_road_left += np.count_nonzero(region_mask == ROAD)
                    count_total_left += x_left - bound_left

                    bound_right = np.clip(x_right + width_lane_offset, x_right, SIZE_CAMERA)
                    region_mask = mask[y, x_right:bound_right]
                    count_road_right += np.count_nonzero(region_mask == ROAD)
                    count_total_right += bound_right - x_right

                # Center of mass
                count_x += sum(range(x_left, x_right))
                count_y += y * width_lane
                count_total += width_lane

                # Road porcentage
                if self._check_area_lane:
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
            if self._check_area_lane:
                self._road_percentage = count_road / count_total * 100
                assert self._road_percentage >= self._threshold_road_per, "Low percentage of lane"

            if self._check_other_lane:
                if count_total_left > 0:
                    self._road_left = count_road_left / count_total_left * 100
                else:
                    self._road_left = 0
                if count_road_right > 0:
                    self._road_right = count_road_right / count_total_right * 100
                else:
                    self._road_right = 0

            # Draw center of mass and vehicle
            cv2.line(img, (x_cm, 0), (x_cm, SIZE_CAMERA - 1), (0, 255, 0), 2)
            cv2.line(img, (middle, 0), (middle, SIZE_CAMERA - 1), (255, 0, 0), 2)
            cv2.circle(img, (x_cm, y_cm), 9, (0, 255, 0), -1)
        else:
            self._deviation = SIZE_CAMERA / 2
            self._road_percentage = 0
            self._area = 0
            self._cm = np.array([0, 0], dtype=np.int32)
            self._lane_left = [] # Mark error
            assert False, "Area zero"

        '''
        if self._lane_network:
            print("Carril DL - Procesado:", time.time_ns() - t)
        else:
            print("Ground Truth - Procesado:", time.time_ns() - t)
        '''

        return img

    def get_deviation(self):
        return self._deviation
    
    def get_road_percentage(self):
        return self._road_percentage
    
    def get_lane_cm(self):
        return self._cm
    
    def get_lane_area(self):
        return self._area

    def show_surface(self, surface:pygame.Surface, pos:tuple[int, int], text:str):
        if text != None:
            write_text(text=text, img=surface, color=(0, 0, 0), side=RIGHT, bold=True,
                        size=self.size_text, point=(SIZE_CAMERA, 0))

        surface = pygame.transform.scale(surface, self.size)
        self._screen.blit(surface, pos)

    def process_data(self):
        image = self.data
        if image == None:
            self._mask = []
            return 
        
        image_data = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        image_data = np.reshape(image_data, (image.height, image.width, 4))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

        # Semantic segmentation
        canvas = image_data
        text_extra = self.text

        t = time.time_ns()
        if self._seg:
            if self.text != None:
                text_extra = "Segmented " + self.text
            else:
                text_extra = "Segmented view"

            image_pil = Image.fromarray(image_data)
            #t = time.time_ns()
            pred = self._seg_model.predict(image_pil)
            #print("EfficientVit - Predicción:", time.time_ns() - t)

            if self._canvas_seg:
                canvas, self._mask = self._seg_model.get_canvas(image_data, pred)
            else:
                self._mask = pred
                if (SIZE_CAMERA, SIZE_CAMERA) != self._mask.shape:
                    self._mask = cv2.resize(self._mask, dsize=(SIZE_CAMERA, SIZE_CAMERA), interpolation=cv2.INTER_NEAREST)
        #if self._seg:
        #    print("Segmentacion de la calzada:", time.time_ns() - t)

        t = time.time_ns()
        if self._lane:
            canvas = self._detect_lane(canvas, self._mask)
    
        if self.init != None and self.update_screen:
            surface = pygame.surfarray.make_surface(image_data[:, :, :3].swapaxes(0, 1))  
            self.show_surface(surface=surface, pos=self.init, text=self.text)

        if self.init_extra != None and self.update_screen:
            self._extra_surface = pygame.surfarray.make_surface(canvas[:, :, :3].swapaxes(0, 1))

            if self._lane:
                write_text(text="Deviation = "+str(int(abs(self._deviation)))+" (pixels)",
                        img=self._extra_surface, color=(0, 0, 0), side=LEFT_FRONT, size=self.size_text,
                        point=(0, SIZE_CAMERA - self.size_text), bold=True)
                if self._check_area_lane:
                    write_text(text=f"{self._road_percentage:.2f}% road", side=RIGHT, bold=True,
                            img=self._extra_surface, color=(0, 0, 0), size=self.size_text,
                            point=(SIZE_CAMERA, SIZE_CAMERA - self.size_text))          

            self.show_surface(surface=self._extra_surface, pos=self.init_extra, text=text_extra)  
        #if self._lane:
        #    print("Detección de carril:", time.time_ns() - t)

    def get_seg_data(self, num_points:int, show=False):
        if len(self._mask) <= 0 or len(self._lane_left) <= 0 or len(self._lane_right) <= 0:
            return (0, np.array([0, 0], dtype=np.int32), np.zeros((num_points, 2), dtype=np.int32),
                    np.zeros((num_points, 2), dtype=np.int32))

        index = np.argwhere(self._mask == ROAD)
        index_sorted = index[np.argsort(index[:, 0])] # Sorted by y

        area = len(index)
        y_cm = np.mean(index[:, 0])
        x_cm = np.mean(index[:, 1]) 
        cm = np.array([x_cm, y_cm], dtype=np.int32)

        if show:
            pygame.draw.circle(self._extra_surface, (255, 0, 255), (x_cm, y_cm), 9, 0)

        min_y = index_sorted[0, 0]
        max_y= index_sorted[-1, 0] 
        if self._lane:
            if self._lane_network:
                min_y = max(self._ymin_lane, min_y)
            else:
                min_y = max(min_y, self._lane_left[0][1], self._lane_right[0][1])

        points_final_left = points_final_right = 0
        for lane in range(2):
            try:
                if lane == LEFT_LANE:
                    index_limit = np.where(index_sorted[:, 1] == 0)[0][0]
                    color = (101, 67, 33)
                else:
                    index_limit = np.where(index_sorted[:, 1] == SIZE_CAMERA - 1)[0][0]
                    color = (0, 0, 0)
                y_middle = index_sorted[index_limit, 0]

                # First 3/4 of the points from the non-maximum part to have more information
                num_points_start = min(int(num_points * 3 / 4), y_middle - min_y)
                y_start = np.linspace(min_y, y_middle, num=num_points_start, dtype=int)

                # Last 1/4 of the points from the maximum part
                num_points_end = num_points - num_points_start
                y_end = np.linspace(y_middle + 1, max_y, num=num_points_end, dtype=int)

                # Combine both parts
                values_y = np.concatenate([y_start, y_end])
            except Exception:
                values_y = np.linspace(min_y, max_y, num_points, dtype=int)

            points_final = np.zeros((num_points, 2), dtype=np.int32)

            j = i = 0
            y = values_y[i]
            while True:
                points_in_y = index_sorted[index_sorted[:, 0] == y]

                if len(points_in_y) > 0:
                    points_final[j, 1] = y
                    if lane == LEFT_LANE:
                        points_final[j, 0] = np.min(points_in_y[:, 1])
                    else:
                        points_final[j, 0] = np.max(points_in_y[:, 1])

                    if show and self.init_extra != None:
                        pygame.draw.circle(self._extra_surface, color, (points_final[j, 0], y), 5, 0)

                    j += 1
                    i += 1
                    if i >= len(values_y):
                        break
                    y = values_y[i]
                else:
                    y += 1

            if lane == LEFT_LANE:
                points_final_left = points_final
            else:
                points_final_right = points_final

        return area, cm, points_final_left, points_final_right

    def get_lane_points(self, num_points:int=5, show:bool=False):
        if not self._lane or len(self._lane_left) == 0 or len(self._lane_right) == 0:
            return [np.zeros((num_points, 2), dtype=np.int32)] * 2
        
        lane_points = []
        for side in range(2):
            if not self._lane_network:
                if side == LEFT_LANE:
                    lane = self._lane_left
                else:
                    lane = self._lane_right
                y_points = np.linspace(lane[0][1], SIZE_CAMERA - 1, num_points)
            else:
                y_points = np.linspace(self._ymin_lane, SIZE_CAMERA - 1, num_points)

            points = np.zeros((num_points, 2), dtype=np.int32)
            for i, y in enumerate(y_points):
                if self._lane_network:
                    coef = self._coefficients[-1, side, 0:2]
                    points[i, 0] = int(y * coef[0] + coef[1]) # x
                else:
                    try:
                        points[i, 0] = lane[int(y - lane[0][1])][0] # x
                    except IndexError:
                        return [np.zeros((num_points, 2), dtype=np.int32)] * 2 
                points[i, 1] = y

                if points[i, 0] < 0:
                    points[i, 0] = 0
                elif points[i, 0] > SIZE_CAMERA - 1:
                    points[i, 0] = SIZE_CAMERA - 1

                if show and self.init_extra != None and self._extra_surface != None:
                    if side == LEFT_LANE:
                        color = (0, 0, 200)
                    else:
                        color = (200, 0, 0)

                    pygame.draw.circle(self._extra_surface, color, points[i], 6, 0)

            lane_points.append(points)

        if show and self.init_extra != None and self._extra_surface != None:
            self._screen.blit(self._extra_surface, self.init_extra)
            
        return lane_points
    
    def check_lane_right(self):
        return self._road_right >= self._threshold_road_per
    
    def check_lane_left(self):
        return self._road_left >= self._threshold_road_per
        
class Lidar(Sensor): 
    def __init__(self, size:tuple[int, int], init:tuple[int, int], sensor:carla.Sensor, scale:int, max_dist:int,
                 front_angle:int, yaw:float, screen:pygame.Surface, show_stats:bool=True, time_show:bool=True,
                 back_zone:bool=False):
        super().__init__(sensor=sensor)

        self._rect = init
        self.show_stats = show_stats
        self.time_show = time_show
        self._points = [np.nan] * NUM_ZONES
        self._points_x = [np.nan] * NUM_ZONES
        self._points_y = [np.nan] * NUM_ZONES
        self._max_dist = max_dist
        self._back = back_zone

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
        self._z_threshold_down = -1.6
        self._z_threshold_up = 1.6
        self._dist_threshold = np.full(NUM_ZONES, self._max_dist)
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

        # Add back right zone
        if self._back:
            back_angle = self._front_angle / 3
        else:
            back_angle = 0

        self._angles.append(angle2 + back_angle)
        self._angles.append(angle2 + back_angle * 2)

        if init != None:
            self._image = self._get_back_image()

    def _get_back_image(self):
        image = pygame.Surface(self._size_screen)
        mult = 10 * self._scale
        text = ['FL', 'FF', 'FR', 'R', 'BR']
        text_zone = ['Front-Left', 'Front-Front', 'Front-Right']

        for i in range(len(self._angles)):
            if (i != RIGHT and i != RIGHT_BACK) or ((i == RIGHT or i == RIGHT_BACK) and self._back):
                x_line = self._center_screen[0] + mult * math.cos(math.radians(self._angles[i]))
                y_line = self._center_screen[1] + mult * math.sin(math.radians(self._angles[i]))
                pygame.draw.line(image, (70, 70, 70), self._center_screen, (x_line, y_line), 2)

                if i < NUM_ZONES:
                    angle = self._angles[i] + self._front_angle / 6
                    x_zone = self._center_screen[0] + mult * math.cos(math.radians(angle))
                    y_zone = max(self._center_screen[1] + mult * math.sin(math.radians(angle)), 
                                self._size_text * (NUM_STATS + 2)) # Make sure to write behind the text
                    
                    write_text(text=text[i], img=image, point=(x_zone, y_zone), bold=True, size=self._size_text)

                    if self.show_stats and i != RIGHT and i != RIGHT_BACK:
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

    def get_points_zone(self, num_points:int, zone:int, show:bool=False):
        points = np.full(num_points, self._max_dist, dtype=np.float64)
        
        try:
            if np.isnan(self._points[zone]):
                pass
        except ValueError:
            if len(self._points[zone]) <= num_points:
                for i in range (len(self._points[zone])):
                    points[i] = self._points[zone][i]

                    if show and self._rect != None:
                        center = (int(self._points_x[zone][i] * self._scale + self._center_screen[0]),
                                  int(self._points_y[zone][i] * self._scale + self._center_screen[1]))
                        pygame.draw.circle(self._sub_screen, (0, 255, 0), center, 4, 0)
            else:
                j = 0
                index = np.linspace(0, len(self._points[zone]) - 1, num_points, dtype=int)

                for i in index:
                    points[j] = self._points[zone][i]
                    j += 1

                    if show and self._rect != None:
                        center = (int(self._points_x[zone][i] * self._scale + self._center_screen[0]),
                                  int(self._points_y[zone][i] * self._scale + self._center_screen[1]))
                        pygame.draw.circle(self._sub_screen, (0, 255, 0), center, 4, 0)
                
        return points
    
    def _update_stats(self, meas_zones):
        for zone in range(NUM_ZONES):
            # Filter distances by z
            filter_min = np.array(meas_zones[Z][zone]) > self._z_threshold_down
            filter_max = np.array(meas_zones[Z][zone]) < self._z_threshold_up
            filtered_dist_z = np.array(meas_zones[DIST][zone])[filter_min & filter_max]

            if len(filtered_dist_z) != 0:
                # Get points zone
                dist_mask = filtered_dist_z < self._dist_threshold[zone]
                filtered_dist = filtered_dist_z[dist_mask]

                x_values = np.array(meas_zones[X][zone])[filter_min & filter_max]
                filtered_x = x_values[dist_mask]

                y_values = np.array(meas_zones[Y][zone])[filter_min & filter_max]
                filtered_y = y_values[dist_mask]

                sorted_index = np.argsort(filtered_x)
                self._points[zone] = filtered_dist[sorted_index]
                self._points_x[zone] = filtered_x[sorted_index]
                self._points_y[zone] = filtered_y[sorted_index]

                # Get stats zone
                self._stat_zones[zone][MIN] = np.min(filtered_dist_z)
                self._stat_zones[zone][MEAN] = np.mean(filtered_dist_z)
                self._stat_zones[zone][MEDIAN] = np.median(filtered_dist_z)
                self._stat_zones[zone][STD] = np.std(filtered_dist_z)
            else:
                for i in range(NUM_STATS):
                    self._stat_zones[zone][i] = np.nan
                
                self._points[zone] = np.nan

            if self.show_stats and self._rect != None and zone != RIGHT_BACK and zone != RIGHT:
                if not self.time_show or time.time_ns() - self._time > SEG_TO_NANOSEG:
                    self._time = time.time_ns()
                    self._stats_text = [
                        "Mean = {:.2f}".format(self._stat_zones[zone][MEAN]),
                        "Median = {:.2f}".format(self._stat_zones[zone][MEDIAN]),
                        "Std = {:.2f}".format(self._stat_zones[zone][STD]),
                        "Min = {:.2f}".format(self._stat_zones[zone][MIN])
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
        
        #t = time.time_ns()
        lidar_data = np.copy(np.frombuffer(lidar.raw_data, dtype=np.dtype('f4')))
        lidar_data = np.reshape(lidar_data, (int(lidar_data.shape[0] / 4), 4))

        meas_zones = [
            [[] for _ in range(NUM_ZONES)], # dist_zones
            [[] for _ in range(NUM_ZONES)], # z_zones
            [[] for _ in range(NUM_ZONES)], # x_zones
            [[] for _ in range(NUM_ZONES)] # y_zones
        ]

        if self._rect != None and self.update_screen:
            self._sub_screen.blit(self._image, (0, 0))

        for x, y, z, i in lidar_data:
            zone = self._get_zone(x=x, y=y)
            if zone < NUM_ZONES and i < self._i_threshold:
                meas_zones[DIST][zone].append(math.sqrt(x ** 2 + y ** 2))
                meas_zones[Z][zone].append(z)
                meas_zones[X][zone].append(x)
                meas_zones[Y][zone].append(y)

            if self._rect != None and self.update_screen:
                thickness = self._interpolate_thickness(num=z)
                color = self._interpolate_color(num=i)

                center = (int(x * self._scale + self._center_screen[0]),
                          int(y * self._scale + self._center_screen[1]))
                pygame.draw.circle(self._sub_screen, color, center, thickness)

        self._update_stats(meas_zones=meas_zones)  
        #print("LiDAR:", time.time_ns() - t)

        if self._rect != None and self.update_screen:
            self._screen.blit(self._sub_screen, self._rect)

    def set_i_threshold(self, i:float):
        self._i_threshold = i

    def get_i_threshold(self):
        return self._i_threshold
    
    def set_z_threshold(self, z_down:float=None, z_up:float=None):
        if z_down != None:
            self._z_threshold_down = z_down
        if z_up != None:
            self._z_threshold_up = z_up

    def get_z_threshold(self):
        return self._z_threshold_down, self._z_threshold_up
    
    def get_dist_threshold(self):
        return self._dist_threshold
    
    def set_dist_threshold(self, dist:float, zone:int):
        self._dist_threshold[zone] = dist
    
    def get_stat_zones(self):
        return self._stat_zones
    
    def get_min(self, zone:int):
        return self._stat_zones[zone][MIN]
    
    def get_mean(self, zone:int):
        return self._stat_zones[zone][MEAN]

class Collision(Sensor):
    def __init__(self, sensor):
        super().__init__(sensor=sensor)

    def process_data(self):
        if self.data != None:
            other_actor_id = self.data.other_actor.type_id
            assert other_actor_id == 'static.road', f"The vehicle crashed with {other_actor_id}"

class Vehicle_sensors:
    def __init__(self, vehicle:carla.Vehicle, world:carla.World, screen:pygame.Surface=None, 
                 color_text:tuple[int, int, int]=(0, 0, 0), update_screen:bool=True):
        self._vehicle = vehicle
        self._world = world
        self._screen = screen
        self.sensors = []

        self.color_text = color_text
        self._time_frame = -1.0
        self._count_frame = 0
        self._write_frame = 0
        
        self.update_screen = update_screen

    def _put_sensor(self, sensor_type:str, transform:carla.Transform=carla.Transform(), type:int=0,
                    max_dist_laser:int=10, type_class:int=0):
        try:
            sensor_bp = self._world.get_blueprint_library().find(sensor_type)
        except IndexError:
            print("Sensor", sensor_type, "doesn't exist!")
            return None
        
        if type == LIDAR:
            sensor_bp.set_attribute('range', str(max_dist_laser))
            sensor_bp.set_attribute('channels', '30')

            if type_class == 0:
                sensor_bp.set_attribute('rotation_frequency', '20')
            elif type_class == 4:
                sensor_bp.set_attribute('rotation_frequency', '100') 
                sensor_bp.set_attribute('points_per_second', '200000') 
            else:
                sensor_bp.set_attribute('rotation_frequency', '110') 
                sensor_bp.set_attribute('points_per_second', '300000') 
        elif type == CAMERA:
            sensor_bp.set_attribute('image_size_x', str(SIZE_CAMERA))
            sensor_bp.set_attribute('image_size_y', str(SIZE_CAMERA))
                
        return self._world.spawn_actor(sensor_bp, transform, attach_to=self._vehicle)
    
    def add_camera_rgb(self, size_rect:tuple[int, int]=None, init:tuple[int, int]=None, seg:bool=False,
                       transform:carla.Transform=carla.Transform(), init_extra:tuple[int, int]=None,
                       text:str=None, lane:bool=False, canvas_seg:bool=True, lane_network:bool=False,
                       check_area_lane:bool=False, check_other_lane:bool=False):
        if self._screen == None:
            init = None
            init_extra = None

        sensor = self._put_sensor(sensor_type='sensor.camera.rgb', transform=transform, type=CAMERA)
        camera = CameraRGB(size=size_rect, init=init, sensor=sensor, screen=self._screen, seg=seg,
                           init_extra=init_extra, text=text, lane=lane, canvas_seg=canvas_seg,
                           transform=transform, vehicle=self._vehicle, world=self._world,
                           lane_network=lane_network, check_area_lane=check_area_lane,
                           check_other_lane=check_other_lane)
        self.sensors.append(camera)
        return camera
    
    def add_lidar(self, size_rect:tuple[int, int]=None, init:tuple[int, int]=None, scale:int=25, time_show:bool=True,
                  transform:carla.Transform=carla.Transform(), front_angle:int=150, show_stats:bool=True, 
                  max_dist:int=10, type_class:int=0, back_zone:int=0):
        if self._screen == None:
            init = None

        sensor = self._put_sensor(sensor_type='sensor.lidar.ray_cast', transform=transform, type=LIDAR, 
                                  max_dist_laser=max_dist, type_class=type_class)
        lidar = Lidar(size=size_rect, init=init, sensor=sensor, front_angle=front_angle, scale=scale,
                      max_dist=max_dist, yaw=transform.rotation.yaw, screen=self._screen, show_stats=show_stats,
                      time_show=time_show, back_zone=back_zone)
        
        self.sensors.append(lidar)
        return lidar
    
    def add_collision(self):
        sensor = self._put_sensor(sensor_type='sensor.other.collision')
        sensor_collision = Collision(sensor=sensor)
        self.sensors.append(sensor_collision)
        return sensor_collision

    def update_data(self, flip:bool=True, vel_ego:int=-1, vel_front:int=-1, front_laser:bool=False, mean:bool=False):
        offset = 20
        dist_lidar = np.nan

        for sensor in self.sensors:
            #t = time.time_ns()
            sensor.update_screen = self.update_screen
            sensor.process_data()

            if type(sensor) == Lidar:
                if not mean:
                    dist_lidar = sensor.get_min(zone=FRONT)
                else:
                    dist_lidar = sensor.get_mean(zone=FRONT)

                #print("LiDAR:", time.time_ns() - t)
            #elif type(sensor) == CameraRGB and sensor._lane:
                #print("Detección de carril:", time.time_ns() - t)

        if self._screen != None:
            elapsed_time = time.time_ns() - self._time_frame
            if elapsed_time > SEG_TO_NANOSEG: 
                self._write_frame = SEG_TO_NANOSEG * self._count_frame / elapsed_time
                self._count_frame = 0
                self._time_frame = time.time_ns()

            self._count_frame += 1
            if vel_ego < 0:
                vel_ego = carla.Vector3D(self._vehicle.get_velocity()).length()

            # Text to be written
            text_write = [
                f"FPS: {self._write_frame:.2f}",
                f"Vel ego: {vel_ego:.2f} m/s"
            ]

            if vel_front > 0:
                text_write.append(f"Vel front: {vel_front:.2f} m/s")
            if front_laser:
                text_write.append(f"Dist front lidar: {dist_lidar:.2f} m")

            # Write text
            for i in range(len(text_write)):
                write_text(text=text_write[i], img=self._screen, color=self.color_text, bold=True,
                           point=(2, offset * i), size=20, side=LEFT_FRONT)

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

        # Deviation
        self._kp_dev = 1 / (SIZE_CAMERA / 2)
        self._kd_dev = -self._kp_dev / 1.7

        self._error_dev = 0
        self._prev_error_dev = 0

        # Velocity
        self._vel_target = 10
        self._max_error_vel = 5

        self._kp_vel = 1.75
        #self._kd_dev = -self._kp / 1.7
    
        self._error_vel = 0
        self._prev_error_vel = 0

    def _scale_value(self, value:float):
        old_min = -self._max_error_vel
        old_max = self._max_error_vel
        new_min = 0.0
        new_max = 0.5

        return ((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

    def controll_vehicle(self, dev:float):
        control = carla.VehicleControl()

        # Deviation error
        self._prev_error_dev = self._error_dev
        self._error_dev = dev

        # Get velocity
        v = self._vehicle.get_velocity()
        v = carla.Vector3D(v).length()

        # Velocity error
        self._prev_error_vel = self._error_vel
        self._error_vel = np.clip(self._vel_target - v, -self._max_error_vel, self._max_error_vel)
        self._error_vel = self._scale_value(self._error_vel)

        # Control vel
        control.throttle = np.clip(self._kp_vel * self._error_vel, 0.1, 0.5)

        # Control steer
        if (self._error_dev > 0 and self._prev_error_dev < 0) or (self._error_dev < 0 and self._prev_error_dev > 0):
            self._prev_error_dev = 0        

        if self._error_dev > 20:
           self._error_dev *= 1.15

        # Control steer
        control.steer = self._kp_dev * self._error_dev + self._kd_dev * self._prev_error_dev

        # Apply control
        print("Throttle:", control.throttle, "\t||\tSteer:", control.steer)
        self._vehicle.apply_control(control)

        return control

def setup_carla(port:int=2000, name_world:str='Town01', fixed_delta_seconds:float=0.0, 
                client:carla.Client=None, syn:bool=False, overtaken:bool=False):
    if client == None:
        client = carla.Client('localhost', port)

    world = client.get_world()
    world = client.load_world(name_world)

    settings = world.get_settings()
    settings.fixed_delta_seconds = fixed_delta_seconds
    if fixed_delta_seconds > 0.1 or overtaken:
        settings.max_substep_delta_time = 0.05
        settings.max_substeps = 4

    if syn:
        settings.synchronous_mode = True
    else:
        settings.synchronous_mode = False
    world.apply_settings(settings)
    client.reload_world(False) # Reload world keeping settings

    return world, client

def add_one_vehicle(world:carla.World, ego_vehicle:bool=False, vehicle_type:str=None,
                    transform:carla.Transform=None): 
    if transform == None:
        spawn_points = world.get_map().get_spawn_points()
        transform = random.choice(spawn_points)

    if vehicle_type == None:
        vehicle_bp = random.choice(vehicle_bp)
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

def center_spectator(world:carla.World, transform:carla.Transform=carla.Transform(),
                     scale:float=5.5, height:float=3.0, pitch:float=-10.0):
    yaw = math.radians(transform.rotation.yaw)
    spectator =  world.get_spectator()

    copied_transform = carla.Transform(
        location=carla.Location(
            x=transform.location.x,
            y=transform.location.y,
            z=transform.location.z
        ),
        rotation=carla.Rotation(
            pitch=transform.rotation.pitch,
            yaw=transform.rotation.yaw,
            roll=transform.rotation.roll
        )
    )

    copied_transform.location.z = height
    copied_transform.location.x -= scale * math.cos(yaw)
    copied_transform.location.y -= scale * math.sin(yaw)
    copied_transform.rotation.pitch = pitch

    spectator.set_transform(copied_transform)
    return spectator

def setup_pygame(size:tuple[int, int], name:str=""):
    pygame.init()
    screen = pygame.display.set_mode(size, pygame.HWSURFACE | pygame.DOUBLEBUF)
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

def traffic_manager(client:carla.Client, vehicles:list[carla.Vehicle], port:int=5000, speed:float=None):
    tm = client.get_trafficmanager(port)
    tm_port = tm.get_port()
    tm.set_global_distance_to_leading_vehicle(2.0)

    if speed != None:
        tm.global_percentage_speed_difference(speed) 

    for v in vehicles:
        v.set_autopilot(True, tm_port)

    return tm