import gymnasium as gym
from gymnasium import spaces
import os
import sys
import numpy as np
import carla
import random
import pygame
import os
import csv
from abc import ABC, abstractmethod
import math

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, src_path)

from configcarla import SIZE_CAMERA, PATH
import configcarla

MAX_DEV = 100
MAX_DIST_LASER = 19.5
MIN_DIST_LASER = 4
MAX_VEL = 60

KEY_LASER = "laser"
KEY_LASER_RIGHT_FRONT = "Distance right front"
KEY_LASER_RIGHT = "Distance right"
KEY_LASER_RIGHT_BACK = "Distance right back"
KEY_VEL = "velocity"
KEY_DEV = "deviation"
KEY_CM = "cm"
KEY_LEFT_POINTS = "left_points"
KEY_RIGHT_POINTS = "right_points"
KEY_AREA = "area"

KEY_REWARD = "Reward"
KEY_STEPS = "Steps"
KEY_EPISODE = "Episode"
KEY_FINISH = "Finish"
KEY_MEAN_VEL = "Mean vel"
KEY_EXP_RATE = "Exploration_rate"
KEY_ACC_REWARD = "Accumulated reward"
KEY_DISTANCE = "Distance"

KEY_SEG_CM = "seg cm"
KEY_SEG_AREA = "seg area"
KEY_SEG_POINTS_LEFT = "seg left points"
KEY_SEG_POINTS_RIGHT = "seg right points"

KEY_THROTTLE = "Throttle"
KEY_STEER = "Steer"
KEY_BRAKE = "Brake"

KEY_ID = "id"
KEY_TOWN = "town"
KEY_LOC = "location"
CIRCUIT_CONFIG = {
    0: [
        {KEY_ID: 0, KEY_TOWN: "Town04", KEY_LOC: carla.Transform(carla.Location(x=352.65, y=-351, z=0.1), carla.Rotation(yaw=-137))},
        {KEY_ID: 1, KEY_TOWN: "Town04", KEY_LOC: carla.Transform(carla.Location(x=-8.76, y=60.8, z=0.1), carla.Rotation(yaw=89.7))},
        {KEY_ID: 2, KEY_TOWN: "Town04", KEY_LOC: carla.Transform(carla.Location(x=-25.0, y=-252, z=0.1), carla.Rotation(yaw=125.0))}
    ],
    1: [
        {KEY_ID: 5, KEY_TOWN: "Town04", KEY_LOC: carla.Transform(carla.Location(x=352.65, y=-351, z=0.1), carla.Rotation(yaw=-137))}
    ],
    2: [
        {KEY_ID: 3, KEY_TOWN: "Town04", KEY_LOC: carla.Transform(carla.Location(x=13.5, y=310, z=0.1), carla.Rotation(yaw=-48))}
    ],
    3: [
        {KEY_ID: 4, KEY_TOWN: "Town03", KEY_LOC: carla.Transform(carla.Location(x=114, y=207.3, z=1.7))}
    ],
    4: [
        {KEY_ID: 6, KEY_TOWN: "Town06", KEY_LOC: carla.Transform(carla.Location(x=457.83, y=244.7, z=0.1))}
    ],
    5: [
        {KEY_ID:7, KEY_TOWN: "Town05", KEY_LOC: carla.Transform(carla.Location(x=75.0, y=-144.5, z=0.1), carla.Rotation(yaw=4.0))}
    ]
}

class CarlaBase(gym.Env, ABC):
    def __init__(self, human:bool, train:bool, config:list, alg:str=None, port:int=2000, num_points:int=5,
                 seg:bool=False, passing:bool=False, fixed_delta_seconds:float=0.0, normalize:bool=False,
                 back_lidar:bool=False, seed:int=None, num_cir:int=0, port_tm:int=1111,
                 lane_network:bool=False, target_vel:int=7):
        self._first = True
        self._dev = 0
        self._dist_laser = MAX_DIST_LASER
        self._steer = 0
        self._velocity = 0
        self._count_ep = 0
        self._count_steps = 0
        self._total_reward = 0
        self._count = 0
        self._human = human
        self._velocity = 0 # It must be update in reward function
        self._passing = passing
        self._front_vehicle = None
        self._exploration_rate = 1.0
        self._lane_network = lane_network
        self._back_lidar = back_lidar
        self._dist_back = MAX_DIST_LASER
        self._seg = seg
        self._target_vel = target_vel

        if passing:
            assert num_cir <= 1, "No passing mode available for circuit " + str(num_cir) 
        assert (self._lane_network and num_cir == 5) or (not self._lane_network and num_cir != 5), "No matching circuit and type of lane detection"

        # CSV file
        self._train = train
        if self._train:
            assert alg != None, "Algorithms are required for training"

            # Episode file
            dir_csv = PATH + '2024-tfg-lara-poves/src/deepRL/csv/train/' + self.__class__.__name__ + '/'
            if not os.path.exists(dir_csv):
                os.makedirs(dir_csv)
            files = os.listdir(dir_csv)
            num_files = len(files) + 1
            self._train_csv = open(dir_csv + alg + '_train_data_' + str(num_files) + '.csv',
                                   mode='w', newline='')
            self._writer_csv_train = csv.writer(self._train_csv)
            self._writer_csv_train.writerow([KEY_EPISODE, KEY_REWARD, KEY_STEPS, KEY_FINISH, KEY_DEV,
                                             KEY_EXP_RATE, KEY_LASER, KEY_MEAN_VEL])

            # Action file
            dir_csv = PATH + '2024-tfg-lara-poves/src/deepRL/csv/actions/' + self.__class__.__name__ + '/'
            if not os.path.exists(dir_csv):
                os.makedirs(dir_csv)
            files = os.listdir(dir_csv)
            num_files = len(files) + 1
            self._actions_csv = open(dir_csv + alg + '_train_actions_' + str(num_files) + '.csv',
                                     mode='w', newline='')
            self._writer_csv_actions = csv.writer(self._actions_csv)
            self._writer_csv_actions.writerow([KEY_THROTTLE, KEY_STEER, KEY_BRAKE])

        # States/Observations
        self._num_points_lane = num_points
        self.observation_space = spaces.Dict(
            spaces={
                KEY_CM: spaces.Box(
                    low=0,
                    high=SIZE_CAMERA - 1,
                    shape=(2,),
                    dtype=np.int32
                ),
                
                KEY_LEFT_POINTS: spaces.Box(
                    low=0,
                    high=SIZE_CAMERA - 1,
                    shape=(self._num_points_lane, 2),
                    dtype=np.int32
                ),

                KEY_RIGHT_POINTS: spaces.Box(
                    low=0,
                    high=SIZE_CAMERA - 1,
                    shape=(self._num_points_lane, 2),
                    dtype=np.int32
                ),

                KEY_AREA: spaces.Box(
                    low=0,
                    high=SIZE_CAMERA * SIZE_CAMERA,
                    shape=(1,),
                    dtype=np.int32
                ),

                KEY_DEV: spaces.Box(
                    low=-SIZE_CAMERA / 2,
                    high=SIZE_CAMERA / 2,
                    shape=(1,),
                    dtype=np.int32
                )
            }
        )

        self.normalize = normalize
        if self.normalize:
            self._obs_norm = self.observation_space # Use to normalize
            self.observation_space = spaces.Dict(
            spaces={
                KEY_CM: spaces.Box(
                    low=0,
                    high=1,
                    shape=(2,),
                    dtype=np.float64
                ),
                
                KEY_LEFT_POINTS: spaces.Box(
                    low=0,
                    high=1,
                    shape=(self._num_points_lane, 2),
                    dtype=np.float64
                ),

                KEY_RIGHT_POINTS: spaces.Box(
                    low=0,
                    high=1,
                    shape=(self._num_points_lane, 2),
                    dtype=np.float64
                ),

                KEY_AREA: spaces.Box(
                    low=0,
                    high=1,
                    shape=(1,),
                    dtype=np.float64
                ),

                KEY_DEV: spaces.Box(
                    low=0,
                    high=1,
                    shape=(1,),
                    dtype=np.float64
                )
            }
        )
            
        # Other parameters
        self._create_actions()
        self.reward_range = (0.0, 1.0)
        self.np_random = seed
        self.model = None

        # Pygame window
        if self._human:
            mult = 2
            if self._passing:
                mult = 3

            if not self._back_lidar:
                self._init_laser = (SIZE_CAMERA * 2, 0)
                rows = 1 
            else:
                self._init_laser = (0, SIZE_CAMERA)
                self._init_back = (SIZE_CAMERA, SIZE_CAMERA)
                mult = 2
                rows = 2
                
            self._init_driver = (SIZE_CAMERA, 0)
            self._screen = configcarla.setup_pygame(size=(SIZE_CAMERA * mult, SIZE_CAMERA * rows), 
                                                    name='Follow lane: ' + self.__class__.__name__)
        else:
            self._screen = None
            self._init_driver = None
            self._init_laser = None

        # Init simulation
        if self._train:
            assert fixed_delta_seconds > 0.0, "In synchronous mode fidex_delta_seconds can't be 0.0"

        self._fixed_delta_seconds = fixed_delta_seconds        
        self._town_locations = [(entry[KEY_ID], entry[KEY_TOWN], entry[KEY_LOC]) for entry in config]
        self._client = None
        self._port = port
        self._port_tm = port_tm

        if self._back_lidar:
            self._town_locations.pop()

    def _swap_ego_vehicle(self):
        if self._train:
            self._id, self._town, self._loc = random.choice(self._town_locations)
        else:
            self._id, self._town, self._loc = self._town_locations[0]

        if self._client == None or not self._town in self._world.get_map().name: 
            self._world, self._client = configcarla.setup_carla(name_world=self._town, client=self._client,
                                                                syn=self._train, port=self._port,
                                                                fixed_delta_seconds=self._fixed_delta_seconds)
        
            # Set the weather to sunny
            if self._town == 'Town04':
                weather = carla.WeatherParameters(
                    cloudiness=10.0,   
                    precipitation=0.0,  
                    sun_altitude_angle=30.0  
                )
                self._world.set_weather(weather)

        self._map = self._world.get_map()
        self.ego_vehicle = configcarla.add_one_vehicle(world=self._world, ego_vehicle=True, vehicle_type='vehicle.lincoln.mkz_2020',
                                                       transform=self._loc)
        self._sensors = configcarla.Vehicle_sensors(vehicle=self.ego_vehicle, world=self._world,
                                                    screen=self._screen)
        if self._lane_network:
            transform = carla.Transform(carla.Location(z=1.4, x=1.75))
        else:
            transform = carla.Transform(carla.Location(x=0.5, z=1.7292))

        self._camera = self._sensors.add_camera_rgb(transform=transform, seg=self._seg, lane=True, lane_network=self._lane_network,
                                                    canvas_seg=False, size_rect=(SIZE_CAMERA, SIZE_CAMERA), check_lane_left=True,
                                                    init_extra=self._init_driver, text='Driver view', check_area_lane=False)
        self._sensors.add_collision() # Raise an exception if the vehicle crashes

        if self._human:
            world_transform = carla.Transform(carla.Location(z=2.5, x=-4.75))
            self._sensors.add_camera_rgb(transform=world_transform, size_rect=(SIZE_CAMERA, SIZE_CAMERA),
                                         init=(0, 0), text='World view')
            
        if self._passing: 
            if self._back_lidar and self._human:
                back_transform = carla.Transform(carla.Location(z=1.5, x=-1.5), carla.Rotation(yaw=180))
                self._sensors.add_camera_rgb(size_rect=(SIZE_CAMERA, SIZE_CAMERA), init=self._init_back,
                                                transform=back_transform, text='Back view')

            lidar_transform = carla.Transform(carla.Location(x=-0.5, z=1.8), carla.Rotation(yaw=90.0))
            self._lidar = self._sensors.add_lidar(init=self._init_laser,size_rect=(SIZE_CAMERA, SIZE_CAMERA), back_zone=self._back_lidar,
                                                  transform=lidar_transform, scale=13, time_show=False, show_stats=False,
                                                  train=self._train, max_dist=MAX_DIST_LASER, front_angle=150) # 50º each part
            self._lidar.set_z_threshold(z_up=1.7, z_down=-1.4)

            t = carla.Transform(
                location=self._loc.location,
                rotation=self._loc.rotation
            )
            
            prob = random.random()
            if self._id == 0 or self._id == 5:
                if prob < 0.4: # 4.4m
                    t.location.x -= 5
                    t.location.y -= 4
                    t.rotation.yaw -= 6
                elif prob < 0.7: # 10m
                    t.location.x -= 10
                    t.location.y -= 8
                    t.rotation.yaw -= 6
                else: # 8m
                    t.location.x -= 8
                    t.location.y -= 6
                    t.rotation.yaw -= 6
            elif self._id == 1:
                if prob < 0.5: # 5m
                    t.location.y += 7
                else: # 9m
                    t.location.y += 11
            else:
                if prob < 0.4: # 6.6m
                    t.location.y += 7
                    t.location.x -= 5
                elif prob < 0.7: # 4.4m
                    t.location.y += 5
                    t.location.x -= 4
                else: # 10m
                    t.location.y += 10
                    t.location.x -= 7

            # Front vehicle
            '''
            self._front_vehicle = configcarla.add_one_vehicle(world=self._world, transform=t,
                                                              vehicle_type='vehicle.carlamotors.carlacola')

            # Set traffic manager
            self._tm = configcarla.traffic_manager(client=self._client, vehicles=[self._front_vehicle], port=self._port_tm)
            self._tm.ignore_lights_percentage(self._front_vehicle, 100)
            '''

    def _get_obs_env(self):
        left_points, right_points = self._camera.get_lane_points(num_points=self._num_points_lane)
        obs = {
            KEY_CM: self._camera.get_lane_cm(), 
            KEY_LEFT_POINTS: left_points,
            KEY_RIGHT_POINTS: right_points, 
            KEY_AREA: np.array([self._camera.get_lane_area()], dtype=np.int32),
            KEY_DEV: np.array([self._dev], dtype=np.int32),
        }

        return obs

    def _get_obs(self):
        obs = self._get_obs_env()

        if self.normalize:
            for key, sub_space in self._obs_norm.spaces.items():
                obs[key] = (obs[key] - sub_space.low) / (sub_space.high - sub_space.low)
                obs[key] = obs[key].astype(np.float64)
                assert obs[key] >= 0 and obs[key] <= 1, "Fallo en obs"

        return obs
    
    def _get_info(self):
        info = {KEY_DEV: self._dev, KEY_VEL: self._velocity}
        return info

    def step(self, action:np.ndarray):
        terminated = False
        finish_ep = False
        error = None

        # Exec action
        control = self._get_control(action)
        self.ego_vehicle.apply_control(control)
        if self._train:
            self._writer_csv_actions.writerow([control.throttle, control.steer, control.brake])

        try:
            # Tick
            if self._train:
                self._world.tick()

            # Get velocity and location
            self._velocity = carla.Vector3D(self.ego_vehicle.get_velocity()).length()
            self._mean_vel += self._velocity
            loc = self.ego_vehicle.get_location()

            if self._back_lidar and self._passing:
                self._dist_back = self._lidar.get_min_back()

            # Update data
            if self._passing:
                self._sensors.update_data(vel_ego=self._velocity, vel_front=self._target_vel,
                                          front_laser=True, back_lidar=self._back_lidar)
            else:
                self._sensors.update_data(vel_ego=self._velocity)

            # Get deviation
            dev_prev = self._dev
            self._dev = self._camera.get_deviation()

            # Obstacle trainings
            if self._passing:
                # Check distance to front car
                self._dist_laser = self._lidar.get_min(zone=configcarla.FRONT)
                assert np.isnan(self._dist_laser) or self._dist_laser > MIN_DIST_LASER, "Distance exceeded"

            # Lane change detection
            if abs(self._dev - dev_prev) > 50:
                self._dev = dev_prev
                assert False, "Lost lane: changing lane"

            # Check if the episode has finished
            if self._id == 0:
                finish_ep = abs(loc.x + 7) <= 3 and abs(loc.y - 55) <= 3
            elif self._id == 1 or self._id == 5:
                finish_ep =  abs(loc.x + 442) <= 3 and abs(loc.y - 30) <= 3
            elif self._id == 2:
                finish_ep = loc.y > -24.5
            elif self._id == 3:
                finish_ep = abs(loc.x - 414) <= 3 and abs(loc.y + 230) <= 3
            elif self._id == 6:
                finish_ep = abs(loc.x - 663) < 3 and abs(loc.y - 169) < 3
            elif self._id == 7 and not self._jump and loc.y > -30:
                self._jump = True
                t = self.ego_vehicle.get_transform()
                t.location.y = 15
                self.ego_vehicle.set_transform(t)
            elif self._id == 7 and self._jump:
                finish_ep = loc.x < 50
            else:
                finish_ep = abs(loc.x - 165) <= 3 and abs(loc.y + 208) <= 3
            terminated = finish_ep
            
        except AssertionError as e:
            terminated = True
            error = str(e)

            print("Circuit not completed:", error)
            print("No termino", self._count_ep, "steps:", self._count, "id:", self._id, "vel target:", self._target_vel,
                  "dev:", self._dev, "is_passing:", self._passing, "dist:", self._dist_laser,
                  "back:", self._back_lidar, "mean vel:", self._mean_vel / self._count)

        # Check if a key has been pressed
        if self._human:
            self.render()

        reward = self._calculate_reward(error)
        self._total_reward += reward
        self._count += 1

        if finish_ep:
            print("Circuit completed!")
            print("Termino:", self._count_ep, "steps:", self._count, "id:", self._id, "target vel:", self._target_vel,
                 "mean vel", self._mean_vel / self._count, "passing:", self._passing)

        if terminated and self._train:
            if self.model != None:
                self._exploration_rate = self.model.exploration_rate
            else:
                self._exploration_rate = -1.0 # No register

            self._count_ep += 1
            self._writer_csv_train.writerow([self._count_ep, self._total_reward, self._count, finish_ep,
                                             self._dev, self._exploration_rate, self._dist_laser, 
                                             self._mean_vel / self._count])
        
        return self._get_obs(), reward, terminated, False, self._get_info()
    
    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Use Ctrl+C to stop the execution")

    def reset(self, seed=None):
        super().reset(seed=seed)

        # Reset variables
        self._total_reward = 0
        self._count = 0
        self._first_step = 0
        self._mean_vel = 0
        self._jump = False

        if self._first:
            self._first = False
        else:
            self._sensors.destroy()

            if self._front_vehicle != None and self._passing:
                try:
                    self._front_vehicle.destroy()
                except RuntimeError:
                    pass # It's so far away that it's not visible and not being processed.

        self._swap_ego_vehicle()

        # Set target velocity (front vehicle)
        if self._passing:
            self._tm.set_desired_speed(self._front_vehicle, self._target_vel * 3.6) # km/h

        while True:
            try:
                if self._train:
                    self._world.tick()
                self._sensors.update_data()

                if self._camera.data != None:
                    self._dev = self._camera.get_deviation()
                    break 

            except AssertionError:
                pass
        
        return self._get_obs(), self._get_info()

    def close(self):
        self._sensors.destroy()

        if self._train:
            self._train_csv.close()
            self._actions_csv.close()

        pygame.quit()

    @abstractmethod
    def _get_control(self, action:np.ndarray):
        pass

    @abstractmethod
    def _create_actions(self):
        pass

    @abstractmethod
    def _calculate_reward(self, error:str):
        pass

class CarlaLaneDiscrete(CarlaBase):
    def __init__(self, human:bool, train:bool, alg:str=None, port:int=2000, num_cir:int=0, retrain:bool=False, target_vel:int=7,
                 fixed_delta_seconds:float=0.0, normalize:bool=False, seed:int=None, lane_network:bool=False):
        if train:
            num_cir = 0
        config = CIRCUIT_CONFIG.get(num_cir, CIRCUIT_CONFIG[0])

        super().__init__(human=human, train=train, alg=alg, port=port, seed=seed, num_cir=num_cir, config=config,
                         normalize=normalize, fixed_delta_seconds=fixed_delta_seconds, lane_network=lane_network)
        
        self._max_vel = 15

    def _create_actions(self):
        self.action_to_control = {
            0: (0.5, 0.0),
            1: (0.45, 0.01), 
            2: (0.45, -0.01),
            3: (0.4, 0.02),
            4: (0.4, -0.02),
            5: (0.4, 0.04),
            6: (0.4, -0.04),
            7: (0.4, 0.06),
            8: (0.4, -0.06),
            9: (0.4, 0.08),
            10: (0.4, -0.08),
            11: (0.3, 0.1),
            12: (0.3, -0.1),
            13: (0.3, 0.12),
            14: (0.3, -0.12),
            15: (0.2, 0.14),
            16: (0.2, -0.14),
            17: (0.2, 0.16),
            18: (0.2, -0.16),
            19: (0.1, 0.18),
            20: (0.1, -0.18)
        }
        self.action_space = spaces.Discrete(len(self.action_to_control))
        
    def set_model(self, model):
        self.model = model
    
    def _get_control(self, action:np.ndarray):
        throttle, steer = self.action_to_control[int(action)]
        control = carla.VehicleControl()
        control.steer = steer 
        control.throttle = throttle
        
        return control
    
    def _calculate_reward(self, error:str):
        if error == None:
            # Clip deviation and velocity
            r_dev = (MAX_DEV - abs(np.clip(self._dev, -MAX_DEV, MAX_DEV))) / MAX_DEV
            r_vel = np.clip(self._velocity, 0.0, self._max_vel) / self._max_vel
            reward = 0.8 * r_dev + 0.2 * r_vel
        else:
            reward = -30

        return reward

class CarlaLaneContinuous(CarlaBase):
    def __init__(self, human:bool, train:bool, alg:str=None, port:int=200, num_cir:int=0, 
                 retrain:bool=False, fixed_delta_seconds:float=0.0, normalize:bool=False,
                 seed:int=None, port_tm:int=1111, lane_network:bool=False, target_vel:int=7):
        if train and human:
            human = False
            print("Warning: Can't activate human mode during training")

        if train:
            num_cir = 0
        config = CIRCUIT_CONFIG.get(num_cir, [])

        super().__init__(human=human, train=train, alg=alg, port=port, seed=seed, num_points=10,
                         normalize=normalize, fixed_delta_seconds=fixed_delta_seconds,
                         num_cir=num_cir, config=config, lane_network=lane_network)
        
        self._max_vel = 45

        # Add velocity to observations
        new_space = spaces.Box(
            low=0.0,
            high=MAX_VEL,
            shape=(1,),
            dtype=np.float64
        )

        if not normalize:
            self.observation_space[KEY_VEL] = new_space
        else:
            self._obs_norm[KEY_VEL] = new_space
            self.observation_space[KEY_VEL] =  spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float64
            )

    def _get_obs_env(self):
        obs = super()._get_obs_env()
        obs[KEY_VEL] = self._velocity
        return obs

    def _create_actions(self):
        self._max_steer = 0.2
        self.action_space = spaces.Box(low=np.array([0.0, -self._max_steer]), high=np.array([1.0, self._max_steer]),
                                       shape=(2,), dtype=np.float64)
        
    def _get_control(self, action:np.ndarray):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self._throttle, self._steer = action
        
        control = carla.VehicleControl()
        control.steer = self._steer 
        control.throttle = self._throttle

        return control
    
    def _calculate_reward(self, error:str):
        if error == None:
            # Deviation normalization
            r_dev = (MAX_DEV - abs(np.clip(self._dev, -MAX_DEV, MAX_DEV))) / MAX_DEV
            
            # Steer conversion
            limit_steer = 0.14
            if abs(self._steer) > limit_steer:
                r_steer = 0
            else:
                r_steer = (limit_steer - abs(self._steer)) / limit_steer

            # Throttle conversion
            limit_throttle = 0.6
            if self._throttle >= limit_throttle:
                r_throttle = 0
            elif self._velocity > self._max_vel:
                r_throttle = (limit_throttle - self._throttle) / limit_throttle
            else:
                r_throttle = self._throttle / limit_throttle

            # Set weights
            if r_steer == 0:
                w_dev = 0.1
                w_throttle = 0.1
                w_steer = 0.8
            elif r_throttle == 0:
                w_dev = 0.1
                w_throttle = 0.8
                w_steer = 0.1
            elif self._velocity > self._max_vel:
                w_dev = 0.1
                w_throttle = 0.65
                w_steer = 0.25
            elif self._throttle < 0.5:
                w_dev = 0.65
                w_throttle = 0.25
                w_steer = 0.1 # Lower accelerations, penalize large turns less
            else: # [0.5, 0.6) throttle
                w_dev = 0.6
                w_throttle = 0.15
                w_steer = 0.25

            reward = w_dev * r_dev + w_throttle * r_throttle + w_steer * r_steer
        else:
            reward = -40

        return reward

class CarlaObstacle(CarlaBase):
    def __init__(self, human:bool, train:bool, alg:str=None, port:int=200, num_cir:int=0, port_tm:int=1111,
                 fixed_delta_seconds:float=0.0, normalize:bool=False, seed:int=None, retrain:bool=False,
                 lane_network:bool=False, target_vel:int=7):
        if train and human:
            human = False
            print("Warning: Can't activate human mode during training")

        if train:
            num_cir = 0
        else:
            retrain = True
        config = CIRCUIT_CONFIG.get(num_cir, [])

        super().__init__(human=human, train=train, alg=alg, port=port, seed=seed, num_points=10, target_vel=target_vel,
                         normalize=normalize, fixed_delta_seconds=fixed_delta_seconds, port_tm=port_tm,
                         num_cir=num_cir, config=config, passing=retrain, lane_network=lane_network) 
        
        self._max_vel = 20

        # Add velocity to observations
        new_space_vel = spaces.Box(
            low=0.0,
            high=MAX_VEL,
            shape=(1,),
            dtype=np.float64
        )

        # Add laser front distance to observations
        self._num_points_laser = 20
        new_space_laser = spaces.Box(
            low=MIN_DIST_LASER - 1.0,
            high=MAX_DIST_LASER,
            shape=(self._num_points_laser,),
            dtype=np.float64
        )

        if not normalize:
            self.observation_space[KEY_VEL] = new_space_vel
            self.observation_space[KEY_LASER] = new_space_laser
        else:
            self._obs_norm[KEY_VEL] = new_space_vel
            self.observation_space[KEY_VEL] =  spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float64
            )
            self._obs_norm[KEY_LASER] = new_space_laser
            self.observation_space[KEY_LASER] =  spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self._num_points_laser,),
                dtype=np.float64
            )
            
    def _get_obs_env(self):
        obs = super()._get_obs_env()
        obs[KEY_VEL] = self._velocity

        if not self._passing or np.isnan(self._dist_laser):
            obs[KEY_LASER] = np.full(self._num_points_laser, MAX_DIST_LASER, dtype=np.float64)
        else:
            obs[KEY_LASER] = self._lidar.get_points_zone(self._num_points_laser, zone=configcarla.FRONT)

        return obs
    
    def _get_info(self):
        info = super()._get_info()
        info[KEY_LASER] = self._dist_laser
        return info

    def _create_actions(self):
        self._max_steer = 0.2
        self.action_space = spaces.Box(low=np.array([0.0, -self._max_steer]), high=np.array([1.0, self._max_steer]),
                                       shape=(2,), dtype=np.float64)
        
    def _get_control(self, action:np.ndarray):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self._throttle, self._steer = action
        
        control = carla.VehicleControl()
        control.steer = self._steer 
        control.throttle = self._throttle

        return control
    
    def _calculate_reward(self, error:str):
        if error == None:
            # Deviation normalization
            r_dev = (MAX_DEV - abs(np.clip(self._dev, -MAX_DEV, MAX_DEV))) / MAX_DEV
            
            # Steer conversion
            limit_steer = 0.14
            if abs(self._steer) > limit_steer:
                r_steer = 0
            else:
                r_steer = (limit_steer - abs(self._steer)) / limit_steer

            # Throttle conversion
            limit_throttle = 0.6
            if self._throttle >= limit_throttle:
                r_throttle = 0
            elif self._velocity > self._max_vel:
                r_throttle = (limit_throttle - self._throttle) / limit_throttle
            else:
                r_throttle = self._throttle / limit_throttle

            # Laser conversion
            laser_threshold = 8
            if self._passing and not np.isnan(self._dist_laser):
                r_laser = np.clip(self._dist_laser, MIN_DIST_LASER, MAX_DIST_LASER) - MIN_DIST_LASER
                r_laser /= (MAX_DIST_LASER - MIN_DIST_LASER)

                if self._dist_laser <= laser_threshold:
                    r_throttle = (limit_throttle - self._throttle) / limit_throttle              
            else:
                r_laser = 0

            # Set weights
            if r_steer == 0:
                w_dev = 0.1
                w_throttle = 0.1
                w_steer = 0.8
                w_laser = 0
            elif r_throttle == 0:
                w_dev = 0.1
                w_throttle = 0.8
                w_steer = 0.1
                w_laser = 0
            elif self._velocity > self._max_vel:
                w_dev = 0.1
                w_throttle = 0.65
                w_steer = 0.25
                w_laser = 0
            elif r_laser != 0:
                if self._dist_laser <= laser_threshold: # Short distance (4, 8]
                    w_dev = 0.2
                    w_throttle = 0.1 # brake
                    w_steer = 0
                    w_laser = 0.7
                elif self._dist_laser <= 12: # Medium distance (8, 12]
                    w_dev = 0.45
                    w_throttle = 0.0
                    w_steer = 0.05
                    w_laser = 0.5
                else: # Large distance (12, 19.5]
                    w_dev = 0.45
                    w_throttle = 0.1
                    w_laser = 0.35
                    w_steer = 0.1
            elif self._throttle < 0.5:
                w_dev = 0.65 
                w_throttle = 0.25 
                w_steer = 0.1 # Lower accelerations, penalize large turns less
                w_laser = 0
            else: # [0.5, 0.6) throttle
                w_dev = 0.6
                w_throttle = 0.15
                w_steer = 0.25
                w_laser = 0

            reward = w_dev * r_dev + w_throttle * r_throttle + w_steer * r_steer + r_laser * w_laser
        else:
            if "Distance" in error:
                reward = -60
            else:
                reward = -40

        return reward

class CarlaPassing(CarlaBase):
    def __init__(self, human:bool, train:bool, alg:str=None, port:int=200, num_cir:int=0, port_tm:int=1111,
                 fixed_delta_seconds:float=0.0, normalize:bool=False, seed:int=None, retrain:int=0,
                 lane_network:bool=False, target_vel:int=7):
        if train and human:
            human = False
            print("Warning: Can't activate human mode during training")

        if train:
            num_cir = 0
            back = False
            seg = False

            if retrain == 2:
                back = True
                seg = True
        else:
            retrain = True # Front and back lidar
            back = True
            seg = True
        config = CIRCUIT_CONFIG.get(num_cir, [])

        super().__init__(human=human, train=train, alg=alg, port=port, seed=seed, num_points=10,
                         normalize=normalize, fixed_delta_seconds=fixed_delta_seconds, port_tm=port_tm,
                         num_cir=num_cir, config=config, passing=retrain, lane_network=lane_network,
                         back_lidar=back, seg=seg, target_vel=target_vel)
        
        self._max_vel = 25

        # Add velocity to observations
        self.observation_space[KEY_VEL] = spaces.Box(
            low=0.0,
            high=MAX_VEL,
            shape=(1,),
            dtype=np.float64
        )

        # Add laser front distance to observations
        self._num_points_laser = 20
        self.observation_space[KEY_LASER] = spaces.Box(
            low=MIN_DIST_LASER - 1.0,
            high=MAX_DIST_LASER,
            shape=(self._num_points_laser,),
            dtype=np.float64
        )

        # Add laser three right zones distance to observations
        self._num_points_laser_right = 10
        self.observation_space[KEY_LASER_RIGHT] = spaces.Box(
            low=MIN_DIST_LASER - 1.0,
            high=MAX_DIST_LASER,
            shape=(self._num_points_laser_right,),
            dtype=np.float64
        )
        self.observation_space[KEY_LASER_RIGHT_BACK] = spaces.Box(
            low=MIN_DIST_LASER - 1.0,
            high=MAX_DIST_LASER,
            shape=(self._num_points_laser_right,),
            dtype=np.float64
        )
        self.observation_space[KEY_LASER_RIGHT_FRONT] = spaces.Box(
            low=MIN_DIST_LASER - 1.0,
            high=MAX_DIST_LASER,
            shape=(self._num_points_laser_right,),
            dtype=np.float64
        )
        
        # Add segmentations observations
        '''
        self.observation_space[KEY_SEG_AREA] = spaces.Box(
            low=0,
            high=SIZE_CAMERA * SIZE_CAMERA,
            shape=(1,),
            dtype=np.int32
        )
        
        self.observation_space[KEY_SEG_CM] = spaces.Box(
            low=0,
            high=SIZE_CAMERA - 1,
            shape=(2,),
            dtype=np.int32
        )

        self._num_points_seg = int(self._num_points_lane * 2)
        new_space_seg_points = spaces.Box(
            low=0,
            high=SIZE_CAMERA - 1,
            shape=(self._num_points_seg,),
            dtype=np.int32
        )
        self.observation_space[KEY_SEG_POINTS_LEFT] = new_space_seg_points
        self.observation_space[KEY_SEG_POINTS_RIGHT] = new_space_seg_points
        '''
        
        if normalize:
            self._obs_norm = self.observation_space

            self.observation_space[KEY_VEL] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float64
            )
            self.observation_space[KEY_LASER] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self._num_points_laser,),
                dtype=np.float64
            )
            
            self.observation_space[KEY_LASER_RIGHT_FRONT] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self._num_points_laser_right,),
                dtype=np.float64
            )
            self.observation_space[KEY_LASER_RIGHT] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self._num_points_laser_right,),
                dtype=np.float64
            )
            self.observation_space[KEY_LASER_RIGHT_BACK] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self._num_points_laser_right,),
                dtype=np.float64
            )
            '''
            self.observation_space[KEY_SEG_AREA] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float64
            )
            
            self.observation_space[KEY_SEG_CM] = spaces.Box(
                low=0,
                high=1,
                shape=(2,),
                dtype=np.float64
            )
            self.observation_space[KEY_SEG_POINTS_LEFT] = spaces.Box(
                low=0,
                high=1,
                shape=(self._num_points_seg, 2),
                dtype=np.float64
            )
            self.observation_space[KEY_SEG_POINTS_RIGHT] = spaces.Box(
                low=0,
                high=1,
                shape=(self._num_points_seg, 2),
                dtype=np.float64
            )
            '''

    def _get_obs_env(self):
        obs = super()._get_obs_env()
        obs[KEY_VEL] = self._velocity

        obs[KEY_LASER] = np.full(self._num_points_laser, MAX_DIST_LASER, dtype=np.float64)
        obs[KEY_LASER_RIGHT_FRONT] = np.full(self._num_points_laser_right, MAX_DIST_LASER, dtype=np.float64)
        obs[KEY_LASER_RIGHT] = np.full(self._num_points_laser_right, MAX_DIST_LASER, dtype=np.float64)
        obs[KEY_LASER_RIGHT_BACK] = np.full(self._num_points_laser_right, MAX_DIST_LASER, dtype=np.float64)

        '''
        if not self._passing:
            obs[KEY_LASER] = np.full(self._num_points_laser, MAX_DIST_LASER, dtype=np.float64)
        else:
            obs[KEY_LASER] = self._lidar.get_points_zone(num_points=self._num_points_laser, zone=configcarla.FRONT)
        
        if self._seg:
            area, cm, left, right = self._camera.get_seg_data(num_points=self._num_points_seg)

            obs[KEY_LASER_RIGHT] = self._lidar.get_points_zone(num_points=self._num_points_laser_right, zone=configcarla.RIGHT)
            obs[KEY_LASER_RIGHT_FRONT] = self._lidar.get_points_zone(num_points=self._num_points_laser_right, zone=configcarla.RIGHT_FRONT)
            obs[KEY_LASER_RIGHT_BACK] = self._lidar.get_points_zone(num_points=self._num_points_laser_right, zone=configcarla.RIGHT_BACK)
        else:
            area = SIZE_CAMERA * SIZE_CAMERA
            #cm = np.array([0, 0], dtype=np.int32)
            #left = np.zeros((self._num_points_seg, 2), dtype=np.int32)
            #right = np.zeros((self._num_points_seg, 2), dtype=np.int32)

            
            obs[KEY_LASER_RIGHT_FRONT] = np.full(self._num_points_laser_right, MAX_DIST_LASER, dtype=np.float64)
            obs[KEY_LASER_RIGHT] = np.full(self._num_points_laser_right, MAX_DIST_LASER, dtype=np.float64)
            obs[KEY_LASER_RIGHT_BACK] = np.full(self._num_points_laser_right, MAX_DIST_LASER, dtype=np.float64)
            
        #obs[KEY_SEG_AREA] = np.array([area], dtype=np.int32)
        #obs[KEY_SEG_CM] = cm
        #obs[KEY_SEG_POINTS_LEFT] = left
        #obs[KEY_SEG_POINTS_RIGHT] = right
        '''

        return obs
    
    def _get_info(self):
        info = super()._get_info()
        info[KEY_LASER] = self._dist_laser
        info[KEY_LASER_RIGHT] = 0
        info[KEY_LASER_RIGHT_FRONT] = 0
        info[KEY_LASER_RIGHT_BACK] = 0
        return info

    def _create_actions(self):
        self._max_steer = 0.2
        self.action_space = spaces.Box(low=np.array([0.0, -self._max_steer]),
                                       high=np.array([1.0, self._max_steer]),
                                       shape=(2,), dtype=np.float64)
        
    def _get_control(self, action:np.ndarray):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self._throttle, self._steer = action
        
        control = carla.VehicleControl()
        control.steer = self._steer 
        control.throttle = self._throttle

        return control
    
    def _calculate_reward(self, error:str):
        if error == None:
            # Deviation normalization
            r_dev = (MAX_DEV - abs(np.clip(self._dev, -MAX_DEV, MAX_DEV))) / MAX_DEV
            
            # Steer conversion
            limit_steer = 0.14
            if abs(self._steer) > limit_steer:
                r_steer = 0
            else:
                r_steer = (limit_steer - abs(self._steer)) / limit_steer
        
            # Throttle conversion
            limit_throttle = 0.6
            if self._throttle >= limit_throttle:
                r_throttle = 0
            elif self._velocity > self._max_vel:
                r_throttle = (limit_throttle - self._throttle) / limit_throttle
            else:
                r_throttle = self._throttle / limit_throttle

            assert r_dev >= 0 and r_dev <= 1 and r_throttle >= 0 and r_throttle <= 1 and r_steer >= 0 and r_throttle <=1, "fallo en las r"

            # Set weights
            if r_steer == 0 and r_throttle == 0:
                w_dev = 0.1
                w_throttle = 0.45
                w_steer = 0.45
            elif r_steer == 0:
                w_dev = 0.1
                w_throttle = 0.1
                w_steer = 0.8
            elif r_throttle == 0:
                w_dev = 0.1
                w_throttle = 0.8
                w_steer = 0.1
            elif self._velocity > self._max_vel:
                w_dev = 0.1
                w_throttle = 0.65
                w_steer = 0.25
            elif self._throttle < 0.5: 
                w_dev = 0.65 
                w_throttle = 0.15 
                w_steer = 0.2 # Lower accelerations, penalize large turns less
            else: # [0.5, 0.6) throttle
                w_dev = 0.6
                w_throttle = 0.05
                w_steer = 0.35

            assert w_dev + w_steer + w_throttle == 1, "Fallo en los pesos w"
            reward = w_dev * r_dev + w_throttle * r_throttle + w_steer * r_steer 
            assert reward >= 0 and reward <= 1, "Fallo en func recompensa"
        else:
            if "Distance" in error:
                reward = -60
            else:
                reward = -40

        return reward