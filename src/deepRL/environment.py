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
import time

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, src_path)

from configcarla import SIZE_CAMERA, PATH
import configcarla

MAX_DEV = 100
MAX_DIST_LASER = 15
MIN_DIST_LASER = 4
FREC_PASSING = 3
CHANGE_VEL_RATE = 5

CIRCUIT_CONFIG = {
    0: [
        {"id": 0, "town": "Town04", "location": carla.Transform(carla.Location(x=352.65, y=-351, z=0.1), carla.Rotation(yaw=-137))},
        {"id": 1, "town": "Town04", "location": carla.Transform(carla.Location(x=-8.76, y=60.8, z=0.1), carla.Rotation(yaw=89.7))},
        {"id": 2, "town": "Town04", "location": carla.Transform(carla.Location(x=-25.0, y=-252, z=0.1), carla.Rotation(yaw=125.0))}
    ],
    1: [
        {"id": 5, "town": "Town04", "location": carla.Transform(carla.Location(x=352.65, y=-351, z=0.1), carla.Rotation(yaw=-137))}
    ],
    2: [
        {"id": 3, "town": "Town04", "location": carla.Transform(carla.Location(x=13.5, y=310, z=0.1), carla.Rotation(yaw=-48))}
    ],
    3: [
        {"id": 4, "town": "Town03", "location": carla.Transform(carla.Location(x=114, y=207.3, z=1.7))}
    ],
    4: [
        {"id": 6, "town": "Town06", "location": carla.Transform(carla.Location(x=457.83, y=244.7, z=0.1))}
    ]
}

class CarlaBase(gym.Env, ABC):
    def __init__(self, human:bool, train:bool, config:list, alg:str=None, port:int=2000, num_points:int=5,
                 passing:bool=False, fixed_delta_seconds:float=0.0, normalize:bool=False,
                 seed:int=None, num_cir:int=0, start_passing=1000, port_tm:int=1111, lane_network:bool=False):
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
        self._vel_percentage = 80
        self._start_passing = start_passing
        self._lane_network = lane_network

        if passing and num_cir > 1:
            assert True, "No passing mode available for circuit " + str(num_cir) 

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
            self._writer_csv_train.writerow(["Episode", "Reward", "Num_steps", "Finish", "Deviation",
                                             "Exploration_rate", "Distance", "Mean veal"])

            # Action file
            dir_csv = PATH + '2024-tfg-lara-poves/src/deepRL/csv/actions/' + self.__class__.__name__ + '/'
            if not os.path.exists(dir_csv):
                os.makedirs(dir_csv)
            files = os.listdir(dir_csv)
            num_files = len(files) + 1
            self._actions_csv = open(dir_csv + alg + '_train_actions_' + str(num_files) + '.csv',
                                     mode='w', newline='')
            self._writer_csv_actions = csv.writer(self._actions_csv)
            self._writer_csv_actions.writerow(["Throttle", "Steer", "Brake"])
        
        # States/Observations
        self._num_points_lane = num_points
        self.observation_space = spaces.Dict(
            spaces={
                "cm": spaces.Box(
                    low=0,
                    high=SIZE_CAMERA - 1,
                    shape=(2,),
                    dtype=np.int32
                ),
                
                "left_points": spaces.Box(
                    low=0,
                    high=SIZE_CAMERA - 1,
                    shape=(self._num_points_lane, 2),
                    dtype=np.int32
                ),

                "right_points": spaces.Box(
                    low=0,
                    high=SIZE_CAMERA - 1,
                    shape=(self._num_points_lane, 2),
                    dtype=np.int32
                ),

                "area": spaces.Box(
                    low=0,
                    high=SIZE_CAMERA * SIZE_CAMERA,
                    shape=(1,),
                    dtype=np.int32
                ),

                "deviation": spaces.Box(
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
                "cm": spaces.Box(
                    low=0,
                    high=1,
                    shape=(2,),
                    dtype=np.float64
                ),
                
                "left_points": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self._num_points_lane, 2),
                    dtype=np.float64
                ),

                "right_points": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self._num_points_lane, 2),
                    dtype=np.float64
                ),

                "area": spaces.Box(
                    low=0,
                    high=1,
                    shape=(1,),
                    dtype=np.float64
                ),

                "deviation": spaces.Box(
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

            self._screen = configcarla.setup_pygame(size=(SIZE_CAMERA * mult, SIZE_CAMERA), 
                                                    name='Follow lane: ' + self.__class__.__name__)
            self._init_driver = (SIZE_CAMERA, 0)
            self._init_laser = (SIZE_CAMERA * 2, 0)
        else:
            self._screen = None
            self._init_driver = None
            self._init_laser = None

        # Init simulation
        if self._train:
            assert fixed_delta_seconds > 0.0, "In synchronous mode fidex_delta_seconds can't be 0.0"

        self._fixed_delta_seconds = fixed_delta_seconds        
        self._town_locations = [(entry["id"], entry["town"], entry["location"]) for entry in config]
        self._client = None
        self._port = port
        self._port_tm = port_tm

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

        self._camera = self._sensors.add_camera_rgb(transform=transform, seg=False, lane=True, lane_network=self._lane_network,
                                                    canvas_seg=False, size_rect=(SIZE_CAMERA, SIZE_CAMERA),
                                                    init_extra=self._init_driver, text='Driver view')
        self._sensors.add_collision() # Raise an exception if the vehicle crashes

        if self._human:
            world_transform = carla.Transform(carla.Location(z=2.5, x=-4.75))
            self._sensors.add_camera_rgb(transform=world_transform, size_rect=(SIZE_CAMERA, SIZE_CAMERA),
                                         init=(0, 0), text='World view')
            
        if self._is_passing_ep: 
            lidar_transform = carla.Transform(carla.Location(x=-0.5, z=1.8), carla.Rotation(yaw=90.0))
            self._lidar = self._sensors.add_lidar(init=self._init_laser,size_rect=(SIZE_CAMERA, SIZE_CAMERA),
                                                  transform=lidar_transform, scale=17, time_show=False, show_stats=False,
                                                  train=self._train, max_dist=MAX_DIST_LASER, front_angle=150) # 50º each part
            self._lidar.set_z_threshold(1.7)

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
            self._front_vehicle = configcarla.add_one_vehicle(world=self._world, transform=t,
                                                              vehicle_type='vehicle.carlamotors.carlacola')

            # Set traffic manager
            self._tm = configcarla.traffic_manager(client=self._client, vehicles=[self._front_vehicle], port=self._port_tm)
            self._tm.ignore_lights_percentage(self._front_vehicle, 100)

    def _get_obs_env(self):
        left_points, right_points = self._camera.get_lane_points(num_points=self._num_points_lane)
        obs = {
            "cm": self._camera.get_lane_cm(), 
            "left_points": left_points,
            "right_points": right_points, 
            "area": np.array([self._camera.get_lane_area()], dtype=np.int32),
            "deviation": np.array([self._dev], dtype=np.int32),
        }

        return obs

    def _get_obs(self):
        obs = self._get_obs_env()

        if self.normalize:
            for key, sub_space in self._obs_norm.spaces.items():
                obs[key] = (obs[key] - sub_space.low) / (sub_space.high - sub_space.low)
                obs[key] = obs[key].astype(np.float64)

        return obs
    
    def _get_info(self):
        info = {"deviation": self._dev, "velocity": self._velocity}
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

        if self._is_passing_ep and (self._count == 0 or time.time() - self._start_time >= CHANGE_VEL_RATE):
            self._start_time = time.time()

            prob = random.random()
            if prob < 0.35: # 35% probability
                self._target_vel = random.uniform(0, 5)
            else: # 65% probability 
                self._target_vel = random.uniform(5, 10)
            self._tm.set_desired_speed(self._front_vehicle, self._target_vel * 3.6) # km/h
            print(self._target_vel)

        try:
            # Tick
            if self._train:
                self._world.tick()

            # Get velocity and location
            self._velocity = carla.Vector3D(self.ego_vehicle.get_velocity()).length()
            self._mean_vel += self._velocity
            loc = self.ego_vehicle.get_location()

            # Update data
            if self._is_passing_ep:
                self._sensors.update_data(vel_ego=self._velocity, vel_front=self._target_vel, front_laser=True)
            else:
                self._sensors.update_data(vel_ego=self._velocity)

            # Get deviation
            dev_prev = self._dev
            self._dev = self._camera.get_deviation()

            # Obstacle trainings
            if self._is_passing_ep:
                # Check distance to front car
                self._dist_laser = self._lidar.get_min_center()
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
            else:
                finish_ep = abs(loc.x - 165) <= 3 and abs(loc.y + 208) <= 3
            terminated = finish_ep
            
        except AssertionError as e:
            terminated = True
            error = str(e)

            #print(error)
            #print("No termino", self._count_ep, "steps:", self._count, "id:", self._id,
             #     "dev:", self._dev, "is_passing:", self._is_passing_ep, "dist:", self._dist_laser)

        # Check if a key has been pressed
        if self._human:
            self.render()

        reward = self._calculate_reward(error)
        self._total_reward += reward
        self._count += 1

        #if finish_ep:
            #print("Termino:", self._count_ep, "steps:", self._count, "id:", self._id,
             #     "mean vel", self._mean_vel / self._count, "passing:", self._is_passing_ep)

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
        self._dev = 0

        if self._first:
            self._first = False
        else:
            self._sensors.destroy()

            if self._front_vehicle != None and self._is_passing_ep:
                self._front_vehicle.destroy()

        if self._passing:
            self._vel_front = 0
            self._is_passing_ep = self._count_ep > self._start_passing 
            self._target_vel = 0
        else:
            self._is_passing_ep = False

        self._swap_ego_vehicle()

        while True:
            try:
                if self._train:
                    self._world.tick()
                self._sensors.update_data()

                if self._camera.data != None:
                    break 

                # Reset info
                self._dev = self._camera.get_deviation()
                self._velocity = 0
                if self._is_passing_ep:
                    self._dist_laser = self._lidar.get_min_center()

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
    def __init__(self, human:bool, train:bool, alg:str=None, port:int=2000, num_cir:int=0, retrain:bool=False,
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
    def __init__(self, human:bool, train:bool, alg:str=None, port:int=200, num_cir:int=0, retrain:bool=False,
                 fixed_delta_seconds:float=0.0, normalize:bool=False, seed:int=None, port_tm:int=1111, lane_network:bool=False):
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
            high=60.0,
            shape=(1,),
            dtype=np.float64
        )

        if not normalize:
            self.observation_space["velocity"] = new_space
        else:
            self._obs_norm["velocity"] = new_space
            self.observation_space["velocity"] =  spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float64
            )

    def _get_obs_env(self):
        obs = super()._get_obs_env()
        obs["velocity"] = self._velocity
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
            if abs(self._steer) > 0.14:
                r_steer = 0
            else:
                r_steer = -50/7 * abs(self._steer) + 1

            # Throttle conversion
            if self._throttle >= 0.6:
                r_throttle = 0
            elif self._velocity > self._max_vel:
                r_throttle = -5/3 * self._throttle + 1
            else:
                r_throttle = 5/3 * self._throttle

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
                 lane_network:bool=False):
        if train and human:
            human = False
            print("Warning: Can't activate human mode during training")

        if train:
            num_cir = 0
        else:
            retrain = True
        config = CIRCUIT_CONFIG.get(num_cir, [])

        super().__init__(human=human, train=train, alg=alg, port=port, seed=seed, num_points=10,
                         normalize=normalize, fixed_delta_seconds=fixed_delta_seconds, port_tm=port_tm,
                         num_cir=num_cir, config=config, passing=retrain, start_passing=-1,
                         lane_network=lane_network) 
        
        self._max_vel = 20

        # Add velocity to observations
        new_space_vel = spaces.Box(
            low=0.0,
            high=60.0,
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
            self.observation_space["velocity"] = new_space_vel
            self.observation_space["laser"] = new_space_laser
        else:
            self._obs_norm["velocity"] = new_space_vel
            self.observation_space["velocity"] =  spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float64
            )
            self._obs_norm["laser"] = new_space_laser
            self.observation_space["laser"] =  spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self._num_points_laser,),
                dtype=np.float64
            )

    def _get_obs_env(self):
        obs = super()._get_obs_env()
        obs["velocity"] = self._velocity

        if not self._is_passing_ep: # No passing
            obs["laser"] = np.full(self._num_points_laser, MAX_DIST_LASER, dtype=np.float64)
        else:
            obs["laser"] = self._lidar.get_points_front(self._num_points_laser)

        return obs
    
    def _get_info(self):
        info = super()._get_info()
        info["distance"] = self._dist_laser
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
            if abs(self._steer) > 0.14:
                r_steer = 0
            else:
                r_steer = -50/7 * abs(self._steer) + 1

            # Throttle conversion
            if self._throttle >= 0.6:
                r_throttle = 0
            elif self._velocity > self._max_vel:
                r_throttle = -5/3 * self._throttle + 1
            else:
                r_throttle = 5/3 * self._throttle

            # Laser conversion
            if self._is_passing_ep and not np.isnan(self._dist_laser):
                r_laser = np.clip(self._dist_laser, MIN_DIST_LASER, MAX_DIST_LASER) - MIN_DIST_LASER
                r_laser /= (MAX_DIST_LASER - MIN_DIST_LASER)

                if self._dist_laser <= 10:
                    r_throttle = -5/3 * self._throttle + 1
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
                #if self._dist_laser <= 6: # Front vehicle too close
                 #   w_dev = 0.1
                  #  w_throttle = 0.2
                  #  w_steer = 0
                   # w_laser = 0.7
                if self._dist_laser <= 10: # Medium distance
                    w_dev = 0.3
                    w_throttle = 0.2
                    w_steer = 0.1
                    w_laser = 0.4
                else:
                    w_dev = 0.45
                    w_throttle = 0.1
                    w_laser = 0.3
                    w_steer = 0.15
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