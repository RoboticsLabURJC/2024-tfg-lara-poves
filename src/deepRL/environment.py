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
from colorama import Fore, Style

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, src_path)

from configcarla import SIZE_CAMERA, PATH
import configcarla

MAX_DEV = 100
MAX_DIST_LASER = 10
MIN_DIST_LASER = 4
FREC_PASSING = 3

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
                 seed:int=None, num_cir:int=0, start_passing=1000):
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
        transform = carla.Transform(carla.Location(x=0.5, z=1.7292))
        self._sensors = configcarla.Vehicle_sensors(vehicle=self.ego_vehicle, world=self._world,
                                                    screen=self._screen)
        self._camera = self._sensors.add_camera_rgb(transform=transform, seg=False, lane=True,
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
                                                  transform=lidar_transform, scale=22, time_show=False)
            self._lidar.set_z_threshold(1.7)

            t = carla.Transform(
                location=self._loc.location,
                rotation=self._loc.rotation
            )
            
            if self._id == 0 or self._id == 5:
                t.location.x -= 5
                t.location.y -= 4
                t.rotation.yaw -= 6
            elif self._id == 1:
                t.location.y += 7
            else:
                t.location.y += 7
                t.location.x -= 5

            # Front vehicle
            self._front_vehicle = configcarla.add_one_vehicle(world=self._world, vehicle_type='vehicle.carlamotors.carlacola',
                                                              transform=t)

            # Set traffic manager
            self._tm = configcarla.traffic_manager(client=self._client, vehicles=[self._front_vehicle], 
                                                   port=7788, speed=random.randint(40, 50))
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
    '''   
    def _func_is_passing_ep(self):
        # Front vehicle
        self._vel_front += carla.Vector3D(self._front_vehicle.get_velocity()).length()
        loc = self._front_vehicle.get_location()

        
        # Module front vehicle velocity
        if self._count_steps % 5 == 0:
            if self._vel_percentage < 95 and self._vel_front > self._target_vel and abs(self._vel_front - self._target_vel) > 1:
                self._vel_percentage += 1
                print("reduzco vel, vel front:", self._vel_front, "target:", self._target_vel, "%", self._vel_percentage)
            elif self._target_vel > self._vel_front and abs(self._target_vel - self._vel_front) > 0.5: 
                self._vel_percentage -= 1
                print("aumento vel, vel front:", self._vel_front, "target:", self._target_vel, "%", self._vel_percentage)

            road_id = self._map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving).road_id
            print("road id:", road_id, "id:", self._id)
            if self._vel_percentage > 85 and ((road_id == 35 and self._id == 0) or 
                ((road_id == 1162 or road_id == 23) and self._id == 2) or (1073 == road_id and self._id == 1)):
                print("retoco")
                self._vel_percentage = 85

            # Apply velocity
            self._tm.global_percentage_speed_difference(int(np.clip(self._vel_percentage, 0, 95)))

            # Reset mean vel front
            self._vel_front = 0
        
        # Check distance to front car
        self._dist_laser = self._lidar.get_min_center()
        assert np.isnan(self._dist_laser) or self._dist_laser > MIN_DIST_LASER, "Distance exceeded: too close to the front car"
    '''
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
            # Update data
            if self._train:
                self._world.tick()
            self._sensors.update_data()

            # Get deviation and velocity
            dev_prev = self._dev
            self._dev = self._camera.get_deviation()
            self._velocity = carla.Vector3D(self.ego_vehicle.get_velocity()).length()
            self._mean_vel += self._velocity
            loc = self.ego_vehicle.get_location()

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

            print(error)
            print("No termino", self._count_ep, "steps:", self._count, "id:", self._id,
                  "dev:", self._dev, "is_passing:", self._is_passing_ep, "dist:", self._dist_laser)

        # Check if a key has been pressed
        if self._human:
            self.render()

        reward = self._calculate_reward(error)
        self._total_reward += reward
        self._count += 1

        if finish_ep:
            print("Termino:", self._count_ep, "steps:", self._count, "id:", self._id,
                  "mean vel", self._mean_vel / self._count)

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
            self._target_vel = random.uniform(10, 15)

            '''
            if self.model != None:
                self._is_passing_ep = self._exploration_rate <= 0.65

                # Set target vel
                if self._exploration_rate > 0.4:
                    self._target_vel = random.uniform(0.9, 1.5)
                elif self._exploration_rate > 0.3:
                    self._target_vel = random.uniform(1.5, 2.5)
                elif self._exploration_rate > 0:
                    self._target_vel = random.uniform(1.5, 4)
                else:
                    self._target_vel = random.uniform(2, 6)
                self._target_vel += 2
            else:
            '''
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
    def __init__(self, human:bool, train:bool, alg:str=None, port:int=2000, num_cir:int=0, 
                 fixed_delta_seconds:float=0.0, normalize:bool=False, seed:int=None):
        if train:
            num_cir = 0
        config = CIRCUIT_CONFIG.get(num_cir, CIRCUIT_CONFIG[0])

        super().__init__(human=human, train=train, alg=alg, port=port, seed=seed, num_cir=num_cir, config=config,
                         normalize=normalize, fixed_delta_seconds=fixed_delta_seconds)
        
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
'''  
class CarlaObstacleDiscrete(CarlaBase):
    def __init__(self, human:bool, train:bool, alg:str=None, port:int=2000, num_cir:int=0,
                 fixed_delta_seconds:float=0.0, normalize:bool=False, seed:int=None, total_steps:int=0):
        if train:
            num_cir = 0
        config = CIRCUIT_CONFIG.get(num_cir, CIRCUIT_CONFIG[0])    

        super().__init__(human=human, train=train, alg=alg, port=port, seed=seed, passing=True, total_steps=total_steps,
                         normalize=normalize, fixed_delta_seconds=fixed_delta_seconds, num_cir=num_cir, config=config)
        
        self._max_vel = 15

        # Add laser front distance to observations
        new_space = spaces.Box(
            low=MIN_DIST_LASER - 1.0,
            high=MAX_DIST_LASER,
            shape=(1,),
            dtype=np.float64
        )

        if not normalize:
            self.observation_space["laser"] = new_space
        else:
            self._obs_norm["laser"] = new_space
            self.observation_space["laser"] =  spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float64
            )
    
    def _get_info(self):
        info = super()._get_info()
        info["dist"] = self._dist_laser
        return info

    def _get_obs_env(self):
        obs = super()._get_obs_env()

        if not self._is_passing_ep or np.isnan(self._dist_laser):
            obs["laser"] = MAX_DIST_LASER
        else:
            obs["laser"] = self._dist_laser

        return obs
    
    def set_model(self, model):
        self.model = model
    
    def _get_control(self, action:np.ndarray):
        throttle, steer = self.action_to_control[int(action)]
        control = carla.VehicleControl()
        control.steer = steer 
        control.throttle = throttle
        
        return control
        
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
            20: (0.1, -0.18),
            21: (0.4, 0.0),
            22: (0.3, 0.0),
            23: (0.2, 0.0),
            24: (0.1, 0.0),
            25: (0.3, 0.01),
            26: (0.3, -0.01),
            27: (0.2, 0.02),
            28: (0.2, -0.02)
        }

        self.action_space = spaces.Discrete(len(self.action_to_control))
        
    def _calculate_reward(self, error:str):
        if error == None:
            # Clip and normalize deviation and velocity
            r_dev = (MAX_DEV - abs(np.clip(self._dev, -MAX_DEV, MAX_DEV))) / MAX_DEV
            r_vel = np.clip(self._velocity, 0.0, self._max_vel) / self._max_vel

            if not self._is_passing_ep or np.isnan(self._dist_laser) or self._dist_laser >= MAX_DIST_LASER_ALLOWED:
                reward = 0.8 * r_dev + 0.2 * r_vel
            else:
                r_laser = (np.clip(self._dist_laser, MIN_DIST_LASER, MAX_DIST_LASER_ALLOWED) - MIN_DIST_LASER)
                r_laser /= (MAX_DIST_LASER_ALLOWED - MIN_DIST_LASER)
                reward = 0.4 * r_dev + 0.1 * r_vel + 0.5 * r_laser
        else:
            if "Distance" in error:
                reward = -60
            else:
                reward = -30

        return reward
   '''

class CarlaLaneContinuous(CarlaBase):
    def __init__(self, human:bool, train:bool, alg:str=None, port:int=200, num_cir:int=0,
                 fixed_delta_seconds:float=0.0, normalize:bool=False, seed:int=None):
        if train and human:
            human = False
            print("Warning: Can't activate human mode during training")

        if train:
            num_cir = 0
        config = CIRCUIT_CONFIG.get(num_cir, [])

        super().__init__(human=human, train=train, alg=alg, port=port, seed=seed, num_points=10,
                         normalize=normalize, fixed_delta_seconds=fixed_delta_seconds,
                         num_cir=num_cir, config=config)
        
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
        # Add brake
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
            if self._velocity > self._max_vel:
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
    def __init__(self, human:bool, train:bool, alg:str=None, port:int=200, num_cir:int=0,
                 fixed_delta_seconds:float=0.0, normalize:bool=False, seed:int=None):
        if train and human:
            human = False
            print("Warning: Can't activate human mode during training")

        if train:
            num_cir = 0
        config = CIRCUIT_CONFIG.get(num_cir, [])

        super().__init__(human=human, train=train, alg=alg, port=port, seed=seed, num_points=10,
                         normalize=normalize, fixed_delta_seconds=fixed_delta_seconds,
                         num_cir=num_cir, config=config, passing=True, start_passing=1250)
        
        self._max_vel = 30

        # Add velocity to observations
        new_space = spaces.Box(
            low=0.0,
            high=60.0,
            shape=(1,),
            dtype=np.float64
        )

        # Add laser front distance to observations
        new_space = spaces.Box(
            low=MIN_DIST_LASER - 1.0,
            high=MAX_DIST_LASER,
            shape=(1,),
            dtype=np.float64
        )

        if not normalize:
            self.observation_space["velocity"] = new_space
            self.observation_space["laser"] = new_space
        else:
            self._obs_norm["velocity"] = new_space
            self.observation_space["velocity"] =  spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float64
            )

            self._obs_norm["laser"] = new_space
            self.observation_space["laser"] =  spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float64
            )

    def _get_obs_env(self):
        obs = super()._get_obs_env()
        obs["velocity"] = self._velocity

        if not self._is_passing_ep or np.isnan(self._dist_laser):
            obs["laser"] = MAX_DIST_LASER
        else:
            obs["laser"] = self._dist_laser

        return obs

    def _create_actions(self):
        # Add brake
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
            if self._velocity > self._max_vel:
                r_throttle = -5/3 * self._throttle + 1
            else:
                r_throttle = 5/3 * self._throttle

            # Laser conversion
            if self._is_passing_ep and not np.isnan(self._dist_laser):
                r_laser = np.clip(self._dist_laser, MIN_DIST_LASER, MAX_DIST_LASER) - MIN_DIST_LASER
                r_laser /= (MAX_DIST_LASER - MIN_DIST_LASER)
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
            elif r_laser != 0: # See vehicle front
                w_dev = 0.35
                w_throttle = 0
                w_steer = 0.1
                w_laser = 0.55
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