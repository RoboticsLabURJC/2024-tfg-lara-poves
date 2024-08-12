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

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, src_path)

from configcarla import SIZE_CAMERA, PATH
import configcarla

MAX_DEV = 100

class CarlaBase(gym.Env, ABC):
    def __init__(self, human:bool, train:bool, alg:str=None, port:int=2000,
                 fixed_delta_seconds:float=0.0, normalize:bool=False, seed:int=None):
        self._first = True
        self._dev = 0
        self._vel = 0
        self._steer = 0
        self._speed = 0
        self._count_ep = 0
        self._total_reward = 0
        self._count = 0
        self._jump = False
        self._human = human
        self._max_vel = 5.0

        # CSV file
        self._train = train
        if self._train:
            assert alg != None, "Algorithms are required for training"
            dir_csv = PATH + '2024-tfg-lara-poves/src/deepRL/csv/train/' + self.__class__.__name__ + '/'
            if not os.path.exists(dir_csv):
                os.makedirs(dir_csv)
            files = os.listdir(dir_csv)
            num_files = len(files) + 1
            self._file_csv = open(dir_csv + alg + '_train_data_' + str(num_files) + '.csv',
                                  mode='w', newline='')
            self._writer_csv = csv.writer(self._file_csv)
            self._writer_csv.writerow(["Episode", "Reward", "Num_steps", "Finish", "Deviation",
                                       "Exploration_rate"])
        
        # States
        self._num_points_line = 5
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
                    shape=(self._num_points_line, 2),
                    dtype=np.int32
                ),

                "right_points": spaces.Box(
                    low=0,
                    high=SIZE_CAMERA - 1,
                    shape=(self._num_points_line, 2),
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
                    shape=(self._num_points_line, 2),
                    dtype=np.float64
                ),

                "right_points": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self._num_points_line, 2),
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

        # Init locations
        self._town = 'Town05'
        z = 0.1
        self._init_locations = [
            carla.Transform(carla.Location(x=50.0, y=-145.7, z=z)),
            carla.Transform(carla.Location(x=120.0, y=-137, z=z), carla.Rotation(yaw=22)),
            carla.Transform(carla.Location(x=43.0, y=141.5, z=z), carla.Rotation(yaw=3)),
            carla.Transform(carla.Location(x=111.0, y=135.5, z=z), carla.Rotation(yaw=-16))
        ]

        # Pygame window
        if self._human:
            self._screen = configcarla.setup_pygame(size=(SIZE_CAMERA * 2, SIZE_CAMERA), 
                                                    name='Follow lane: ' + self.__class__.__name__)
            self._init_driver = (SIZE_CAMERA, 0)
        else:
            self._screen = None
            self._init_driver = None

        # Init simulation
        if self._train:
            assert fixed_delta_seconds > 0.0, "In synchronous mode fidex_delta_seconds can't be 0.0"
        self._world, _ = configcarla.setup_carla(name_world=self._town, port=port, syn=self._train, 
                                                 fixed_delta_seconds=fixed_delta_seconds)

    def _swap_ego_vehicle(self):
        if self._train:
            self._index_loc = random.randint(0, len(self._init_locations) - 1)
        else:
            self._index_loc = 0

        self.ego_vehicle = configcarla.add_one_vehicle(world=self._world, ego_vehicle=True,
                                                        vehicle_type='vehicle.lincoln.mkz_2020',
                                                        transform=self._init_locations[self._index_loc])
        transform = carla.Transform(carla.Location(z=1.4, x=1.75), carla.Rotation(roll=90.0))
        self._sensors = configcarla.Vehicle_sensors(vehicle=self.ego_vehicle, world=self._world,
                                                    screen=self._screen)
        self._camera = self._sensors.add_camera_rgb(transform=transform, seg=False, lane=True,
                                                    canvas_seg=False, size_rect=(SIZE_CAMERA, SIZE_CAMERA),
                                                    init_extra=self._init_driver, text='Driver view')
        self._sensors.add_collision() # Raise an exception if the vehicle crashes

        if self._human:
            world_transform = carla.Transform(carla.Location(z=2.5, x=-4.75), carla.Rotation(roll=90.0))
            self._sensors.add_camera_rgb(transform=world_transform, size_rect=(SIZE_CAMERA, SIZE_CAMERA),
                                         init=(0, 0), text='World view')

    def _get_obs_env(self):
        left_points, right_points = self._camera.get_lane_points()
        obs = {
            "cm": self._camera.get_lane_cm(), 
            "left_points": left_points,
            "right_points": right_points, 
            "area": np.array([self._camera.get_lane_area()], dtype=np.int32),
            "deviation": np.array([self._dev], dtype=np.int32)
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
        info = {"deviation": self._dev, "speed": self._speed}
        return info
    
    def step(self, action:np.ndarray):
        # Exec action
        control = self._get_control(action)
        self.ego_vehicle.apply_control(control) 

        terminated = False
        finish_ep = False

        try:
            # Update data
            self._world.tick()
            self._sensors.update_data()

            reward = self._calculate_reward()

            t = self.ego_vehicle.get_transform()
            if self._index_loc >= 2:
                if not self._jump and t.location.y < 13:
                    t.location.y = -20
                    self.ego_vehicle.set_transform(t)
                    self._jump = True
                elif t.location.y < -143: 
                    terminated = True
                    finish_ep = True
            else:
                if not self._jump and t.location.y > -30:
                    t.location.y = 15
                    self.ego_vehicle.set_transform(t)
                    self._jump = True
                elif t.location.x < 50:
                    terminated = True
                    finish_ep = True
            
        except AssertionError:
            terminated = True
            reward = -20

            t = self.ego_vehicle.get_transform()
            if self._index_loc >= 2:
                base = -143
                t = t.location.y
            else:
                base = 50
                t = t.location.x

            print("base", base, "dev:", self._dev, "t:", t)

        # Check if a key has been pressed
        if self._human:
            self.render()

        self._total_reward += reward
        self._count += 1

        if terminated and self._train:
            if self.model != None:
                exploration_rate = self.model.exploration_rate
            else:
                exploration_rate = -1.0 # No register

            self._count_ep += 1
            self._writer_csv.writerow([self._count_ep, self._total_reward, self._count,
                                       finish_ep, self._dev, exploration_rate])
        
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
        self._jump = False

        if self._first:
            self._first = False
        else:
            self._sensors.destroy()

        self._swap_ego_vehicle()

        while True:
            try:
                self._world.tick()
                self._sensors.update_data()

                if self._camera.data != None:
                    break 

                # Reset info
                self._dev = self._camera.get_deviation()
                self._speed = 0
            except AssertionError:
                pass
        
        return self._get_obs(), self._get_info()

    def close(self):
        self._sensors.destroy()

        if self._train:
            self._file_csv.close()

        pygame.quit()

    @abstractmethod
    def _get_control(self, action:np.ndarray):
        pass

    @abstractmethod
    def _create_actions(self):
        pass

    @abstractmethod
    def _calculate_reward(self):
        pass

class CarlaLaneDiscrete(CarlaBase):
    def __init__(self, human:bool, train:bool, alg:str=None, port:int=2000,
                 fixed_delta_seconds:float=0.0, normalize:bool=False, seed:int=None):
        super().__init__(human=human, train=train, alg=alg, port=port, seed=seed,
                         normalize=normalize, fixed_delta_seconds=fixed_delta_seconds)

    def _create_actions(self):
        self._max_steer = 0.18
        self.action_to_control = {
            0: (0.5, 0.0),
            1: (0.5, 0.01), 
            2: (0.5, -0.01),
            3: (0.4, 0.02),
            4: (0.4, -0.02),
            5: (0.4, 0.04),
            6: (0.4, -0.04),
            7: (0.3, 0.06),
            8: (0.3, -0.06),
            9: (0.3, 0.08),
            10: (0.3, -0.08),
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
        throttle, self._steer = self.action_to_control[int(action)] # Use steer in reward
        control = carla.VehicleControl()
        control.steer = self._steer 
        control.throttle = throttle
        
        return control
    
    def _calculate_reward(self):
        # Get deviation
        self._dev = self._camera.get_deviation()
        dev = np.clip(self._dev, -MAX_DEV, MAX_DEV)

        # Get velocity
        speed = self.ego_vehicle.get_velocity()
        self._speed = carla.Vector3D(speed).length()
        vel = np.clip(self._speed, 0.0, self._max_vel) # Reaches a speed of 5m/s after 5 seconds

        # Calculate reward
        reward = 0.8 * (MAX_DEV - abs(dev)) / MAX_DEV + 0.2 * vel / self._max_vel
        return reward

class CarlaLaneContinuousSimple(CarlaBase):
    def __init__(self, human:bool, train:bool, alg:str=None, port:int=2000,
                 fixed_delta_seconds:float=0.0, normalize:bool=False, seed:int=None):
        if train and human:
            human = False
            print("Warning: Can't activate human mode during training")

        super().__init__(human=human, train=train, alg=alg, port=port, seed=seed,
                         normalize=normalize, fixed_delta_seconds=fixed_delta_seconds)
        
    def _create_actions(self):
        self.action_space = spaces.Box(low=np.array([0.1, -0.3]), high=np.array([0.5, 0.3]),
                                       shape=(2,), dtype=np.float64)
        
    def _get_control(self, action:np.ndarray):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        throttle, steer = action

        control = carla.VehicleControl()
        control.steer = steer 
        control.throttle = throttle

        return control
    
    def _calculate_reward(self):
        # Get deviation
        self._dev = self._camera.get_deviation()
        dev = np.clip(self._dev, -MAX_DEV, MAX_DEV)

        # Get velocity
        speed = self.ego_vehicle.get_velocity()
        self._speed = carla.Vector3D(speed).length()
        vel = np.clip(self._speed, 0.0, self._max_vel)

        # Calculate reward
        reward = 0.8 * (MAX_DEV - abs(dev)) / MAX_DEV + 0.2 * vel / self._max_vel
        return reward

class CarlaLaneContinuousComplex(CarlaBase):
    def __init__(self, human:bool, train:bool, alg:str=None, port:int=2000,
                 fixed_delta_seconds:float=0.0, normalize:bool=False, seed:int=None):
        if train and human:
            human = False
            print("Warning: Can't activate human mode during training")

        super().__init__(human=human, train=train, alg=alg, port=port, seed=seed,
                         normalize=normalize, fixed_delta_seconds=fixed_delta_seconds)
        
    def _create_actions(self):
        # Add brake
        self.action_space = spaces.Box(low=np.array([0.1, -0.3, 0.0]), high=np.array([0.5, 0.3, 1.0]),
                                       shape=(3,), dtype=np.float64)
        
    def _get_control(self, action:np.ndarray):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        throttle, steer, brake = action

        control = carla.VehicleControl()
        control.steer = steer 
        control.throttle = throttle
        control.brake = brake

        return control
    
    def _calculate_reward(self):
        # Get deviation
        self._dev = self._camera.get_deviation()
        dev = np.clip(self._dev, -MAX_DEV, MAX_DEV)

        # Get velocity
        speed = self.ego_vehicle.get_velocity()
        self._speed = carla.Vector3D(speed).length()
        vel = np.clip(self._speed, 0.0, self._max_vel)

        # Calculate reward
        reward = 0.7 * (MAX_DEV - abs(dev)) / MAX_DEV + 0.3 * vel / self._max_vel
        return reward
    
class CarlaObstacle(CarlaBase):
    def __init__(self, human:bool, train:bool, alg:str=None, port:int=2000,
                 fixed_delta_seconds:float=0.0, normalize:bool=False, seed:int=None):
        if train and human:
            human = False
            print("Warning: Can't activate human mode during training")

        super().__init__(human=human, train=train, alg=alg, port=port, seed=seed,
                         normalize=normalize, fixed_delta_seconds=fixed_delta_seconds)
        
        new_space = spaces.Box(
            low=0.0,
            high=100.0,
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
                shape=(3,),
                dtype=np.float64
            )
    
    def _get_info(self):
        info = super()._get_info()
        info["dist"] = self._dist
        return info

    def _get_obs_env(self):
        obs = super()._get_obs_env()
        self._dist = 20.0
        obs["laser"] = self._dist
        return obs
        
    def _create_actions(self):
        # Add brake
        self.action_space = spaces.Box(low=np.array([0.1, -0.3, 0.0]), high=np.array([0.5, 0.3, 1.0]),
                                       shape=(3,), dtype=np.float64)
        
    def _get_control(self, action:np.ndarray):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        throttle, steer, brake = action

        control = carla.VehicleControl()
        control.steer = steer 
        control.throttle = throttle
        control.brake = brake

        return control
    
    def _calculate_reward(self):
        # Get deviation
        self._dev = self._camera.get_deviation()
        dev = np.clip(self._dev, -MAX_DEV, MAX_DEV)

        # Get velocity
        speed = self.ego_vehicle.get_velocity()
        self._speed = carla.Vector3D(speed).length()
        vel = np.clip(self._speed, 0.0, self._max_vel)

        # Calculate reward
        reward = 0.8 * (MAX_DEV - abs(dev)) / MAX_DEV + 0.2 * vel / self._max_vel
        return reward
