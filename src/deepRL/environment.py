import gymnasium as gym
from gymnasium import spaces
import os
import sys
import numpy as np
import carla
import random
import pygame

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, src_path)

from configcarla import SIZE_CAMERA
import configcarla

class CarlaDiscreteBasic(gym.Env):
    def __init__(self, human:bool, train:bool, port:int=2000, fixed_delta_seconds=0.0, 
                 num_points_line:int=5, range_vel:int=10, range_steer:int=50):
        self._dev = 0
        self._jump = False
        self._human = human

        # States
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
                    shape=(num_points_line, 2),
                    dtype=np.int32
                ),

                "right_points": spaces.Box(
                    low=0,
                    high=SIZE_CAMERA - 1,
                    shape=(num_points_line, 2),
                    dtype=np.int32
                ),

                "area": spaces.Box(
                    low=0,
                    high=SIZE_CAMERA * SIZE_CAMERA,
                    shape=(1,),
                    dtype=np.int32
                ),
            }
        )

        # Actions
        self._action_to_control = {}
        self._range_vel = range_vel
        self._range_steer = range_steer

        for i in range(self._range_vel):
            for j in range(self._range_steer + 1):
                vel = (i + 1) / self._range_vel * 10 # [1, 10]
                steer = j / self._range_steer * 0.4 - 0.2 # [-0.2, 0.2]
                self._action_to_control[i * self._range_steer + j] = np.array([vel, steer])
        self.action_space = spaces.Discrete(self._range_steer * (self._range_vel - 1))

        # Init locations
        self._town = 'Town05'
        z = 0.1
        self._init_locations = [
            carla.Transform(carla.Location(x=50.0, y=-145.7, z=z)),
            carla.Transform(carla.Location(x=120.0, y=-137, z=z), carla.Rotation(yaw=22)),
            carla.Transform(carla.Location(x=43.0, y=141.5, z=z), carla.Rotation(yaw=3)),
            carla.Transform(carla.Location(x=111.0, y=135.5, z=z), carla.Rotation(yaw=-16))
        ]
        self._index_loc = random.randint(0, len(self._init_locations) - 1)

        # Pygame window
        if self._human:
            self._screen = configcarla.setup_pygame(size=(SIZE_CAMERA * 2, SIZE_CAMERA), 
                                                    name='Follow lane: CarlaDiscreteBasic')
            init_driver = (SIZE_CAMERA, 0)
        else:
            self._screen = None
            init_driver = None

        # Init simulation
        if train:
            assert fixed_delta_seconds > 0.0, "In synchronous mode fidex_delta_seconds can't be 0.0"
        self._world, _ = configcarla.setup_carla(name_world=self._town, port=port, syn=train, 
                                                 fixed_delta_seconds=fixed_delta_seconds)
        
        # Swap ego vehicle
        self._ego_vehicle = configcarla.add_one_vehicle(world=self._world, ego_vehicle=True,
                                                        vehicle_type='vehicle.lincoln.mkz_2020',
                                                        transform=self._init_locations[self._index_loc])
        transform = carla.Transform(carla.Location(z=2.0, x=1.25), carla.Rotation(roll=90.0, pitch=-2.0))
        self._sensors = configcarla.Vehicle_sensors(vehicle=self._ego_vehicle, world=self._world,
                                                    screen=self._screen)
        self._camera = self._sensors.add_camera_rgb(transform=transform, seg=False, lane=True,
                                                    canvas_seg=False, size_rect=(SIZE_CAMERA, SIZE_CAMERA),
                                                    init_extra=init_driver, text='Driver view')

        if self._human:
            world_transform = carla.Transform(carla.Location(z=2.5, x=-4.75), carla.Rotation(roll=90.0))
            self._sensors.add_camera_rgb(transform=world_transform, size_rect=(SIZE_CAMERA, SIZE_CAMERA),
                                         init=(0, 0), text='World view')

        self._max_count = 600
        self._count = 0

    def _get_obs(self):
        cm = self._camera.get_lane_cm()
        area = np.array([self._camera.get_lane_area()], dtype=np.int32)
        left_points, right_points = self._camera.get_lane_points()

        return {"cm": cm, "left_points": left_points, "right_points": right_points, "area": area}
    
    def _get_info(self):
        return {"deviation": self._dev}

    def step(self, action:int):
        # Exec actions
        vel, steer = self._action_to_control[action]
        control = carla.VehicleControl()
        control.steer = steer

        # Set velocity
        vel_current = self._ego_vehicle.get_velocity()
        vel_current = carla.Vector3D(vel_current).length()
        diff = vel - vel_current
        if diff > 0:
            control.throttle = min(diff, 1)
        else:
            control.brake = min(-diff, 1.0)

        # Apply control
        self._ego_vehicle.apply_control(control)

        terminated = False
        truncated = False

        try:
            # Update data
            self._world.tick()
            self._sensors.update_data()

            # Calculate reward
            self._dev = self._camera.get_deviation()
            reward = 1 / (abs(self._dev) + 1)

            t = self._ego_vehicle.get_transform()
            if self._index_loc >= 2:
                if not self._jump and t.location.y < 10:
                    t.location.y = -20
                    self._ego_vehicle.set_transform(t)
                    self._jump = True
                elif t.location.y < -146:
                    reward = 1
                    terminated = True
            else:
                if not self._jump and t.location.y > -18:
                    t.location.y = 15
                    self._ego_vehicle.set_transform(t)
                    self._jump = True
                elif t.location.x < 43:
                    reward = 1
                    terminated = True
            
        except AssertionError:
            terminated = True
            reward = 0
            if self._dev >= 0:
                self._dev = abs(reward)
            else:
                self._dev = reward

        if self._count > self._max_count:
            terminated = True 
            reward = 0
            truncated = True
        self._count += 1

        if not terminated:
            reward += vel_current / 100 

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def reset(self, seed=None):
        if seed is not None:
            super().reset(seed=seed)
            
        self._count = 0
        self._jump = False

        self._index_loc = random.randint(0, len(self._init_locations) - 1)
        self._ego_vehicle.set_transform(self._init_locations[self._index_loc])
        self._ego_vehicle.apply_control(carla.VehicleControl())
        self._sensors.reset()

        return self._get_obs(), {}

    def close(self):
        pygame.quit()