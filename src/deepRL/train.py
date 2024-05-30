import gymnasium as gym
from gym import spaces
import os
import sys
import numpy as np
import carla
import random

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, src_path)

from configcarla import SIZE_CAMERA
import configcarla

class CarlaDiscreteBasic(gym.Env):
    def __init__(self, train:bool, port:int=2000, fixed_delta_seconds=0.0, num_points_line:int=5, range_throttle:int=10, range_steer:int=50):
        self._dev = 0
        
        # States
        self.observation_space = spaces.Dict(
            {
                "cm": spaces.Box(
                    low=np.full((2,), -SIZE_CAMERA / 2, dtype=np.int32),
                    high=np.full((2,), SIZE_CAMERA / 2, dtype=np.int32),
                    shape=(2,),
                    dtype=np.int32
                ),

                "left_points": spaces.Box(
                    low=np.full((num_points_line, 2), 0, dtype=np.int32),
                    high=np.full((num_points_line, 2), SIZE_CAMERA - 1, dtype=np.int32),
                    shape=(num_points_line, 2),
                    dtype=np.int32
                ),

                "right_points": spaces.Box(
                    low=np.full((num_points_line, 2), 0, dtype=int),
                    high=np.full((num_points_line, 2), SIZE_CAMERA - 1, dtype=np.int32),
                    shape=(num_points_line, 2),
                    dtype=np.int32
                ),

                "area": spaces.Box(
                    low=0,
                    high=SIZE_CAMERA * SIZE_CAMERA,
                    shape=(1,),
                    dtype=np.int32
                )
            }
        )

        # Actions
        self._action_to_control = {}
        self._range_throttle = range_throttle
        self._range_steer = range_steer

        for i in range(self._range_throttle + 1):
            for j in range(self._range_steer + 1):
                self._action_to_control[i * self._range_steer + j] = np.array([i / self._range_throttle, j / self._range_steer])

        self.action_space = spaces.Discrete(self._range_steer * self._range_throttle)

        # Init locations
        self._town = 'Town04'
        self._init_locations = [
            carla.Transform(carla.Location(x=-25.0, y=-252, z=0.5), carla.Rotation(yaw=125.0)),
            carla.Transform(carla.Location(x=-25.0, y=-245.75, z=0.5), carla.Rotation(yaw=125.0)),
            carla.Transform(carla.Location(x=198.5, y=-163, z=0.5), carla.Rotation(yaw=90.0)),
        ]

        # Init simulation
        if train:
            assert fixed_delta_seconds > 0.0, "In synchronous mode fidex_delta_seconds can't be 0.0"
        self._world, _ = configcarla.setup_carla(name_world=self._town, port=port, syn=train, fixed_delta_seconds=fixed_delta_seconds)
        
        # Swap ego vehicle
        self._ego_vehicle = configcarla.add_one_vehicle(world=self._world, vehicle_type='vehicle.lincoln.mkz_2020',
                                                        ego_vehicle=True, transform=random.choice(self._init_locations))
        transform = carla.Transform(carla.Location(z=2.0, x=1.25), carla.Rotation(roll=90.0, pitch=-2.0))
        self._sensors = configcarla.Vehicle_sensors(vehicle=self._ego_vehicle, world=self._world)
        self._camera = self._sensors.add_camera_rgb(transform=transform, seg=True, lane=True, canvas_seg=False)

    def _get_obs(self):
        cm = self._camera.get_lane_cm()
        area = self._camera.get_lane_area()
        left_points, right_points = self._camera.get_lane_points()
        
        return {"cm": cm, "area": area, "left_points": left_points, "right_points": right_points}

    def _get_info(self):
        return {"desviation": self._dev}

    def step(self, action):
        # Exec actions
        throttle, steer = self._action_to_control[action]
        control = carla.VehicleControl()
        control.throttle = throttle
        control.steer = steer
        self._ego_vehicle.apply_control(control)

        terminated = False
        try:
            # Update data
            self._world.tick()
            self._sensors.update_data()

            # Calculate reward
            self._dev = self._camera.get_deviation()
            reward = self._dev if self._dev < 0 else self._dev * -1
            
        except AssertionError:
            terminated = True
            reward = -SIZE_CAMERA / 2
            if self._dev >= 0:
                self._dev = abs(reward)
            else:
                self._dev = reward

        return self._get_obs(), reward, terminated, False, self._get_info()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self._ego_vehicle.set_transform(random.choice(self._init_locations))

    def render(self):
        pass # No rendering

def main():
    cv = CarlaDiscreteBasic(train=True, fixed_delta_seconds=0.05)
    print(cv.observation_space)
    print(cv.action_space)
    cv.reset()
    for i in range(100):
        obs, reward, terminate, truncated, info = cv.step(action = 533)
        print(reward, info)
        if terminate or truncated:
            break

if __name__ == "__main__":
    main()