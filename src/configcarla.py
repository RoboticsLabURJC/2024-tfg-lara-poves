import carla
import pygame
import numpy as np
import math
import random

class Sensor:
    def __init__(self, size, init, sensor):
        self.rect = size
        self.data = None
        self.sensor = sensor

        if size != None: 
            sub_screen = pygame.Surface(size)
            self.rect = sub_screen.get_rect(topleft=init)

    def update_data(self, data):
        self.data = data
    
    def show_image(self, screen):
        if self.data == None or self.rect == None:
            return
        
        if isinstance(self.data, carla.Image):
            # Convert the image to a numpy array
            array = np.frombuffer(self.data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.data.height, self.data.width, 4))

            # Swap blue and red channels
            array = array[:, :, (2, 1, 0)]

            # Create a Pygame surface 
            image_surface = pygame.surfarray.make_surface(array)

            # Reverse mirror effect
            flipped_surface = pygame.transform.flip(image_surface, True, False)

            # Resize the image
            screen_surface = pygame.transform.scale(flipped_surface, self.rect.size)

            screen.blit(screen_surface, self.rect)

class Vehicle_sensors:
    def __init__(self, vehicle, world, screen):
        self.vehicle = vehicle
        self.world = world
        self.screen = screen
        self.sensors = []

    def add_sensor(self, sensor, size=None, init=(0, 0), transform=carla.Transform()):
        try:
            sensor_bp = self.world.get_blueprint_library().find(sensor)
        except IndexError:
            print("Sensor", sensor, "doesn't exist!")
            return
        
        sensor = self.world.spawn_actor(sensor_bp, transform, attach_to=self.vehicle)
        sensor_class = Sensor(size=size, init=init, sensor=sensor)
        self.sensors.append(sensor_class)
        sensor.listen(lambda data: sensor_class.update_data(data))
        
    def resize_screen(self, width=None, height=None):
        current_width, current_height = self.screen.get_size()
        if width == None:
            width = current_width
        if height == None:
            height = current_height

        self.screen = pygame.display.set_mode((width, height))

    def update_screen(self):
        for sensor in self.sensors:
            sensor.show_image(self.screen)

        pygame.display.flip()

    def destroy(self):
        for sensor in self.sensors:
            sensor.sensor.destroy()

        self.vehicle.destroy()

class Teleoperator:
    def __init__(self, vehicle, steer=0.3, throttle=0.6, brake=1.0):
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

    def set_steer(self, steer):
        self.steer = max(0.0, min(1.0, steer))

    def set_throttle(self, throttle):
        self.throttle = max(0.0, min(1.0, throttle))

    def set_brake(self, brake):
        self.brake = max(0.0, min(1.0, brake))

def setup_carla(port=2000, vehicle='vehicle.lincoln.mkz_2020', 
                name_world='Town01', transform=None):
    # Connect to the server
    client = carla.Client('localhost', port)
    world = client.get_world()
    client.load_world(name_world)

    if transform == None:
        spawn_points = world.get_map().get_spawn_points()
        transform = random.choice(spawn_points)

    # Create and locate ego vehicle 
    ego_bp = world.get_blueprint_library().find(vehicle)
    ego_bp.set_attribute('role_name', 'hero')
    ego_vehicle = world.spawn_actor(ego_bp, transform)

    return world, ego_vehicle, client

def center_spectator(world, transform, scale=5.5, height=3.0, pitch=-10.0):
    yaw = math.radians(transform.rotation.yaw)
    spectator =  world.get_spectator()

    transform.location.z = height
    transform.location.x -= scale * math.cos(yaw)
    transform.location.y -= scale * math.sin(yaw)
    transform.rotation.pitch = pitch

    spectator.set_transform(transform)

def setup_pygame(size, name):
    pygame.init()
    clock = pygame.time.Clock()

    screen = pygame.display.set_mode(size)
    pygame.display.set_caption(name)

    return screen, clock

def add_vehicles(world, number):
    vehicle_bp = world.get_blueprint_library().filter('*vehicle*')
    spawn_points = world.get_map().get_spawn_points()

    vehicles = []
    for _ in range(number):
        v = world.try_spawn_actor(random.choice(vehicle_bp), random.choice(spawn_points))
        if v is not None:
            vehicles.append(v)

    return vehicles

def traffic_manager(client, vehicles, port=5000, dist=3.0, speed_lower=10.0):
    tm = client.get_trafficmanager(port)
    tm_port = tm.get_port()

    for v in vehicles:
        v.set_autopilot(True, tm_port)
        tm.auto_lane_change(v, False) 

    tm.set_global_distance_to_leading_vehicle(dist)
    tm.global_percentage_speed_difference(speed_lower)

    return tm
