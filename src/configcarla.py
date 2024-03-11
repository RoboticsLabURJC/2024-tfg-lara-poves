import carla
import pygame
import numpy as np
import math
import random

# Control velocity 
BRAKE = 1.0
STEER = 0.3
THROTTLE = 0.6

class Camera_stream:
    def __init__(self, vehicle, world, rect,  transform=carla.Transform()):
        self.rect = rect
        self.image = None

        # Add camera
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        self.camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
        self.camera.listen(lambda image: self.update_image(image))

    def update_image(self, image):
        self.image = image

    def show_camera(self, screen):
        if self.image is not None:
            # Convert the image to a numpy array
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))

            # Swap blue and red channels
            array = array[:, :, (2, 1, 0)]

            # Create a Pygame surface 
            image_surface = pygame.surfarray.make_surface(array)

            # Reverse mirror effect
            flipped_surface = pygame.transform.flip(image_surface, True, False)

            # Resize the image
            screen_surface = pygame.transform.scale(flipped_surface, self.rect.size)

            screen.blit(screen_surface, self.rect)

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

def setup_pygame(width, height, name):
    pygame.init()
    clock = pygame.time.Clock()

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(name)

    return screen, clock

def teleoperator(vehicle):
    control = carla.VehicleControl()
    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT]:
        control.steer = -STEER
    if keys[pygame.K_RIGHT]:
        control.steer = STEER
    if keys[pygame.K_UP]:
        control.throttle = THROTTLE
    if keys[pygame.K_DOWN]:
        control.brake = BRAKE

    vehicle.apply_control(control)

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
