import carla
import pygame
import numpy as np

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

            # Convert RGBA to RGB
            array = array[:, :, :3]

            # Create a Pygame surface 
            image_surface = pygame.surfarray.make_surface(array)

            # Resize the image
            screen_surface = pygame.transform.scale(image_surface, self.rect.size)
            
            screen.blit(screen_surface, self.rect)

def setup_carla(port=2000, vehicle='vehicle.lincoln.mkz_2020', 
                name_world='Town01', transform=carla.Transform()):
    # Connect to the server
    client = carla.Client('localhost', port)
    world = client.get_world()
    client.load_world(name_world)

    # Create and locate ego vehicle 
    ego_bp = world.get_blueprint_library().find(vehicle)
    ego_bp.set_attribute('role_name', 'hero')
    ego_vehicle = world.spawn_actor(ego_bp, transform)

    return world, ego_vehicle

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
        control.steer = STEER
    if keys[pygame.K_RIGHT]:
        control.steer = -STEER
    if keys[pygame.K_UP]:
        control.throttle = THROTTLE
    if keys[pygame.K_DOWN]:
        control.brake = BRAKE

    vehicle.apply_control(control)