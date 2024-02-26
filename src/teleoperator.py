import carla
import math
import pygame
import numpy as np

# Spectator
SCALE = 5.0
ELEVATION = 3.0
PITCH = -10.0

# Screen
HEIGHT = 500
WIDTH = 700

camera = None

def center_spectator(spectator):
    transform = spectator.get_transform()
    yaw = math.radians(transform.rotation.yaw)

    transform.location.x -= SCALE * math.cos(yaw)
    transform.location.y -= SCALE * math.sin(yaw)
    transform.rotation.pitch = PITCH

    spectator.set_transform(transform)

def setup_carla(port=2000, name_world='Town01', vehicle='vehicle.lincoln.mkz_2020', transform=carla.Transform()):
    client = carla.Client('localhost', port)
    world = client.get_world()
    client.load_world(name_world)

    # Retrieve and locate spectator object
    spectator = world.get_spectator()
    transform.rotation.pitch = 0.0
    transform.rotation.roll = 0.0
    transform.location.z = ELEVATION
    spectator.set_transform(transform)

    # Define Ego Vehicle and spawn it at spectator's location
    ego_bp = world.get_blueprint_library().find(vehicle)
    ego_bp.set_attribute('role_name', 'hero')
    ego_vehicle = world.spawn_actor(ego_bp, transform)

    return world, spectator, ego_vehicle

def add_camera(vehicle, world):
    global camera

    # Create a transform to place the camera on top of the vehicle
    camera_init_trans = carla.Transform(carla.Location(z=2.5, x=0.5), carla.Rotation(pitch=-10.0, roll=90.0))

    # We create the camera through a blueprint that defines its properties
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')

    # We spawn the camera and attach it to our ego vehicle
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

    return camera

def setup_pygame():
    pygame.init()
    pygame.display.set_caption('Camera stream')
    screen = pygame.display.set_mode((WIDTH, HEIGHT), 0)
    clock = pygame.time.Clock()

    return screen, clock

def show_image(image, screen):
    # Convert the image to a numpy array
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))

    # Convert RGBA to RGB
    array = array[:, :, :3]

    # Create a Pygame surface 
    image_surface = pygame.surfarray.make_surface(array)

    # Resize the image
    screen_surface = pygame.transform.scale(image_surface, (WIDTH, HEIGHT))

    screen.blit(screen_surface, (0, 0))
    pygame.display.flip()

def main():
    global camera

    # Setup CARLA
    world, spectator, ego_vehicle = setup_carla(name_world='Town03', transform=carla.Transform(carla.Location(x=100.0, y=-6.0)))
    center_spectator(spectator)

    # Init pygame
    screen, clock = setup_pygame()

    # Add camera
    camera = add_camera(vehicle=ego_vehicle, world=world)
    camera.listen(lambda image: show_image(image, screen))

    # problemas al cerrar a veces

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            clock.tick(5)

    except KeyboardInterrupt:
        return

    finally:
        pygame.quit()

if __name__ == "__main__":
    main()

# Teleoperador
# Añadir vehiculos con autopilo

