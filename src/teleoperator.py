import carla
import math
import pygame
import numpy as np
import time

# Spectator
SCALE = 5.0
ELEVATION = 3.0
PITCH = -10.0

# Screen
HEIGHT_SCREEN = 500
WIDTH_SCREEN = 700

# Control velocity 
BRAKE = 1.0
STEER = 0.3
INCREMENT = 0.1

# Global variables
camera = None
image_stream = None

def center_spectator(spectator, transform):
    yaw = math.radians(transform.rotation.yaw)

    transform.location.z = ELEVATION
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
    center_spectator(spectator, transform)  #no va

    # Create Ego Vehicle and spawn it at spectator's location
    ego_bp = world.get_blueprint_library().find(vehicle)
    ego_bp.set_attribute('role_name', 'hero')
    ego_vehicle = world.spawn_actor(ego_bp, transform)

    # Configure Ego Vehicle
    control = carla.VehicleControl()
    control.throttle = 0.0  
    control.steer = 0.0     
    control.brake = 0.0     
    control.hand_brake = True  # cambiar y mirar si luego s equita soloß
    control.reverse = False  
    control.manual_gear_shift = False
    ego_vehicle.apply_control(control)
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
    screen = pygame.display.set_mode((WIDTH_SCREEN, HEIGHT_SCREEN), 0)
    clock = pygame.time.Clock()

    return screen, clock

def update_image(image):
    global image_stream
    image_stream = image

def show_camera(screen):
    global image_stream

    if image_stream is not None:
        # Convert the image to a numpy array
        array = np.frombuffer(image_stream.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image_stream.height, image_stream.width, 4))

        # Convert RGBA to RGB
        array = array[:, :, :3]

        # Create a Pygame surface 
        image_surface = pygame.surfarray.make_surface(array)

        # Resize the image
        screen_surface = pygame.transform.scale(image_surface, (WIDTH_SCREEN, HEIGHT_SCREEN))

        screen.blit(screen_surface, (0, 0))
        pygame.display.flip()

def update_vel(vehicle, event):
    control = vehicle.get_control()
    control.steer = 0.0     
    control.brake = 0.0  

    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_LEFT:
            control.steer = STEER
        elif event.key == pygame.K_RIGHT:
            control.steer = -STEER
        elif event.key == pygame.K_UP:
            control.throttle = min(1.0, control.throttle + INCREMENT)
        elif event.key == pygame.K_DOWN:
            control.brake = BRAKE
            control.throttle = max(0.0, control.throttle - INCREMENT)

    vehicle.apply_control(control)

    # Imprimir por pantalla, como el ejemplo que vi
    #print(control.throttle)
    
def main():
    global camera

    # Setup CARLA and pygame
    world, spectator, ego_vehicle = setup_carla(name_world='Town03', transform=carla.Transform(carla.Location(x=100.0, y=-6.0)))
    screen, clock = setup_pygame()

    print("Preparing Carla and pygame...")
    time.sleep(3)
    print("Setup completed")

    # Add camera
    camera = add_camera(vehicle=ego_vehicle, world=world)
    camera.listen(lambda image: update_image(image))

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                
                update_vel(ego_vehicle, event)
                #center_spectator()

            show_camera(screen)
            clock.tick(60) # Frame rate

    except KeyboardInterrupt:
        pygame.quit()

if __name__ == "__main__":
    main()
