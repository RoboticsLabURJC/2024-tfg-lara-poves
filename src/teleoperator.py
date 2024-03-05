import carla
import pygame
import numpy as np
import time

# Screen
HEIGHT_SCREEN = 600
WIDTH_SCREEN = 600
ELEVATION = 2.5

# Control velocity 
BRAKE = 1.0
STEER = 0.4
THROTTLE = 0.7

class Camera_screen:
    def __init__(self, name, vehicle, world, num_screen, transform=carla.Transform()):
        pygame.display.set_caption(name)
        self.screen = pygame.Surface((WIDTH_SCREEN, HEIGHT_SCREEN))
        self.screen_rect = self.screen.get_rect(topleft=((num_screen - 1) * WIDTH_SCREEN, 0))
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
            screen_surface = pygame.transform.scale(image_surface, (WIDTH_SCREEN, HEIGHT_SCREEN))
            
            screen.blit(screen_surface, self.screen_rect)

def setup_carla(port=2000, vehicle='vehicle.lincoln.mkz_2020', name_world='Town01', transform=carla.Transform()):
    # Connect to the server
    client = carla.Client('localhost', port)
    world = client.get_world()
    client.load_world(name_world)

    # Create and locate ego vehicle 
    ego_bp = world.get_blueprint_library().find(vehicle)
    ego_bp.set_attribute('role_name', 'hero')
    ego_vehicle = world.spawn_actor(ego_bp, transform)

    # Configure ego vehicle's control
    control = carla.VehicleControl()
    control.throttle = 0.0  
    control.steer = 0.0     
    control.brake = 0.0     
    control.hand_brake = False
    control.reverse = False  
    control.manual_gear_shift = False
    ego_vehicle.apply_control(control)

    return world, ego_vehicle

def setup_pygame(num_screen):
    pygame.init()
    clock = pygame.time.Clock()

    screen = pygame.display.set_mode((WIDTH_SCREEN * num_screen, HEIGHT_SCREEN))
    pygame.display.set_caption('Stream cameras')

    return screen, clock

def update_vel(vehicle):
    control = vehicle.get_control()
    control.steer = 0.0     
    control.brake = 0.0  
    control.throttle = 0.0

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
    
def main():
    # Setup CARLA and pygame
    vehicle_transform = carla.Transform(carla.Location(x=100.0, y=-6.0, z=ELEVATION))
    world, ego_vehicle = setup_carla(name_world='Town03', transform=vehicle_transform)
    screen, clock = setup_pygame(num_screen=2)

    print("Preparing Carla and pygame...")
    time.sleep(2)
    print("Setup completed")

    # Create cameras' screens
    driver_transform = carla.Transform(carla.Location(z=ELEVATION, x=0.5), carla.Rotation(pitch=-10.0, roll=90.0))
    driver = Camera_screen(name='Stream camera driver', num_screen=1, vehicle=ego_vehicle, world=world, transform=driver_transform)

    spectator_transform = carla.Transform(carla.Location(x=-4.0, z=ELEVATION), carla.Rotation(pitch=-10.0, roll=90.0))
    spectator = Camera_screen(name='Teleoperator', num_screen=2, vehicle=ego_vehicle, world=world, transform=spectator_transform)

    try:
        while True:
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    update_vel(ego_vehicle)

            driver.show_camera(screen)
            spectator.show_camera(screen)
            pygame.display.flip()
            clock.tick(60) # Frame rate

    except KeyboardInterrupt:
        pygame.quit()

if __name__ == "__main__":
    main()
