import pygame
import carla
import configcarla
from PIL import Image

HEIGHT= 610
WIDTH = 610

ROAD = 0

def show_mask(mask:list[list[int]], screen:pygame.Surface, rect:pygame.Rect):
    if len(mask) == 0:
        return 

    # Create image from mask including only the road
    image = Image.new('RGB', (len(mask), len(mask[0])), color=0) 
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if mask[i][j] == ROAD:
                image.putpixel((i, j), (128, 64, 128))
    image = image.resize((rect.width, rect.height))

    # Show in Pygame
    image_data = image.tobytes()
    surface = pygame.image.fromstring(image_data, image.size, image.mode)
    surface = pygame.transform.flip(surface, True, False)
    screen.blit(surface, rect)

def main():
    world, _ = configcarla.setup_carla(name_world='Town04', port=2000)
    screen = configcarla.setup_pygame(size=(WIDTH * 3, HEIGHT), name='PID')

    # Surface to show segmentation mask road
    sub_screen_mask = pygame.Surface((WIDTH, HEIGHT))
    rect_mask = sub_screen_mask.get_rect(topleft=(WIDTH * 2 , 0))

    # Add Ego Vehicle
    ego_transform = carla.Transform(carla.Location(x=402.0, y=-94.0, z=0.4), carla.Rotation(yaw=-90.0))
    ego_vehicle = configcarla.add_one_vehicle(world=world, ego_vehicle=True, transform=ego_transform,
                                              vehicle_type='vehicle.lincoln.mkz_2020')
    sensors = configcarla.Vehicle_sensors(vehicle=ego_vehicle, world=world, screen=screen)

    # Add sensors to Ego Vehicle
    camera_transform = carla.Transform(carla.Location(z=2.5, x=0.75), carla.Rotation(pitch=-0.0, roll=90.0))
    camera = sensors.add_camera_rgb(size_rect=(WIDTH, HEIGHT), init=(0, 0), transform=camera_transform,
                                    seg=True, init_seg=(WIDTH, 0))
    
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
           
            # Upadate data of sensors
            sensors.update_data(flip=False)

            # Treat and display mask
            mask = camera.get_segmentation_mask()
            show_mask(mask=mask, rect=rect_mask, screen=screen)

            # Update pygame and carla
            pygame.display.flip()
            world.tick()

    except KeyboardInterrupt:
        return

    finally:
        sensors.destroy()
        pygame.quit()

if __name__ == "__main__":
    main()