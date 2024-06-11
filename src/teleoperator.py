import pygame
import carla
import configcarla
import argparse

# Buttom
BUTTOM_H = 30
BUTTOM_W = 90
TEXT_SIZE = 20
OFFSET = 0.05

class Buttom:
    def __init__(self, text:str, x:int=0, y:int=0, color_background:tuple[int, int, int]=(0, 0, 0),
                 color_text:tuple[int, int, int]=(255, 255, 255)):
        self.rect = pygame.Rect(x, y, BUTTOM_W, BUTTOM_H)
        self.color_text = color_text
        self.color_background = color_background
   
        font = pygame.font.Font(None, TEXT_SIZE * 2)
        self.text = font.render(text, True, color_text)
        self.text_rect = self.text.get_rect(center=self.rect.center)

    def draw(self, screen:pygame.Surface):
        pygame.draw.rect(screen, self.color_background, self.rect)
        screen.blit(self.text, self.text_rect)

    def collision(self, point):
        return self.rect.collidepoint(point)
    
def main(args):
    # Setup CARLA and Pygame
    world, _ = configcarla.setup_carla(name_world='Town03', syn=False, port=args.port)
    screen = configcarla.setup_pygame(size=(configcarla.SIZE_CAMERA * 2, configcarla.SIZE_CAMERA), 
                                      name='Teleoperator')

    # Add Ego Vehicle
    vehicle_transform = carla.Transform(carla.Location(x=100.0, y=-6.0, z=2.5))
    ego_vehicle = configcarla.add_one_vehicle(world=world, vehicle_type='vehicle.lincoln.mkz_2020',
                                              transform=vehicle_transform, ego_vehicle=True)

    # Create teleoperator
    teleop = configcarla.Teleoperator(ego_vehicle, throttle=0.6)

    # Add cameras
    cameras = configcarla.Vehicle_sensors(vehicle=ego_vehicle, world=world, screen=screen)

    camera_transform = carla.Transform(carla.Location(z=2.5, x=0.5), carla.Rotation(pitch=-10.0, roll=90.0))
    cameras.add_camera_rgb(size_rect=(configcarla.SIZE_CAMERA, configcarla.SIZE_CAMERA), init=(0, 0), 
                           text='Driver view', transform=camera_transform)
    
    camera_transform.location.x = -4.0
    cameras.add_camera_rgb(size_rect=(configcarla.SIZE_CAMERA, configcarla.SIZE_CAMERA),
                           init=(configcarla.SIZE_CAMERA, 0),
                           text='World View', transform=camera_transform)

    # Instance buttoms
    buttom_decrease = Buttom(text="-", x=configcarla.SIZE_CAMERA - BUTTOM_W, 
                             y=int(configcarla.SIZE_CAMERA / 2) + 12, color_background=(230, 50, 25))
    buttom_increase = Buttom(text="+", x=configcarla.SIZE_CAMERA, 
                             y=int(configcarla.SIZE_CAMERA / 2) + 12, color_background=(50, 230, 25))
    buttoms = [buttom_increase, buttom_decrease]

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    for i in range(len(buttoms)):
                        if buttoms[i].collision(event.pos):
                            if i == 0:
                                throttle += OFFSET
                            else:
                                throttle -= OFFSET
                            teleop.set_throttle(throttle)

            cameras.update_data(flip=False)
            teleop.control()

            # Draw buttoms
            for b in buttoms:
                b.draw(screen)

            throttle = teleop.get_throttle()
            text = "Throttle = {:.2f}".format(throttle)
            configcarla.write_text(text=text, img=screen, size=TEXT_SIZE, bold=True, background=(0, 0, 0),
                                   point=(configcarla.SIZE_CAMERA, int(configcarla.SIZE_CAMERA / 2)))

            pygame.display.flip()

    except KeyboardInterrupt:
        return

    finally:
        cameras.destroy()
        pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execute an inference trial on a specified Gym environment",
        usage="python3 %(prog)s --port <port_number>"
    )
    parser.add_argument(
        '--port', 
        type=int, 
        required=False, 
        default=2000,
        help='Port for Carla'
    )

    main(parser.parse_args())
