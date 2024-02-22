import pygame
import sys
import carla

WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Teleoperator")

operator_width, operator_height = 50, 50
operator_x, operator_y = WIDTH // 2 - operator_width // 2, HEIGHT // 2 - operator_height // 2

operator_speed = 5

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT]:
        operator_x -= operator_speed
    if keys[pygame.K_RIGHT]:
        operator_x += operator_speed
    if keys[pygame.K_UP]:
        operator_y -= operator_speed
    if keys[pygame.K_DOWN]:
        operator_y += operator_speed

    operator_x = max(0, min(WIDTH - operator_width, operator_x))
    operator_y = max(0, min(HEIGHT - operator_height, operator_y))

    screen.fill(BLACK)

    pygame.draw.rect(screen, WHITE, (operator_x, operator_y, operator_width, operator_height))

    pygame.display.flip()

    pygame.time.Clock().tick(30)
