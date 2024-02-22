import carla
import math

SCALE = 7.0
camera = None

def center_spectator(spectator):
    transform = spectator.get_transform()
    yaw = math.radians(transform.rotation.yaw)

    transform.location.z += 0.3
    transform.location.x -= SCALE * math.cos(yaw)
    transform.location.y -= SCALE * math.sin(yaw)
    transform.rotation.pitch = -15.0

    spectator.set_transform(transform)

def setup_carla(port=2000, name_world='Town01', vehicle='vehicle.lincoln.mkz_2020', transform=carla.Transform()):
    client = carla.Client('localhost', port)
    world = client.get_world()
    client.load_world(name_world)

    # Retrieve and locate spectator object
    spectator = world.get_spectator()
    transform.rotation.pitch = 0.0
    transform.rotation.roll = 0.0
    transform.rotation.yaw = 0.0 # cmabiar
    spectator.set_transform(transform)

    # Define Ego Vehicle and spawn it at spectator's location
    ego_bp = world.get_blueprint_library().find(vehicle)
    ego_bp.set_attribute('role_name', 'hero')
    transform.location.z = 1.0
    ego_vehicle = world.spawn_actor(ego_bp, transform)

    return world, spectator, ego_vehicle

def add_camera(vehicle, world):
    # Create a transform to place the camera on top of the vehicle
    camera_init_trans = carla.Transform(carla.Location(z=2.5, x=0.5), carla.Rotation(pitch=-10.0))

    # We create the camera through a blueprint that defines its properties
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')

    # We spawn the camera and attach it to our ego vehicle
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

    return camera

def main():
    global camera

    world, spectator, ego_vehicle = setup_carla(name_world='Town03', transform=carla.Transform(carla.Location(x=100.0, y=-6.0, z=4.0)))
    center_spectator(spectator)
    camera = add_camera(vehicle=ego_vehicle, world=world)

    camera.listen(lambda image: image.save_to_disk('out/%06d.png' % image.frame))
    import time
    time.sleep(3)

if __name__ == "__main__":
    main()

# Start camera with PyGame callback
#camera.listen(lambda image: image.save_to_disk('out/%06d.png' % image.frame))

# Teleoperador
# Añadir vehiculos con autopilo

