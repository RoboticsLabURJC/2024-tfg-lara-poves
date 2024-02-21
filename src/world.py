import carla

# Connect to the client and retrieve the world object
client = carla.Client('localhost', 2000)
world = client.get_world()

# Loading map
client.load_world('Town03')

# Retrieve the spectator object
spectator = world.get_spectator()

# Get the location and rotation of the spectator through its transform
spectator_transform = spectator.get_transform()
spectator_transform.location.x = 100.0

# Define Ego Vehicle
ego_bp = world.get_blueprint_library().find('vehicle.lincoln.mkz_2020')
ego_bp.set_attribute('role_name', 'hero')

# Spawn the Ego Vehicle at the spectator's location
ego_vehicle = world.spawn_actor(ego_bp, spectator_transform)


# Create a transform to place the camera on top of the vehicle
camera_init_trans = carla.Transform(carla.Location(z=2.0, x=0.75), carla.Rotation(pitch=-5.0))

# We create the camera through a blueprint that defines its properties
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')

# We spawn the camera and attach it to our ego vehicle
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)

# Start camera with PyGame callback
#camera.listen(lambda image: image.save_to_disk('out/%06d.png' % image.frame))

# Change spectator location in order to see the vehicle
spectator_transform.location.z += 0.3
spectator_transform.location.x += 7.0
spectator_transform.rotation.pitch -= 15.0

# Set the spectator 
spectator.set_transform(spectator_transform)

# Teleoperador
# Añadir vehiculos con autopilo

