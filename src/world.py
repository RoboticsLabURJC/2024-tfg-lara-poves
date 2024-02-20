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

# Change spectator location in order to see the vehicle
spectator_transform.location.z += 1.0
spectator_transform.location.x += 9.0

# Set the spectator 
spectator.set_transform(spectator_transform)
