import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import LoadMonitorResultsError, load_results
import os

def main():
    #model = DQN.load("model")
    #env = gym.make("CartPole-v1", render_mode="human")

    print(gym.envs.registry)

   

    # Imprimir los nombres de los entornos de CartPole
    for env in cartpole_envs:
        print(env)

    '''
    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True) # _states: internal state of the model
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated: # the episode
            break
    '''
    

if __name__ == "__main__":
    main()            