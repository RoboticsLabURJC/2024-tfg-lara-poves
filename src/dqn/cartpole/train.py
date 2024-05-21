import gymnasium as gym
from stable_baselines3 import DQN

def main():
    env = gym.make("CartPole-v1", render_mode="human")

    model_params = {
        "learning_rate": 2.3e-3,
        "batch_size": 64, # Experiences per batch (N)
        "buffer_size": 100000, # Experiences in replay memory
        "learning_starts": 1000,
        "gamma": 0.99,
        "target_update_interval": 10, # (F)
        "train_freq": 256, # Update the model every 256 steps
        "gradient_steps": 128, # Updates per batch (U)
        "exploration_fraction": 0.16, # Decay epsilon
        "exploration_final_eps": 0.04, # Final value of epsilon
        "policy_kwargs": dict(net_arch=[256, 256])
    }

    # Verbose: 0 (nothing), 1 (info messages), 2 (debug)
    model = DQN("MlpPolicy", env, verbose=1, **model_params)

    # log interval: number of episodes before logging information
    model.learn(total_timesteps=10000, log_interval=4, tb_log_name='DQN-CartPole', progress_bar=True)

    model.save("model")

if __name__ == "__main__":
    main()