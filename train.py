import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from snake_env import SnakeEnv
from constants import *

# --- Setup ---
# Create directories for logs and models
log_dir = "logs/"
model_dir = "models/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Instantiate the environment
env = SnakeEnv()
env = Monitor(env, log_dir)

# --- Callbacks ---
# Save a checkpoint of the model every 10,000 steps
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix='ppo_snake')

# --- Model Definition ---
# Here we define the PPO model
model = PPO('MlpPolicy',
            env,
            verbose=1,
            n_steps=PPO_N_STEPS,
            batch_size=PPO_BATCH_SIZE,
            n_epochs=PPO_N_EPOCHS,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            learning_rate=LEARNING_RATE,
            tensorboard_log="./ppo_snake_tensorboard/")

# --- Training ---
print("--- Starting Training ---")
model.learn(total_timesteps=TRAINING_TIMESTEPS, callback=checkpoint_callback)
print("--- Training Finished ---")

# --- Save Final Model ---
model.save(f"{model_dir}/ppo_snake_final")
print(f"Final model saved to {model_dir}/ppo_snake_final")

env.close()