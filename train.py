import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from snake_env import SnakeEnv
from constants import *

# --- Setup ---
log_dir = "logs/"
model_dir = "models/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Create training and evaluation environments
train_env = SnakeEnv()
train_env = Monitor(train_env, log_dir)

eval_env = SnakeEnv()
eval_env = Monitor(eval_env, log_dir + "eval/")

# --- Callbacks ---
# Save checkpoints every 50,000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path=log_dir,
    name_prefix='ppo_snake'
)

# Evaluate the agent every 25,000 steps
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=log_dir + "best_model/",
    log_path=log_dir + "eval/",
    eval_freq=25000,
    deterministic=True,
    render=False,
    verbose=1
)

# --- Model Definition ---
print("Creating PPO model...")
model = PPO(
    'MlpPolicy',
    train_env,
    verbose=1,  # This will show progress automatically
    n_steps=PPO_N_STEPS,
    batch_size=PPO_BATCH_SIZE,
    n_epochs=PPO_N_EPOCHS,
    gamma=GAMMA,
    gae_lambda=GAE_LAMBDA,
    learning_rate=LEARNING_RATE,
    tensorboard_log="./ppo_snake_tensorboard/"
)

# --- Training ---
print(f"--- Starting Training for {TRAINING_TIMESTEPS:,} timesteps ---")
print(f"Estimated time: {TRAINING_TIMESTEPS / 2000:.0f}-{TRAINING_TIMESTEPS / 1000:.0f} minutes")
print("Use 'tensorboard --logdir ./ppo_snake_tensorboard/' to monitor training")

start_time = time.time()

model.learn(
    total_timesteps=TRAINING_TIMESTEPS,
    callback=[checkpoint_callback, eval_callback],
    progress_bar=True  # Shows a progress bar during training
)

training_time = time.time() - start_time
print(f"--- Training Finished in {training_time / 60:.1f} minutes ---")

# --- Save Final Model ---
model.save(f"{model_dir}/ppo_snake_final")
print(f"Final model saved to {model_dir}/ppo_snake_final")

# --- Quick Performance Test ---
print("\n--- Quick Performance Test ---")
obs, info = train_env.reset()
total_reward = 0
steps = 0

while steps < 1000:  # Test for 1000 steps max
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = train_env.step(action)
    total_reward += reward
    steps += 1

    if terminated or truncated:
        print(f"Test game ended after {steps} steps with score {info.get('score', 0)} and reward {total_reward:.1f}")
        break

train_env.close()
eval_env.close()