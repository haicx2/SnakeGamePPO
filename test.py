import time
from stable_baselines3 import PPO
from snake_env import SnakeEnv
from constants import *

# Load the trained model
model_path = "models/ppo_snake_final"
try:
    model = PPO.load(model_path)
    print(f"Model loaded from {model_path}")
except FileNotFoundError:
    print(f"Model not found at {model_path}. Please train the model first.")
    exit(1)

# --- Run the Environment ---
env = SnakeEnv()
obs, info = env.reset()  # gymnasium returns (obs, info)

print("--- Starting Test ---")
for episode in range(5):  # Run 5 full episodes
    done = False
    terminated = False
    truncated = False
    total_reward = 0

    print(f"Starting Episode {episode + 1}")

    while not (terminated or truncated):
        env.render()
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)  # gymnasium returns 5 values
        total_reward += reward
        time.sleep(1 / SPEED)  # Slow down rendering to watchable speed

    print(f"Episode {episode + 1} finished! Score: {info.get('score')}, Total Reward: {total_reward:.2f}")

    # Reset for next episode (except on the last one)
    if episode < 4:
        obs, info = env.reset()

print("--- Test Finished ---")
env.close()