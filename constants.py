# Game settings
WIDTH = 640
HEIGHT = 480
BLOCK_SIZE = 20
SPEED = 40  # For rendering speed in test.py

# RL Hyperparameters for PPO
TRAINING_TIMESTEPS = 100000  # Total steps to train the agent
LEARNING_RATE = 0.0003
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_N_STEPS = 2048 # Number of steps to run for each environment per update
PPO_N_EPOCHS = 10
PPO_BATCH_SIZE = 64

# Reward values
REWARD_EAT_FOOD = 25.0
REWARD_GAME_OVER = -50.0
REWARD_MOVED_CLOSER = 1.0
REWARD_MOVED_AWAY = -1.5