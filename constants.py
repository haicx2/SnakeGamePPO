# Game settings
WIDTH = 640
HEIGHT = 480
BLOCK_SIZE = 20
SPEED = 40

TRAINING_TIMESTEPS = 1000000  # Increased from 100k to 500k for better learning

LEARNING_RATE = 0.0003  # Good default
GAMMA = 0.99  # Good for long-term rewards
GAE_LAMBDA = 0.95  # Good default

# PPO-specific parameters - optimized for Snake
PPO_N_STEPS = 2048  # Steps per environment per update
PPO_N_EPOCHS = 10   # Number of epochs per update
PPO_BATCH_SIZE = 64 # Batch size for training

# Early stopping criteria (optional)
TARGET_SCORE = 15
PATIENCE_EPISODES = 100

# Reward values - balanced for better learning
REWARD_EAT_FOOD = 25.0
REWARD_GAME_OVER = -50.0
REWARD_MOVED_CLOSER = 1.0
REWARD_MOVED_AWAY = -1.5

REWARD_SURVIVAL = 0.1  # Small reward for staying alive each step
REWARD_LENGTH_BONUS = 2.0  # Bonus per body segment length