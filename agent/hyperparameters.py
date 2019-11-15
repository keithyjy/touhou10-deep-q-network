# The hyperparameters for the model

possible_actions = [1,2,3,4,5,6,7,8,9]
state_size = [92,92,4] # as we are using a stack of 4 frames of 84x84 each
learning_rate = 0.00001 # Arbitrary learning rate here
total_episodes = 300
# max steps in an ep?
max_steps = 500
batch_size = 8 # The batch size from experience replay to run batch optimization/gradient descent on, 64 previously
pretrain_size = batch_size # Pretraining 64 episodes to put into memory in order to do batch optimization
start_explore_rate = 1 # Exploration rate at the start set to 1 to randomize initial actions to learn
min_explore_rate = 0.01 # Minimum exploration rate
decay_rate = 0.0001 # The exponential decay rate for the exploration probability

# For Q learning
gamma = 0.95 # Set a discount rate of 0.95 so that agent will tend to long term rewards

# For experience replay
# Need to initialize Memory to store at least batch_size number of experiences at the startz68z
max_mem_size = 30000 # Total number of experiences that can be kept in memory
 
