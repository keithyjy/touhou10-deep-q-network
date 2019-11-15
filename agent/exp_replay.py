from collections import deque
import random
from .hyperparameters import *
from .key_manipulation import *
from .data_processor import *
from .stack_frames import *
from .Entities import *
from .agent import *
from .find_process import *
from .DQNetwork import *
import numpy as np
import copy


# Create Memory object in Train_Model and when it is playing, append experiences until we have sufficient
# experiences to take a random batch to train the model
class Memory:
    def __init__(self, max_size):
        self.storage = deque(maxlen=max_size)

    # An experience is a tuple of (state, action, reward, next_state, isEpisodeFinished)
    def add(self, experience):
        self.storage.append(experience)

    def sample(self, batch_size):
        # Sample batch_size number of experiences randomly without replacement from Memory
        indices_to_extract = np.random.choice(len(self.storage), size=batch_size, replace=False)

        return [self.storage[i] for i in indices_to_extract]

    def instantiate(self):
        agent = Agent()
        for i in range(pretrain_size):
            # For the first step taken
            if i == 0:
                # Get the initial state
                state = agent.get_state(agent.win_handle)
                # Preprocess the frame 
                state = agent.preprocess_state(state)
                stack_size = 4
                stack = deque([np.zeros((92, 92), dtype=np.int) for i in range(stack_size)], maxlen=4)
                # Stack the states
                state, stacked_frames = stack_frames(stack, state, True)

            # Randomly pick some action
            action = random.choice(possible_actions)

            # Retrieve the reward
            reward = agent.take_action(action)

            # Check if the episode is finished based on whether the agent is still alive
            is_episode_finished = not agent.player.isAlive()

            # If the agent is dead
            if is_episode_finished:
                # Make the next state an matrix of zeros corresponding to dimensions of previous state
                next_state = np.zeros(state.shape)

                # Create a copy and store the previous stacked frames first
                prev_stacked_frames = copy.deepcopy(stacked_frames)
                # Add this experience to the storage
                self.add((np.stack(prev_stacked_frames,axis=2), action, reward, next_state, is_episode_finished))

                # Restart the episode
                # Action_value == 9 corresponds to restarting the game (ESC + R)
                #agent.take_action(9)

                # We need to get the initial state again for the next iteration (episode)
                state = agent.get_state(agent.win_handle)
                # Preprocess the frame
                state = agent.preprocess_state(state)
                stack = deque([np.zeros((92, 92), dtype=np.int) for i in range(stack_size)], maxlen=4)
                # Stack the states
                state, stacked_frames = stack_frames(stacked_frames, state, True)

            else:
                # Since episode is not yet finished, we get the next state with a screen capture
                next_state = agent.get_state(agent.win_handle)
                # Preprocess the frame
                state = agent.preprocess_state(state)
                # Create a copy and store the previous stacked frames first
                prev_stacked_frames = copy.deepcopy(stacked_frames)
                # Stack it with the 3 previous frames, isEpisodeFinished is set to False
                next_state, stacked_frames = stack_frames(stacked_frames, state, False)

                # Add this experience to storage
                self.add((np.stack(prev_stacked_frames,axis=2), action, reward, next_state, is_episode_finished))

                # Update our state variable to hold the next state
                state = next_state



