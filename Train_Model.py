from agent.DQNetwork import * 
from agent.hyperparameters import *
from agent.data_processor import *
from agent.Entities import *
from agent.exp_replay import *
from agent.find_process import *
from agent.stack_frames import *
from agent.key_manipulation import *
from agent.agent import *
import numpy as np
import keras
import h5py
import copy
from PIL import ImageChops
import time

# import tensorflow as tf

#tf.disable_v2_behavior()
# Method to check for the end of the game, takes in the previous stacked frames
# Method grabs the next screenshot and compares the last 3 screenshots in the stack for equality
def checkIfGameEnded(stacked_f):

    # Get two frames separated by 0.1s
    # Add a delay to skip some frames so we don't get identical frames   
    next_state_1 = agent.get_state(agent.win_handle)
    time.sleep(0.1)
    next_state_2 = agent.get_state(agent.win_handle)
    # .differece() finds the difference between the two images and getbbox()
    # calculates the bounding box of the non-zero regions in the resulting image
    # If the original two images are identical, all pixels in the difference image will be 0 and the bbox() function returns None
    return ImageChops.difference(next_state_1,next_state_2).getbbox() is None


# Method to predict the action to be taken or whether action should be a random action
# Balancing between exploration and exploitation
def action_predictor(decay_step, decay_rate, state):
    # First we generate a random number between [0,1)
    rand_num = random.random()

    # Generate a threshold where we take random actions if our value is below the threshold
    random_action_threshold = min_explore_rate + (start_explore_rate - min_explore_rate)*np.exp(-1*decay_rate*decay_step)

    # We take a random action if the random number is under the threshold (Exploration)
    if rand_num < random_action_threshold:
        # Generate a random action
        action = random.choice(possible_actions)
    # If not, we use the DQN to predict the best action for us
    else:
        # We generate the output vector containing each output value of the DQN
        # sess.run runs the graph till DQNetwork using the inputs in
        # the form of key:value mapping from feed_dict as the input, reshape to (1x92x92x4)
        #output_vector = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs:state.reshape((1, 92, 92, 4))})
        state_input = np.expand_dims(state,axis=0)
        output_vector = DQNetwork.model.predict(state_input)

        # Taking the indexz62z42z68z2 of the largest output value from the vector
        best_val_idx = np.argmax(output_vector)
        # Get the corresponding action
        action = possible_actions[best_val_idx]

    # Can return explore probability for printing if needed
    return action

if __name__ == '__main__':
    # Instantiate the DQNetwork
    DQNetwork = DQNetwork(state_size, learning_rate)
    #DQNetwork.model.load_weights("DQN.h5")
    # tf.train.Saver() would allow us to save our model at different checkpoints
    #saver = tf.train.Saver()
    #print("test") #Debug

    # Initialize number of decay steps taken for the exploration
    decay_step = 0
    # Initialize the memory storage and let it be instantiated with an initial batch of size pretrain_size
    memory = Memory(max_mem_size)
    memory.instantiate()

    stack_size = 4

    # Initialize an array for the Q values per episodes and for the rewards per 5 episodes
    q_list = []
    rewards_list = []
    # Temporary list to store rewards up till 5 episodes to take the avg to store in rewards_list
    ep_rewards = []
    # Temp list to store rewards for each episode to take avg 
    q_per_ep = []

    for episode in range(1,total_episodes+1):

        # Initialize the number of steps taken in the episode and the list of rewards obtained
        step = 0
        rewards = []
        # Initialize the target Q values list
        targetQ = []

        # Create a new agent and restart the game
        agent = Agent()
        # Keep track of the number of lives
        lives = agent.player.life

        # Get the initial state
        state = agent.get_state(agent.win_handle)
        # Preprocess the frame
        state = agent.preprocess_state(state)

        stack = deque([np.zeros((92, 92), dtype=int) for i in range(stack_size)], maxlen=4)
        # Stack the states
        state, stacked_frames = stack_frames(stack, state, True)

        while step < max_steps:
            # Increment number of steps and decay steps
            step = step + 1
            decay_step = decay_step + 1

            # We first predict whether we should take an action through
            # exploration or exploitation and also the exact action to take
            action = action_predictor(decay_step, decay_rate, state)
            # Carry out the action and obtain the reward and next state
            # Release Z to evaluate if state ended
            release_key(DIR_Z)
        
            # Get the boolean if it is the end of the episode
            isEpisodeFinished = checkIfGameEnded(stacked_frames)

            # Only take action if episode is not finished
            if not isEpisodeFinished:
                reward = agent.take_action(action)
            # Append the reward to the reward list
            if reward != None:
                rewards.append(reward)
            
            
            #isEpisodeFinished = not agent.player.isAlive()
            #isEpisodeFinished = agent.player.life == 0

            # If episode has ended (agent is dead etc)
            if isEpisodeFinished:
                print("episode ended")
                # We initialize an empty frame of zeros
                next_state = np.zeros((92,92), dtype = int)

                # Create a copy and store the previous stacked frames first
                prev_stacked_frames = copy.deepcopy(stacked_frames)
                # Stack this new frame to the previous 3
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                # Sum the rewards for the episode
                episode_reward_sum = np.sum(rewards)

                # Print debugging statements here

                # Add this experience to memory
                # Add in stacked_frames which is a stack of 4 at the current time step before appending next_state, previously is state
                memory.add((np.stack(prev_stacked_frames,axis=2), action, reward, next_state, isEpisodeFinished))

                # Restart the game
                release_key(DIR_Z)
                #time.sleep(0.1)
                press_key(DIR_Z)
         
                time.sleep(0.2)
                release_key(DIR_Z)
                press_key(DIR_DOWN)
               
                time.sleep(0.2)
                release_key(DIR_DOWN)
                press_key(DIR_Z)
                
                time.sleep(0.2)
                release_key(DIR_Z)
                
                # For debugginng
                #print('Episode: {}'.format(episode),
                 #     'Total reward: {}'.format(episode_reward_sum),
                  #    'Training loss: {:.4f}'.format(loss))

                # We break out of this episode as our agent is dead
                break
            else:
                # Press Z to shoot
                press_key(DIR_Z)
                # Else we get the next state
                next_state = agent.get_state(agent.win_handle)
                # Preprocess the frame
                state = agent.preprocess_state(next_state)
                # Create a copy and store the previous stacked frames first
                prev_stacked_frames = copy.deepcopy(stacked_frames)
                # Stack it with the 3 previous frames, isEpisodeFinished is set to False
                next_state, stacked_frames = stack_frames(stacked_frames, state, False)
                # Add this experience to memory
                memory.add((np.stack(prev_stacked_frames,axis=2), action, reward, next_state, isEpisodeFinished))

                # Assign the next state to our state variable
                state = next_state

            # To train the model, we randomly sample batches of experience from memory
            batch = memory.sample(batch_size)

            # Counter to break out of training if agent is dead for too long(meaning not in game)
            dead_counter = 0
            for state, action, reward, next_state, isEpFin in batch:
                # Constantly check the status and break out if not in game anymore
             
                # If it is the terminal state, make our target the reward
                target = reward

                # Else we change it to the Q update definition
                if not isEpFin:
                    # Converting the shape to 4D for input
                    next_state_input = np.expand_dims(next_state,axis=0)
                    # Predicting the future discounted reward
                    target = reward + gamma*np.amax(DQNetwork.model.predict(next_state_input)[0])
                    # Append the Q value for that step
                    q_per_ep.append(target)
                # Now we want to map this future reward at the next state to the action taken at this state
                # Get the outputs from predicting the state
                # Convert the state to a 4D input for Keras
                state_input = np.expand_dims(state,axis=0)
                target_mapping = DQNetwork.model.predict(state_input)
                # Replacing the index of the action with the future reward we calculated
                # Index at action-1 because action is from 1 to 8
                target_mapping[0][action-1] = target

                # New addition: Take an action here corresponding to that best action predicted, just discard the reward
                #_ = agent.take_action(action)

                # Now fit the model with the 4D state as input and target mapping as output to train the model
                DQNetwork.model.fit(state_input, target_mapping, epochs = 1)
                time.sleep(0.1)

        # Every episode, take the avg of the cumulative rewards for each episode
       
        avg_reward = np.mean(ep_rewards)
        # Append to our output list
        rewards_list.append(avg_reward)
        # Re-initialise the list
        ep_rewards = []

        print("Writing to Rewards_per_ep: " + str(avg_reward))

        # Write outputs to output file for rewards per 5 episodes
        with open("Rewards_per_ep.txt", 'a') as file2:
            #for value in rewards_list:
            file2.write("%f\n" % avg_reward)

        # Append the sum of rewards in an episode to the temp array
        ep_rewards.append(np.sum(rewards,axis=0))
        # Take the average Q value obtained in that episode
        # Each episode is 500 steps
        q_rwd = np.mean(q_per_ep)
        q_list.append(q_rwd)
        # Re-initialize the list
        q_per_ep = []

        print("writing\n") #Debug
        print(ep_rewards)

        # Write these outputs of Q value to the output file every episode
        with open("Q_values_per_ep.txt", 'a') as file1:
            #for value in q_list:
            file1.write('%f\n' % q_rwd)


        if episode % 2 == 0:
            DQNetwork.model.save_weights("DQN.h5")
            # Print the loss
        
        # Write the outputs of Q and rewards to files

        # Restart the game since 500 steps are taken
        
        release_key(DIR_Z)
        
        """
        press_key(KEY_ESC)
        time.sleep(0.2)
        release_key(KEY_ESC)
        press_key(DIR_UP)
        time.sleep(0.2)
        release_key(DIR_UP)
        press_key(DIR_Z)
        time.sleep(0.2)
        release_key(DIR_Z)
        press_key(DIR_UP)
        time.sleep(0.2)
        release_key(DIR_UP)
        press_key(DIR_Z)
        time.sleep(0.2)
        release_key(DIR_Z)
        """











