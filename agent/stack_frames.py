import numpy as np 
from collections import deque

def stack_frames(stack, new_state, is_new_episode):
    # Stacking a max of 4 frames
    stack_size = 4

    if is_new_episode == True:
        # Initialize a new stack of frames using a deque with max size of 4
        # Each frame inside is of dimensions 92x92, the same as the size we re-sized to
        stack = deque([np.zeros((92,92), dtype = int) for i in range(stack_size)], maxlen = 4)

        # Since we are starting a new game(episode), we add the initial state 4 times
        for j in range(4):
            stack.append(new_state)

        # Stack the frames
        stacked_states = np.stack(stack, axis = 2)

    else:
        stack.append(new_state)

        #for idx,state in enumerate(stack):
            #print("state is: " + str(state.shape) + '\n')

        # Stack the frames
        stacked_states = np.stack(stack, axis=2)

    return stacked_states, stack

