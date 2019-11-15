# touhou10-deep-q-network
Deep reinforcement learning project on training an AI agent to play Touhou 10: Mountain of Faith. This project adopts ideas from DeepMind's project on Atari 2600 games and is also inspired by the following works:

* [th10-dqn](https://github.com/actumn/touhou10-dqn)
* [Deep Q-Learning on Doom](https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Deep%20Q%20Learning/Doom/Deep%20Q%20learning%20with%20Doom.ipynb)

## Game
* Touhou 10: Mountain of Faith

## Environment
* Anaconda

```
conda env create -f touhou10.yml
conda activate touhou10
# Change directory to folder containing Train_Model.py
# Run the following line after starting the game
python Train_Model.py
```
## Running the code
1. Activate the environment following steps above
2. Run the game
3. Enter a stage on "Game start" or "Practice mode"
4. Run Train_Model.py

## Description
A screenshot is taken at every step, pre-processed and stacked with three other frames. This is fed into the DQN and the agent takes either a random action with decaying probability or predicts an action to take using the model. 

Work in progress:
* Agent is currently unable to automatically restart training on some occasions due to lag times introduced the hardware.
* Use a double deep Q-network instead to avoid overestimation of Q-values
* Implement prioritized experience replay instead as some experiences may be more important to train on
* Implement the usage of 'Bombs' (special attacks) in the agent

## Files
1. Train_Model.py - Driver method with logic to play the game and train the network
2. agent.py - Carries out actions to take and gets the state at each time step
3. data_processor.py - Code to extract information from the game. Adopted from [TH10_DataReversing](https://github.com/binvec/TH10_DataReversing)
4. DQNetwork.py - Contains the network used 
5. Entities.py - Class file to create objects of entities in the game
6. exp_replay.py - Experience replay implementation to store past experiences and initialize experiences
7. find_process.py - Retrieves the process ID of the running process of Touhou 10: Mountain of Faith
8. hyperparameter.py - Contains the hyperparameters used with the model
9. key_manipulation.py - Code to make virtual key presses. Adopted from [StackOverflow answer](https://stackoverflow.com/questions/13564851/how-to-generate-keyboard-events-in-python)
10. stack_frames.py - Stacks four frames together/add the newest frame to stack and returns the stack.
11. touhou10.yml - Environment file for running the agent
12. DQN.h5 - File of weights for the network 
