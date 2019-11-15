from .hyperparameters import *
from .key_manipulation import *
from .data_processor import *
from .stack_frames import *
from .Entities import * 
from .find_process import *
from .DQNetwork import *
import win32gui
from PIL import Image, ImageFilter, ImageGrab # To take screenshots
import cv2 # To convert frame from BGR to grayscale
#from skimage import transform # Helps in pre-processing frames
import warnings # This ignore all the warning messages that are normally printed during the training because of skimage
import numpy as np
import time

warnings.filterwarnings('ignore')

#LIVING_REWARD = 10
#DEATH_REWARD = -500
LIVING_REWARD = 0.2
DEATH_REWARD = -10

class Agent:
    # First link to the TH10 process
    def __init__(self):
        # Getting the process name and port id of TH10
        pid = find_process('th10.exe')
        # Depending on the name of the game window, this argument below must be set to that
        self.win_handle = win32gui.FindWindow(None,'Touhou Eastern Wind God Chronicles ~ Mountain of Faith v1.00a')
        self.pid = pid
        self.dataReader = DataReader(pid)
        self.player = self.dataReader.player_info()

    # Method to capture and return a screen shot
    # Pass in the window handle from dataReader.process_handle
    def get_state(self, win_handle):
        # Set the window to the foreground
        # Need to pass in win handle and not process handle
        win32gui.SetForegroundWindow(self.win_handle)
        window_bbox = win32gui.GetWindowRect(self.win_handle)
        # Return the screenshot
        image = ImageGrab.grab(window_bbox)
        # Get dimensions of the image
        width, height = image.size
        # This crop removes the fps and life and score etc 
        state = image.crop((0, 0, 3*width/4, height-4))
        #state.show()
        return image.convert("L")

    # The actions that the agent can take
    def actions(self, action_value):
        # Always press the Z key
        press_key(DIR_Z)

        if action_value == 0:
            return
        elif action_value == 1:
            press_key(DIR_LEFT)
            time.sleep(0.1)
            release_key(DIR_LEFT)
        elif action_value == 2:
            press_key(DIR_RIGHT)
            time.sleep(0.1)
            release_key(DIR_RIGHT)
        elif action_value == 3:
            press_key(DIR_UP)
            time.sleep(0.1)
            release_key(DIR_UP)
        elif action_value == 4:
            press_key(DIR_DOWN)
            time.sleep(0.1)
            release_key(DIR_DOWN)
        elif action_value == 5:
            press_key(DIR_LEFT)
            press_key(DIR_UP)
            time.sleep(0.1)
            release_key(DIR_LEFT)
            release_key(DIR_UP)
        elif action_value == 6:
            press_key(DIR_LEFT)
            press_key(DIR_DOWN)
            time.sleep(0.1)
            release_key(DIR_LEFT)
            release_key(DIR_DOWN)
        elif action_value == 7:
            press_key(DIR_RIGHT)
            press_key(DIR_UP)
            time.sleep(0.1)
            release_key(DIR_RIGHT)
            release_key(DIR_UP)
        elif action_value == 8:
            press_key(DIR_RIGHT)
            press_key(DIR_DOWN)
            time.sleep(0.1)
            release_key(DIR_RIGHT)
            release_key(DIR_DOWN)

        # The "Do Nothing" action for action_value == 9
        
        # An additional action_value of 9 that the DQN will not be outputting
        # This is to allow for the restart of game in experience replay
        elif action_value == 10:
            press_key(KEY_ENTER)
            time.sleep(0.02)
            release_key(KEY_ENTER)
            press_key(DIR_UP)
            time.sleep(0.02)
            release_key(DIR_UP)
            press_key(DIR_UP)
            time.sleep(0.02)
            release_key(DIR_UP)
            press_key(KEY_ENTER)
            time.sleep(0.02)
            release_key(KEY_ENTER)

    # A take action function
    def take_action(self, action_value):
        prev_life = self.player.life
        
        reward = 0
        # Agent takes action based on action value passed in
        self.actions(action_value)
        # Get current player info and return Player object
        self.player = self.dataReader.player_info()
        current_life = self.player.life
        
        if current_life < prev_life:
            reward = DEATH_REWARD
        else:
            reward = LIVING_REWARD

        #if self.player.isAlive():
        #    reward = LIVING_REWARD
        #else:
        #    reward = DEATH_REWARD
        #curr_life = self.player.life
        #if curr_life > 0:
        #    reward = LIVING_REWARD
        #else:
        #    reward = DEATH_REWARD

        #if action_value == 9:
        #    reward = 0
        #state = get_state(self.dataReader.process_handle)
        # Pre-processing the state
        #preprocessed_state = preprocess_state(state)
        # Returning only reward here, removed returning the state
        return reward

    # Function to preprocess state
    def preprocess_state(self, state):
        # Converting the state to grayscale
        state = np.array(state)

        #image_gray = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = np.resize(state, (92,92))
        image_gray = np.array(state)


        # Normalize pixel values

        normalized_state = image_gray/255.0

        # Can consider cropping right side of frame which is just info for the player

        # Resize to 92x92
        #normalized_state = transform.resize(normalized_state, [92,92])
        
        #print("state size: " + str(normalized_state.shape) + '\n\n')
        #normalized_state = normalized_state.reshape(normalized_state.shape[:2])
        print("state size: " + str(normalized_state.shape) + '\n\n')
        return normalized_state




