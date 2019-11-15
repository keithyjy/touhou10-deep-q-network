import numpy as np

#import keras

from tensorflow.keras import *

from .hyperparameters import *

#import tensorflow as tf



class DQNetwork:

    def __init__(self, state_size, learning_rate, name = "DQNetwork"):

        self.state_size = state_size

        self.learning_rate = learning_rate



        self.model = models.Sequential()

        # Output is 90x90x32
 
        self.model.add(layers.Conv2D(32, (3,3), activation='relu',input_shape=(92,92,4)))

        # Output is 45x45x32

        self.model.add(layers.MaxPooling2D((2,2)))

        # Output is 44x44x64

        self.model.add(layers.Conv2D(64, (2,2), activation='relu'))

        # Output is 22x22x64

        self.model.add(layers.MaxPooling2D((2,2)))

        # Output is 20x20x128

        self.model.add(layers.Conv2D((128), (3,3), activation='relu'))

        # Output is 18x18x196

        self.model.add(layers.Conv2D(196, (3,3), activation = 'relu'))

        self.model.add(layers.Flatten())

        # Output is 64

        self.model.add(layers.Dense(64, activation='relu'))

        # To get the final outputs, use Softmax activation function

        # Output is 9

        self.model.add(layers.Dense(9))

        

        self.optimizer = optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(optimizer = self.optimizer, loss ='mse')

        # Apply sigmoid function to output?

        # Max pooling layers?

        # Predicted Q value

        # find out how to calculate Q

        #self.q_value = tf.reduce_sum(input_tensor=tf.multiply(self.output, self.actions), axis = 1)

        #self.loss = tf.reduce_mean(input_tensor=tf.square(self.target_q - self.q_value))

       # self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)





