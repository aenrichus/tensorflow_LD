"""
Author:     Henry Wolf
Twitter:    @chaoticneural

Date:       2017.01.11

Notes:      This is a simple convolutional neural network (CNN) that was created for the UConn NBL J-Term 2017 session
            on deep learning. It trains on images of words and nonwords and learns their lexicality.
"""

import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image

# %% First, need some data. In this example, we are training a network to tell the difference between images of words
# %% and images of nonwords (lexical decision). We will use the same words as used in one simulation of baboon lexical
# %% decision from Deep Learning of Orthographic Representations in Baboons (Hannagan et al., 2014).

print("Loading files...")

data = pd.read_table('baboon.txt', header=None)  # the complete training data
data_test = pd.read_table('baboon_test.txt', header=None)  # only the unique items in the training data
data_mini = pd.read_table('baboon_mini.txt', header=None)  # subset of 100 of the items in the training data

words = data[0]
words_test = data_test[0]
words_mini = data_mini[0]

labels = data[2]
labels_test = data_test[2]
labels_mini = data_mini[2]

# %% We then load the images. For this, we will use a function. This would normally go in a separate file, because it
# %% can be reused in many different models. We create a placeholder array full of zeros for the images. Then, we loop
# %% through each of the words to load the correct image data using the function.


def load_image(word_file):
    img = Image.open(word_file)  # opens the image file
    img.load()  # loads the image into memory
    img_data = np.asarray(img, dtype="float")  # converts the image into an array of values
    flat_data = img_data.ravel()  # flattens the image into a vector
    flat_data_scaled = flat_data / 255  # converts each value to between 0 and 1
    return flat_data_scaled


print("Loading images...")

words_input = np.zeros((len(data), 32*64*1))
words_test_input = np.zeros((len(data_test), 32*64*1))
words_mini_input = np.zeros((len(data_mini), 32*64*1))

for i in range(len(words)):
    words_input[i] = load_image('images/' + words[i] + '.tiff')

for i in range(len(words_test)):
    words_test_input[i] = load_image('images/' + words_test[i] + '.tiff')

for i in range(len(words_mini)):
    words_mini_input[i] = load_image('images/' + words_mini[i] + '.tiff')

# %% We also want to convert the labels one-hot values that look like [1, 0] and [0, 1] instead of just 0 and 1. We use
# %% the eye function from numpy in a way that is very similar to loading the images.

print("Loading labels...")

labels_output = np.zeros((len(data), 2))
labels_test_output = np.zeros((len(data_test), 2))
labels_mini_output = np.zeros((len(data_mini), 2))

for i in range(len(labels)):
    labels_output[i] = np.eye(2)[labels[i]]

for i in range(len(labels_test)):
    labels_test_output[i] = np.eye(2)[labels_test[i]]

for i in range(len(labels_mini)):
    labels_mini_output[i] = np.eye(2)[labels_mini[i]]

# %% Next, we start the interactive session to build the graph. This is where we design the network. We use placeholders
# %% for the input and output, because the word or nonword being trained on will change during each trial.

sess = tf.InteractiveSession()

# %% The input placeholder (x) needs a type (float) and a shape, which is the dimensions of the image. Here we use
# %% images that are 64 pixels wide, 32 pixels high, and have 1 color values (grayscale) for each pixel. We then need to
# %% reshape the image data, so that it is seen as an image and not one long 1 pixel high line.

x = tf.placeholder("float", shape=[None, 32*64*1])
x_image = tf.reshape(x, [-1, 32, 64, 1])

# %% The output placeholder (y_) needs a type (float) and a shape. The shape is the number of nodes needed to represent
# %% the labels. In this case, the number of labels is 2 (word, nonword).

y_ = tf.placeholder("float", shape=[None, 2])

# %% Weights and biases need to be initialized for each level. We also want to perform some calculations at each level.
# %% It is easier to make functions to do these things. They would usually go at the top of the file. The choices made
# %% regarding the distribution may sample from will have to do with your activation function.


def weight_variable(shape):  # function to initialize weights with noise
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):  # function to initialize with zero biases
    initial = tf.zeros(shape)
    return tf.Variable(initial)


def convolve(x_in, weights):  # function for convolution with output and input the same size
    return tf.nn.conv2d(x_in, weights, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x_in):  # function for max pooling over 2x2 blocks
    return tf.nn.max_pool(x_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# %% Now we are ready to implement the each layer of the model. When we initialize the weights, we need to tell the
# %% model the the size of the patch in the previous layer to look at, the number of input channels, and the number of
# %% output channels. The bias variables only need the number of output channels.

# %% It is generally better to have a smaller input patch and more layers, but there may be theoretical reasons for
# %% using a larger input patch. We use 3x3 patches in this example.

# %% LAYER 1 ::: Takes 1 image (shape 32x64x1) and results in 32 feature maps (shape 32x16)

W_conv1 = weight_variable([3, 3, 1, 32])  # 3x3 patch, 1 input channels (colors), 32 output channels (feature maps)
b_conv1 = bias_variable([32])  # 32 output channels (feature maps)

h_conv1 = convolve(x_image, W_conv1) + b_conv1  # perform convolution of image on weights using function and add biases
h_act1 = tf.nn.relu(h_conv1)  # apply the ReLU activation function to get activation at this level
h_pool1 = max_pool(h_act1)  # perform max pooling using the function, which changes the shape to 32x16

# %% LAYER 2 ::: Takes 32 feature maps (shape 32x16) and results in 64 feature maps (shape 16x8)

W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = convolve(h_pool1, W_conv2) + b_conv2
h_act2 = tf.nn.relu(h_conv2)
h_pool2 = max_pool(h_act2)

# %% LAYER 3 ::: Takes 64 feature maps (shape 16x8) and results in 128 feature maps (shape 8x4)

W_conv3 = weight_variable([3, 3, 64, 128])
b_conv3 = bias_variable([128])

h_conv3 = convolve(h_pool2, W_conv3) + b_conv3
h_act3 = tf.nn.relu(h_conv3)
h_pool3 = max_pool(h_act3)

# %% DENSELY (FULLY) CONNECTED LAYER ::: Takes 128 feature maps (shape 8x4) and results in 500 hidden units (nodes).
# %% Before these calculations, we flatten the feature maps using reshape. We also use matrix multiplication, instead of
# %% convolution. There is no max pooling at this level either.

h_flat3 = tf.reshape(h_pool3, [-1, 8 * 4 * 128])  # flatten the feature maps

W_fc1 = weight_variable([8 * 4 * 128, 500])  # 8x4x128 for the 128 flattened feature maps, 500 output channels (nodes)
b_fc1 = bias_variable([500])  # 500 output channels (nodes)

h_mm1 = tf.matmul(h_flat3, W_fc1) + b_fc1  # perform matrix multiplication and add biases
h_fc1 = tf.nn.relu(h_mm1)

# %% READOUT (OUTPUT) LAYER ::: Takes 500 nodes and results in a decision of size 2 (word or nonword)

W_fc2 = weight_variable([500, 2])  # 500 input channels (nodes), 2 output channels (word or nonword)
b_fc2 = bias_variable([2])  # 2 output channels (word or nonword)

h_mm2 = tf.matmul(h_fc1, W_fc2) + b_fc2
y = tf.nn.softmax(h_mm2)  # apply SoftMax to activations for the final output


# %% Now that we have a complete model from input to output, we need to determine how the weights will be changed during
# %% training. There are many ways to do this, but we will use cross entropy and a learning rate of 1e-4.

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))  # our cost function is cross entropy between target and prediction
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # training should minimize cross entropy

# %% We also need some code to determine if the model is making accurate decisions. First, we will take the node with
# %% the highest activation in the prediction (y) and the label (y_) and see if they are the same. We then convert this
# %% to a numerical (float) value and take the mean of all of the input items being tested on.

high_pred = tf.argmax(y, 1)  # gives the node with highest activation from the model
high_real = tf.argmax(y_, 1)  # gives the node that should have the highest activation from the label

correct_prediction = tf.equal(high_pred, high_real)  # checks if they are equal and gives a boolean (TRUE or FALSE)
correct_float = tf.cast(correct_prediction, "float")  # converts to float (TRUE = 1, FALSE = 0)
accuracy = tf.reduce_mean(correct_float)  # calculates the mean accuracy of all items in the test set

# %% After the graph is complete, we need to initialize all of the variables.
sess.run(tf.initialize_all_variables())

# %% Here is where we actually input the data into the graph. We test after training after a certain number of input
# %% items by running the eval on the accuracy variable in the graph. We feed this function a subset of the training
# %% data with by feeding it a dictionary of values to use. We then print out the result. We train on each input item in
# %% the same manner, but by running train_step.

print("Starting training...")

for i in range(len(words)):
    if i % 100 == 0:  # how frequently to check the accuracy
        test_accuracy = accuracy.eval(feed_dict={x: words_mini_input, y_: labels_mini_output})
        print("Step %d accuracy %g" % (i, test_accuracy))

    train_step.run(feed_dict={x: [words_input[i]], y_: [labels_output[i]]})

# %% After training is complete, we test on the entire set of unique input items to determine the final accuracy.

print("Final accuracy %g" % accuracy.eval(feed_dict={x: words_test_input, y_: labels_test_output}))
