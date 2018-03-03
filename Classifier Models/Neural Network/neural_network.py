import os
import numpy as np
from sklearn.cross_validation import train_test_split
import pandas as pd
import tensorflow as tf
import sys

# Python optimisation variables
learning_rate = 0.01
epochs = 100000
batch_size = 10


def next_batch(num, data, labels):
    """
    Return a total of `num` random samples and labels.
    :param num: Number of examples
    :param data: Input
    :param labels: Output
    :return: batch of num
    """

    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


CLEAN_DATA_PATH = 'C:\\Users\\Swapnil.Walke\\MusicGenreClassifier\\clean_data.csv'
data = pd.read_csv(CLEAN_DATA_PATH)

X_data = data.iloc[:, :-1].copy()
y_data = data.iloc[:, -1:].copy()

X_data = X_data.values
labels = {
"blues":0,
"classical":1,
"country":2,
"disco":3,
"hiphop":4,
"jazz":5,
"metal":6,
"pop":7,
"reggae":8,
"rock":9
}

y_data [y_data == "blues"] = 0
y_data [y_data == "classical"] = 1
y_data [y_data == "country"] = 2
y_data [y_data == "disco"] = 3
y_data [y_data == "hiphop"] = 4
y_data [y_data == "jazz"] = 5
y_data [y_data == "metal"] = 6
y_data [y_data == "pop"] = 7
y_data [y_data == "reggae"] = 8
y_data [y_data == "rock"] = 9


X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25, random_state=333, stratify=data.iloc[:, -1:])

arr = []
temp = y_train.values
for i in range(0,len(y_train.values)):
    arr.append(temp[i][0])
arr = np.array(arr)
# print(arr.ndim)
# print(arr.size)
y_train = arr

arr = []
temp = y_test.values
print(y_test.shape)
for i in range(0, len(y_test.values)):
    arr.append(temp[i][0])
arr = np.array(arr)
print(arr.ndim)
print(arr.size)
y_test = arr

x = tf.placeholder(tf.float32, [None, 28])
y = tf.placeholder(tf.int32, [None])


# Fully connected layer
layer1 = tf.contrib.layers.fully_connected(X_train, 100, tf.nn.tanh)
layer2 = tf.contrib.layers.fully_connected(layer1, 85, tf.nn.tanh)
layer3 = tf.contrib.layers.fully_connected(layer2, 90, tf.nn.tanh)
layer4 = tf.contrib.layers.fully_connected(layer3, 10, tf.nn.relu)

# Define a loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y,
                                                                    logits = layer4))

# Define an optimizer
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)


# Convert logits to label indexes
correct_pred = tf.equal(tf.argmax(layer4, 1), tf.cast(y, tf.int64))

# Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.set_random_seed(1234)
sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(1500):
        print('EPOCH', i)
        _, accuracy_val = sess.run([train_op, loss], feed_dict={x: X_train, y: y_train})
        if i % 10 == 0:
            print("Loss: ", accuracy_val)
        print('DONE WITH EPOCH')
print(sess.run(accuracy, feed_dict={x: X_train, y: y_train}))


# predicted = sess.run([accuracy], feed_dict={x: X_test})[0]
# print(predicted)
