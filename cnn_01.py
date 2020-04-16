# coding=utf-8

import tensorflow as tf
import pickle
import random

# defines the size of the batch.
BATCH_SIZE = 40
# one channel in our grayscale images.
NUM_CHANNELS = 1
# The random seed that defines initialization.
SEED = 42223

IMAGE_SIZE = 64

NUM_LABELS = 2

import numpy as np
from PIL import Image
import os

class_num = 2
img1_num = 10000
img2_num = 10000
img1_test_num = 1000
img2_test_num = 1000
train_num = img1_num + img2_num
test_total_num = img1_test_num + img2_test_num

image_height = 64
image_width = 64
image_channle = 1

data_path1 = '/home/skywalker/桌面/cnn/1.1'
data_path2 = '/home/skywalker/桌面/cnn/2.1'
#data_path3 = '/Users/xinruyue/Desktop/python_test/3'

def ImageToMatrix(filename):
    im = Image.open(filename)
    width,height = im.size
    im = im.convert("L")
    data = im.getdata()
    data = np.matrix(data)
    #new_data = np.reshape(data,(width,height))
    new_data = np.reshape(data,(height,width))
    return new_data

for root, dirs, files in os.walk(data_path1):
    f1 = files
img1 = []
for each in f1:
    path = os.path.join(data_path1,each)
    img1.append(ImageToMatrix(path))
print(len(img1))

for root, dirs, files in os.walk(data_path2):
    f2 = files
img2 = []
for each in f2:
    path = os.path.join(data_path2,each)
    img2.append(ImageToMatrix(path))
print(len(img2))

'''
for root, dirs, files in os.walk(data_path3):
    f3 = files
img3 = []
for each in f3:
    path = os.path.join(data_path3,each)
    img3.append(ImageToMatrix(path))
'''

dummy_train_data = img1[:img1_num] + img2[:img2_num]

dummy_train_labels = np.zeros((train_num,class_num))
dummy_train_labels[:img1_num, 0 ] = 1
dummy_train_labels[img2_num:, 1 ] = 1
#dummy_train_labels[3000:, 2 ] = 1

data_label_pair = list(zip(dummy_train_data, dummy_train_labels))
random.shuffle(data_label_pair)

train_data_temp = list(zip(*data_label_pair))[0]
train_labels_temp = list(zip(*data_label_pair))[1]
print(len(train_data_temp))
train_data = np.array(train_data_temp).reshape((train_num,image_height,image_width,image_channle)).astype(np.float32)
train_labels = np.array(train_labels_temp)

train_size = train_labels.shape[0]

# prepare test datas and labels
dummy_test_data = img1[img1_num:img1_num + img1_test_num] + img2[img2_num:img2_num + img2_test_num]

dummy_test_labels = np.zeros((test_total_num,class_num))
dummy_test_labels[:img1_test_num, 0 ] = 1
dummy_test_labels[img2_test_num:, 1 ] = 1
#dummy_test_labels[400:, 2 ] = 1

test_data_label_pair = list(zip(dummy_test_data, dummy_test_labels))
random.shuffle(test_data_label_pair)

test_data_temp = list(zip(*test_data_label_pair))[0]
test_labels_temp = list(zip(*test_data_label_pair))[1]
print len(test_data_temp)
test_data = np.array(test_data_temp).reshape((test_total_num,image_height,image_width,image_channle)).astype(np.float32)
test_labels = np.array(test_labels_temp)

train_data_node = tf.placeholder(
  tf.float32,
  shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
train_labels_node = tf.placeholder(tf.float32,
                                   shape=(BATCH_SIZE, NUM_LABELS))

test_data_node = tf.constant(test_data)

# parameter initialize.
conv1_weights = tf.Variable(
  tf.truncated_normal([5, 5, NUM_CHANNELS, 64],
                      stddev=0.1,
                      seed=SEED))

conv1_biases = tf.Variable(tf.zeros([64]))

conv2_weights = tf.Variable(
  tf.truncated_normal([5, 5, 64, 64],
                      stddev=0.1,
                      seed=SEED))

conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))

conv3_weights = tf.Variable(
    tf.truncated_normal([5,5,64,128],
        stddev=0.1,
        seed=SEED
        )
)
conv3_biases = tf.Variable(tf.constant(0.1, shape=[128]))
fc1_weights = tf.Variable(
  tf.truncated_normal([int(IMAGE_SIZE / 8 * IMAGE_SIZE / 8 * 128), 256],
                      stddev=0.1,
                      seed=SEED))

fc1_biases = tf.Variable(tf.constant(0.1, shape=[256]))

fc2_weights = tf.Variable(
  tf.truncated_normal([200, NUM_LABELS],
                      stddev=0.1,
                      seed=SEED))

fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

fc3_weights = tf.Variable(
  tf.truncated_normal([256, 200],
                      stddev=0.1,
                      seed=SEED))
fc3_biases = tf.Variable(tf.constant(0.1, shape=[200]))


def model(data, train=False):
    """The Model definition."""
    conv1 = tf.nn.conv2d(data,
                      conv1_weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')

    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    pool1 = tf.nn.max_pool(relu1,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')
    conv2 = tf.nn.conv2d(pool1,
                      conv2_weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    pool2 = tf.nn.max_pool(relu2,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')
    conv3 = tf.nn.conv2d(pool2,
                       conv3_weights,
                       strides=[1,1,1,1],
                       padding='SAME'
                       )
    relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
    pool3 = tf.nn.max_pool(relu3,
                           ksize=[1,2,2,1],
                           strides=[1,2,2,1],
                           padding='SAME'
                           )
    pool_shape = pool3.get_shape().as_list()
    reshape = tf.reshape(
      pool3,
      [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)

    if train:
        hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    hidden1 = tf.matmul(hidden, fc3_weights) + fc3_biases
    return tf.matmul(hidden1, fc2_weights) + fc2_biases

def error_rate(predictions, labels):
    """Return the error rate and confusions."""
    correct = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    total = predictions.shape[0]

    error = 100.0 - (100 * float(correct) / float(total))

    return error

logits = model(train_data_node, True)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
  logits=logits, labels=train_labels_node))

regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases) +
                tf.nn.l2_loss(fc3_weights) + tf.nn.l2_loss(fc3_biases))
loss += 5e-4 * regularizers

batch = tf.Variable(0)

learning_rate = tf.train.exponential_decay(
  0.00001,                # Base learning rate.
  batch * BATCH_SIZE,  # Current index into the dataset.
  1000,          # Decay step.
  0.9,                # Decay rate.
  staircase=True)

optimizer = tf.train.MomentumOptimizer(learning_rate,
                                       0.9).minimize(loss,
                                                     global_step=batch)

train_prediction = tf.nn.softmax(logits)
test_prediction = tf.nn.softmax(model(test_data_node))

#train
s = tf.InteractiveSession()

s.as_default()

tf.initialize_all_variables().run()

steps = int(train_size / BATCH_SIZE)
num = 10000
for step in range(steps):

    print(step)
    offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
    print(offset)
    batch_data = train_data[offset:(offset + BATCH_SIZE), :, :, :]
    batch_labels = train_labels[offset:(offset + BATCH_SIZE)]

    feed_dict = {train_data_node: batch_data,
                train_labels_node: batch_labels}

    _, l, lr, predictions = s.run(
      [optimizer, loss, learning_rate, train_prediction],
      feed_dict=feed_dict)

    error = error_rate(predictions, batch_labels)

    print('Step %d of %d' % (step, steps))
    print('Mini-batch loss: %.5f Error: %.5f Learning rate: %.5f' % (l, error, lr))

test_error = error_rate(test_prediction.eval(), test_labels)
print(test_prediction.eval())
print(test_labels)
print('Test error: %.1f%%' % test_error)
