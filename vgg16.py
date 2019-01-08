########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import tensorflow as tf
import numpy as np


# from scipy.misc import imread, imresize
# from imagenet_classes import class_names


class VGG16:
    def __init__(self, inputs, regularizer, name="", flag=False):  # flag: True--使用全连接层
        self.name = name
        self.image = inputs
        self.regularizer = regularizer
        self.convlayers()
        if flag:
            self.fc_layers()
        

    def convlayers(self):
        self.parameters = []

        # conv1_1
        with tf.variable_scope('conv1_1' + self.name, reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable('weights', shape=[3, 3, 3, 64], trainable=True, regularizer=self.regularizer,
                                     initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(self.image, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', shape=[64], trainable=True,
                                     initializer=tf.zeros_initializer(dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.variable_scope('conv1_2' + self.name, reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable('weights', shape=[3, 3, 64, 64], trainable=True, regularizer=self.regularizer,
                                     initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', shape=[64], trainable=True,
                                     initializer=tf.zeros_initializer(dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool1')

        # conv2_1
        with tf.variable_scope('conv2_1' + self.name, reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable('weights', shape=[3, 3, 64, 128], trainable=True, regularizer=self.regularizer,
                                     initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', shape=[128], trainable=True,
                                     initializer=tf.zeros_initializer(dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.variable_scope('conv2_2' + self.name, reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable('weights', shape=[3, 3, 128, 128], trainable=True, regularizer=self.regularizer,
                                     initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', shape=[128], trainable=True,
                                     initializer=tf.zeros_initializer(dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool2')

        # conv3_1
        with tf.variable_scope('conv3_1' + self.name, reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable('weights', shape=[3, 3, 128, 256], trainable=True, regularizer=self.regularizer,
                                     initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', shape=[256], trainable=True,
                                     initializer=tf.zeros_initializer(dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.variable_scope('conv3_2' + self.name, reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable('weights', shape=[3, 3, 256, 256], trainable=True, regularizer=self.regularizer,
                                     initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', shape=[256], trainable=True,
                                     initializer=tf.zeros_initializer(dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.variable_scope('conv3_3' + self.name, reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable('weights', shape=[3, 3, 256, 256], trainable=True, regularizer=self.regularizer,
                                     initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', shape=[256], trainable=True,
                                     initializer=tf.zeros_initializer(dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool3')

        # conv4_1
        with tf.variable_scope('conv4_1' + self.name, reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable('weights', shape=[3, 3, 256, 512], trainable=True, regularizer=self.regularizer,
                                     initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', shape=[512], trainable=True,
                                     initializer=tf.zeros_initializer(dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.variable_scope('conv4_2' + self.name, reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable('weights', shape=[3, 3, 512, 512], trainable=True, regularizer=self.regularizer,
                                     initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', shape=[512], trainable=True,
                                     initializer=tf.zeros_initializer(dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.variable_scope('conv4_3' + self.name, reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable('weights', shape=[3, 3, 512, 512], trainable=True, regularizer=self.regularizer,
                                     initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', shape=[512], trainable=True,
                                     initializer=tf.zeros_initializer(dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool4')

        # conv5_1
        with tf.variable_scope('conv5_1' + self.name, reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable('weights', shape=[3, 3, 512, 512], trainable=True, regularizer=self.regularizer,
                                     initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', shape=[512], trainable=True,
                                     initializer=tf.zeros_initializer(dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.variable_scope('conv5_2' + self.name, reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable('weights', shape=[3, 3, 512, 512], trainable=True, regularizer=self.regularizer,
                                     initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', shape=[512], trainable=True,
                                     initializer=tf.zeros_initializer(dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.variable_scope('conv5_3' + self.name, reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable('weights', shape=[3, 3, 512, 512], trainable=True, regularizer=self.regularizer,
                                     initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', shape=[512], trainable=True,
                                     initializer=tf.zeros_initializer(dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool4')

    def fc_layers(self):
        # fc1
        with tf.variable_scope('fc1' + self.name, reuse=tf.AUTO_REUSE) as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.get_variable('weights', shape=[shape, 4096], trainable=True, regularizer=self.regularizer,
                                    initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-1))
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                               trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.variable_scope('fc2' + self.name, reuse=tf.AUTO_REUSE) as scope:
            fc2w = tf.get_variable('weights', shape=[4096, 4096], trainable=True, regularizer=self.regularizer,
                                    initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-1))
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                               trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]


    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            try:
                print(i, k, np.shape(weights[k]))
                sess.run(self.parameters[i].assign(weights[k]))
            except IndexError:
                continue

    def get_poolled_features(self):
        return self.pool5

    def get_pool3(self):
        return self.pool3

    def get_pool4(self):
        return self.pool4
    
    def get_fc(self):
        return self.fc2

