import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from vgg16 import VGG16
import matplotlib.pyplot as plt
import os
import cv2
import random

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def conv2d(input_x, filter_shape, strides, padding, name, reuse, regularizer):
    assert padding in ['SAME', 'VALID']
    with tf.variable_scope(name, reuse=reuse):
        W_conv = tf.get_variable('weights', shape=filter_shape, regularizer=regularizer)
        b_conv = tf.get_variable('biases', shape=[filter_shape[-1]])
        h_conv = tf.nn.conv2d(
            input_x, W_conv, strides=strides, padding=padding)
        output = b_conv + h_conv
    return output


def conv2d_transpose(feature, filter_shape, output_size, strides, padding, name, regularizer):
    assert padding in ['SAME', 'VALID']
    with tf.variable_scope(name):
        W_conv = tf.get_variable('weights', shape=filter_shape, regularizer=regularizer)
        b_conv = tf.get_variable('biases', shape=[filter_shape[-2]])
        ans_conv = tf.nn.conv2d_transpose(feature, W_conv, output_size, strides, padding) + b_conv
    return ans_conv


def dropout(input_, prob, name=None):
    return tf.nn.dropout(input_, prob, name=name)


# group_wise feature representation
def group_wise_feature_representation(group_feature, dropout_prob, regularizer, is_training, name, channels):
    with tf.name_scope('concat_feature_' + name):
        concat = tf.concat(group_feature, axis=3)
    conv1 = conv2d(concat, [3, 3, channels * 5, channels], [1, 1, 1, 1], 'SAME',
                   name=name+'_group_conv1', reuse=False, regularizer=regularizer)
    relu1 = tf.nn.relu(conv1)
    dropout1 = tf.cond(tf.equal(is_training, True), lambda: dropout(relu1, dropout_prob), lambda: relu1)
    conv2 = conv2d(dropout1, [3, 3, channels, channels], [1, 1, 1, 1], 'SAME',
                   name=name+'_group_conv2', reuse=False, regularizer=regularizer)
    relu2 = tf.nn.relu(conv2)
    dropout2 = tf.cond(tf.equal(is_training, True), lambda: dropout(relu2, dropout_prob), lambda: relu2)
    inter_feature = conv2d(dropout2, [3, 3, channels, channels], [1, 1, 1, 1], 'SAME',
                           name=name+'_group_conv3', reuse=False, regularizer=regularizer)
    return inter_feature


def single_feature_representation(group_feature, regularizer):
    scope_name = 'single'
    ans = []
    for i, single_feature in enumerate(group_feature):
        conv1 = conv2d(single_feature, [3, 3, 512, 512], [1, 1, 1, 1], 'SAME',
                       name=scope_name + '_1_' + str(i + 1), reuse=False, regularizer=regularizer)
        relu1 = tf.nn.relu(conv1)
        conv2 = conv2d(relu1, [3, 3, 512, 512], [1, 1, 1, 1], 'SAME',
                       name=scope_name + '_2_' + str(i + 1), reuse=False, regularizer=regularizer)
        relu2 = tf.nn.relu(conv2)
        conv3 = conv2d(relu2, [3, 3, 512, 512], [1, 1, 1, 1], 'SAME',
                       name=scope_name + '_3_' + str(i + 1), reuse=False, regularizer=regularizer)
        ans.append(conv3)
    return ans


# conv = []
def collaborative_learning(single_feature_group, inter_feature_pool5, inter_feature_pool4,
                           is_training, regularizer, group):
    concat_pool5 = []
    ans = []
    scope_conv_name = 'collaborative_conv'
    scope_deconv_name = 'collaborative_deconv'
    with tf.name_scope('collaborative_concat'):
        for feature in single_feature_group:
            concat_pool5.append(tf.concat([feature, inter_feature_pool5], axis=3))
    # output_size_tmp = [tf.shape(concat[0])[0], 224, 224]
    batch_size = tf.shape(concat_pool5[0])[0]

    factors = [2, 2, 8]
    kernel_width = [2 * x - x % 2 for x in factors]
    output_size = [[batch_size, 14, 14, 512], [batch_size, 28, 28, 256], [batch_size, 224, 224, 1]]
    for i, feature in enumerate(concat_pool5):
        conv = conv2d(feature, [3, 3, 512 * 2, 512], [1, 1, 1, 1], 'SAME',
                      name=scope_conv_name + str(i + 1), reuse=False, regularizer=regularizer)
        deconv1 = conv2d_transpose(conv, [kernel_width[0], kernel_width[0], 512, 512], output_size[0],
                                   strides=[1, factors[0], factors[0], 1], padding='SAME',
                                   name=scope_deconv_name + '_1_' + str(i + 1), regularizer=regularizer)
        concat1 = tf.concat([deconv1, inter_feature_pool4], axis=3)    # channels: 512 + 512
        conv1 = conv2d(concat1, [3, 3, 512 * 2, 512], [1, 1, 1, 1], 'SAME',
                        name=scope_conv_name + '_1_' + str(i + 1), reuse=False, regularizer=regularizer)
        relu1 = tf.nn.relu(conv1)
        deconv2 = conv2d_transpose(relu1, [kernel_width[1], kernel_width[1], 256, 512], output_size[1],
                                   strides=[1, factors[1], factors[1], 1], padding='SAME',
                                   name=scope_deconv_name + '_2_' + str(i + 1), regularizer=regularizer)
        concat2 = tf.concat([deconv2, group[i].get_pool3()], axis=3)    # channels: 256 + 256
        conv2 = conv2d(concat2, [3, 3, 256 * 2, 256], [1, 1, 1, 1], 'SAME',
                        name=scope_conv_name + '_2_' + str(i + 1), regularizer=regularizer, reuse=False)
        relu2 = tf.nn.relu(conv2)
        deconv_ans = conv2d_transpose(relu2, [kernel_width[2], kernel_width[2], 1, 256], output_size[2],
                                      strides=[1, factors[2], factors[2], 1], padding='SAME',
                                      name=scope_deconv_name + '_3_' + str(i + 1), regularizer=regularizer)
        ans.append(tf.nn.sigmoid(deconv_ans))

    return ans


def main():
    learning_rate = 1e-10
    momentum = 0.99
    dropout_prob = 0.5
    batch_size = 8
    train_step = 60000
    gama = 6
    smooth_epsion = 0.5
    min_value = 1e-5
    weight_decay = 0.0005
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    train_path  = '../dataset/coco7336'
    test_path = '../dataset/icoseg_groups'

    group_input = []
    group_groundtruth = []
    for i in range(5):
        group_input.append(tf.placeholder(tf.float32, [None, 224, 224, 3]))
        group_groundtruth.append(tf.placeholder(tf.float32, [None, 224, 224, 1]))
    prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)

    group = [VGG16(image, regularizer) for image in group_input]
    group_feature_pool5 = [vgg.get_poolled_features() for vgg in group]
    group_feature_pool4 = [vgg.get_pool4() for vgg in group]

    inter_feature5 = group_wise_feature_representation(group_feature_pool5, prob,
                                                       regularizer=regularizer,
                                                       is_training=is_training,
                                                       name='pool5',
                                                       channels=512)
    inter_feature4 = group_wise_feature_representation(group_feature_pool4, prob,
                                                       regularizer=regularizer,
                                                       is_training=is_training,
                                                       name='pool4',
                                                       channels=512)
    intra_feature = single_feature_representation(group_feature_pool5, regularizer=regularizer)
    ans_feature_map = collaborative_learning(
        intra_feature, inter_feature5, inter_feature4, is_training, regularizer=regularizer, group=group)

    with tf.name_scope('calculate_loss'):
        loss_list_cross_entropy = []
        loss_list_smooth = []
        groundtruth = []
        for i in range(5):
            groundtruth.append(tf.reshape(group_groundtruth[i], [-1, 224, 224]))
            ans_feature_map[i] = tf.reshape(ans_feature_map[i], [-1, 224, 224])

        # count negative num
        positive_cnt = tf.count_nonzero(groundtruth)
        negative_cnt = batch_size * 224 * 224 * 5 - positive_cnt
        beta = positive_cnt / (positive_cnt + negative_cnt)
        for i in range(5):
            bool_arr = tf.cast(tf.equal(groundtruth[i], 1), tf.float64)
            ans_feature_map[i] = tf.cast(ans_feature_map[i], tf.float64)
            # cal cross entropy
            loss_list_cross_entropy.append(
                -beta * tf.reduce_sum(tf.log(tf.maximum(ans_feature_map[i], min_value)) * bool_arr)
                - (1 - beta) * tf.reduce_sum(tf.log(tf.maximum((1 - ans_feature_map[i]), min_value)) *
                                             (1 - bool_arr)))
        loss_cross_entropy = tf.reduce_sum(loss_list_cross_entropy)

        # cal smooth L1 loss
        for i in range(5):
            groundtruth[i] = tf.cast(groundtruth[i], tf.float64)
            d = ans_feature_map[i] - groundtruth[i]
            bool_arr = tf.cast(tf.abs(d) < smooth_epsion, tf.float64)
            loss_list_smooth.append(tf.reduce_sum(0.5 * tf.square(d) * bool_arr
                                                  + (smooth_epsion * tf.abs(d) - 0.5 * smooth_epsion * smooth_epsion)
                                                  * (1 - bool_arr)))
        loss_smooth = tf.reduce_sum(loss_list_smooth)
        loss = loss_cross_entropy + gama * loss_smooth + tf.cast(
            tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), tf.float64)

    with tf.name_scope('optimize'):
        train = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss)

    with tf.name_scope('get_training_data'):
        iterator = construct_dataset(train_path, batch_size)
        batch = iterator.get_next()

    with tf.name_scope('get_testing_data'):
        iterator_fortest = construct_dataset(test_path, batch_size)
        batch_fortest = iterator_fortest.get_next()

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    group[0].load_weights('vgg16_weights/vgg16_weights.npz', sess)

    for count in range(train_step):
        feed_dict = {}
        data_batch, label_batch, _ = sess.run(batch)
        for i in range(5):
            feed_dict[group_input[i]] = data_batch[:, i]
            feed_dict[group_groundtruth[i]] = label_batch[:, i]
            feed_dict[is_training] = True
        feed_dict[prob] = dropout_prob

        _ans, _loss, _ = sess.run([ans_feature_map, loss, train], feed_dict=feed_dict)
        if (count + 1) % 100 == 0:
            feed_dict = {}
            data_fortest_batch, label_fortest_batch, _ = sess.run(batch_fortest)
            for i in range(5):
                feed_dict[group_input[i]] = data_fortest_batch[:, i]
                feed_dict[group_groundtruth[i]] = label_fortest_batch[:, i]
                feed_dict[is_training] = False
            feed_dict[prob] = dropout_prob
            _loss_test = sess.run(loss, feed_dict=feed_dict)
            print('step {} train_loss: {}; test_loss: {}'.format(count + 1, _loss, _loss_test))
    saver.save(sess, 'model/our_model_trained.ckpt')


def construct_dataset(dataset_path, batchsize):
    image_dirs_path = os.path.join(dataset_path, 'images')
    groundtruth_path = os.path.join(dataset_path, 'groundtruth')
    groups = os.listdir(image_dirs_path)
    random.shuffle(groups)
    data = []
    label = []
    names = []
    for group in groups:
        img_names = os.listdir(os.path.join(image_dirs_path, group))
        group_data = []
        group_label = []
        group_name = []
        for name in img_names:
            img = cv2.imread(os.path.join(image_dirs_path, group, name), cv2.IMREAD_COLOR)
            img = img / np.max(img, axis=(0, 1))
            groundtruth = cv2.imread(os.path.join(groundtruth_path, group, os.path.splitext(name)[0] + '.png'), cv2.IMREAD_GRAYSCALE)
            groundtruth = groundtruth / np.max(groundtruth)
            groundtruth = groundtruth.reshape([224, 224, 1])
            group_data.append(img)
            group_label.append(groundtruth)
            group_name.append(name)
        data.append(group_data)
        label.append(group_label)
        names.append(group_name)
    data_tmp = np.array(data)
    label_tmp = np.array(label)
    names_tmp = np.array(names)
    data_placeholder = tf.placeholder(dtype=data_tmp.dtype, shape=data_tmp.shape)
    label_placeholder = tf.placeholder(dtype=label_tmp.dtype, shape=label_tmp.shape)
    names_placeholder = tf.placeholder(dtype=names_tmp.dtype, shape=names_tmp.shape)
    dataset = tf.data.Dataset.from_tensor_slices((data_placeholder, label_placeholder, names_placeholder))
    dataset = dataset.batch(batchsize).repeat().shuffle(20)
    iterator = dataset.make_initializable_iterator()
    sess.run(iterator.initializer, feed_dict={data_placeholder: data, label_placeholder: label, names_placeholder: names})
    return iterator


main()
