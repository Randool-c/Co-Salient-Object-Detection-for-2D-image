import tensorflow as tf
import numpy as np
from vgg16 import VGG16
import matplotlib.pyplot as plt
import os
import cv2
import random



def conv2d(input_x, filter_shape, strides, padding, name, reuse, regularizer):
    assert padding in ['SAME', 'VALID']
    with tf.variable_scope(name, reuse=reuse):
        W_conv = tf.get_variable('weights', shape=filter_shape, regularizer=regularizer)
        b_conv = tf.get_variable('biases', shape=[filter_shape[-1]])
        h_conv = tf.nn.conv2d(
            input_x, W_conv, strides=strides, padding=padding)
        output = b_conv + h_conv
    return output

def brn_conv2d(input_img, filter_shape, strides, padding, name, regularizer):
    assert padding in ['SAME', 'VALID']
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        kernel = tf.get_variable('weights', shape=filter_shape, trainable=True, regularizer=regularizer,
                                    initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=0.1))
        biases = tf.Variable([0] * filter_shape[-1], name='biases', trainable=True, dtype=tf.float32)
        conv = tf.nn.conv2d(input_img, kernel, strides=strides, padding=padding)
        out = tf.nn.bias_add(conv, biases)
        relu = tf.nn.relu(out)
    return relu


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


#n neighbors.
def brn_network(input_img, n, regularizer):
    conv1 = brn_conv2d(input_img, filter_shape=[3, 3, 4, 64], strides=[1, 1, 1, 1],
                        padding='SAME', name='brn_conv1', regularizer=regularizer)
    conv2 = brn_conv2d(conv1, filter_shape=[3, 3, 64, 64], strides=[1, 1, 1, 1],
                        padding='SAME', name='brn_conv2', regularizer=regularizer)
    conv3 = brn_conv2d(conv2, filter_shape=[3, 3, 64, 64], strides=[1, 1, 1, 1],
                        padding='SAME', name='brn_conv3', regularizer=regularizer)
    conv4 = brn_conv2d(conv3, filter_shape=[3, 3, 64, 128], strides=[1, 1, 1, 1],
                        padding='SAME', name='brn_conv4', regularizer=regularizer)
    conv5 = brn_conv2d(conv4, filter_shape=[3, 3, 128, 128], strides=[1, 1, 1, 1],
                        padding='SAME', name='brn_conv5', regularizer=regularizer)
    conv6 = brn_conv2d(conv5, filter_shape=[3, 3, 128, 128], strides=[1, 1, 1, 1],
                        padding='SAME', name='brn_conv6', regularizer=regularizer)
     
    with tf.variable_scope('brn_conv7', reuse=tf.AUTO_REUSE):
        kernel = tf.get_variable('weights', shape=[3, 3, 128, n * n], trainable=True, regularizer=regularizer,
                                    initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=0.1))
        tmp = int((n * n - 1) / 2)
        biases = tf.Variable([0] * tmp + [1] + [0] * tmp, name='biases', trainable=True, dtype=tf.float32)
        conv = tf.nn.conv2d(conv6, kernel, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases)
    return out

def get_blocked(arr, n):
	multi1 = tf.reshape(arr[:, :224, :224], [-1, 224, 224, 1])
	for i in range(n):
		for j in range(n):
			if i == 0 and j == 0:
				continue
			multi1 = tf.concat([multi1, tf.reshape(arr[:, i: 224 + i, j: 224 + j], [-1, 224, 224, 1])], axis=3)
	return multi1

# saliency_map.shape = [-1, 224, 224], coefficients.shape = [-1, 224, 224, n * n]
# 计算coefficients和saliency_map在每个position n*n范围内的积和
def product_sum(saliency_map, coefficients, n):
    padded_space = int(n / 2)
    padded = tf.pad(saliency_map, [[0, 0], [padded_space, padded_space], [padded_space, padded_space]])
    multi1 = tf.reshape(get_blocked(padded, n), [-1, 224 * 224, n * n])
    multi2 = tf.reshape(coefficients, [-1, 224 * 224, n * n])
    multied = multi1 * multi2
    ans = tf.reduce_sum(multied, axis=2)
    ans = tf.reshape(ans, [-1, 224, 224])
    return ans


def main():
    learning_rate = 1e-10
    momentum = 0.99
    dropout_prob = 0.5
    batch_size = 8
    train_step = 40000
    gama = 6
    smooth_epsion = 0.5
    min_value = 1e-5
    weight_decay = 0.0005
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    train_path  = '../dataset/coco7336'
    test_path = '../dataset/icoseg_groups'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    cosal_graph = tf.Graph()
    brn_graph = tf.Graph()

    # co-saliency network graph
    with cosal_graph.as_default():
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

        sess_cosal = tf.Session(graph=cosal_graph, config=config)
        sess_cosal.run(tf.global_variables_initializer())

        with tf.name_scope('get_training_data'):
            iterator = construct_dataset(train_path, batch_size, sess_cosal)
            batch = iterator.get_next()

        with tf.name_scope('get_testing_data'):
            iterator_fortest = construct_dataset(test_path, batch_size, sess_cosal)
            batch_fortest = iterator_fortest.get_next()
        saver_cosal = tf.train.Saver()
        saver_cosal.restore(sess_cosal, 'model/our_model.ckpt')
        

    # brn network graph
    with brn_graph.as_default():
        n = 5
        brn_input = tf.placeholder(tf.float32, [None, 224, 224, 3])
        brn_groundtruth_input = tf.placeholder(tf.float32, [None, 224, 224, 1])
        saliency_map = tf.placeholder(tf.float32, [None, 224, 224])  # 输入的saliency map不考虑channel维度
        brn_groundtruth = tf.reshape(brn_groundtruth_input, [-1, 224, 224])
        with tf.name_scope('brn_concat'):
            concat = tf.concat([brn_input, tf.reshape(saliency_map, [-1, 224, 224, 1])], axis=3)
        with tf.name_scope('brn_generate_final_saliencymap'):
            brn_propagation_coefficients = brn_network(concat, n, regularizer)
            final_map = product_sum(
                saliency_map, brn_propagation_coefficients, n)  # final_map.shape=[-1, 224, 224]
        
        with tf.name_scope('cal_brn_loss'):
            # cross entropy
            bool_arr = tf.cast(tf.equal(brn_groundtruth, 1), tf.float64)
            final_map = tf.cast(final_map, tf.float64)
            brn_cross_entropy = (-tf.reduce_sum(tf.log(tf.maximum(final_map, min_value)) * bool_arr)
                    - tf.reduce_sum(tf.log(tf.maximum((1 - final_map), min_value)) *
                                        (1 - bool_arr)))

            # calculate smooth l1 loss
            brn_groundtruth = tf.cast(brn_groundtruth, tf.float64)
            d = final_map - brn_groundtruth
            bool_arr = tf.cast(tf.abs(d) < smooth_epsion, tf.float64)
            brn_smoothl1 = tf.reduce_sum(0.5 * tf.square(d) * bool_arr
                                                + (smooth_epsion * tf.abs(d) - 0.5 * smooth_epsion * smooth_epsion)
                                                * (1 - bool_arr))
            brn_loss = brn_cross_entropy + gama * brn_smoothl1
        
        with tf.name_scope('brn_optimizer'):
            brn_train = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(brn_loss)
        
        sess_brn = tf.Session(graph=brn_graph, config=config)
        sess_brn.run(tf.global_variables_initializer())
        saver_brn = tf.train.Saver()

    # training process
    for count in range(train_step):
        feed_dict = {}
        data_batch, label_batch, _ = sess_cosal.run(batch)
        for i in range(5):
            feed_dict[group_input[i]] = data_batch[:, i]
            feed_dict[group_groundtruth[i]] = label_batch[:, i]
            feed_dict[is_training] = True
        feed_dict[prob] = dropout_prob

        _ans = sess_cosal.run(ans_feature_map, feed_dict=feed_dict)  # 原网络生成结果
        brn_loss_group = 0
        for i in range(5):
            _ans[i] = _ans[i].reshape(-1, 224, 224)
            _brn_loss, _ = sess_brn.run([brn_loss, brn_train], 
                feed_dict={brn_input: data_batch[:, i], saliency_map: _ans[i], brn_groundtruth_input: label_batch[:, i]})
            brn_loss_group += _brn_loss

        if (count + 1) % 100 == 0:
            feed_dict = {}
            data_fortest_batch, label_fortest_batch, _ = sess_cosal.run(batch_fortest)
            for i in range(5):
                feed_dict[group_input[i]] = data_fortest_batch[:, i]
                feed_dict[group_groundtruth[i]] = label_fortest_batch[:, i]
                feed_dict[is_training] = False
            feed_dict[prob] = dropout_prob
            _ans_test = sess_cosal.run(ans_feature_map, feed_dict=feed_dict)
            brn_loss_group_fortest = 0
            for i in range(5):
                _ans_test[i] = _ans_test[i].reshape(-1, 224, 224)
                _brn_loss_test = sess_brn.run(brn_loss,
                    feed_dict={
                        brn_input: data_fortest_batch[:, i], 
                        saliency_map: _ans_test[i], 
                        brn_groundtruth_input: 
                        label_fortest_batch[:, i]})
                brn_loss_group_fortest += _brn_loss_test
            print('step {} brn train_loss: {:.6f}, test_loss: {:.6f}'.format(
                count + 1, brn_loss_group, brn_loss_group_fortest))
        if (count + 1) % 1000 == 0:
            saver_brn.save(sess_brn, 'model/brn_v4_trained.ckpt')
    saver_brn.save(sess_brn, 'model/brn_v4_trained.ckpt')


def construct_dataset(dataset_path, batchsize, sess):
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
            groundtruth = cv2.imread(
                os.path.join(groundtruth_path, group, os.path.splitext(name)[0] + '.png'), cv2.IMREAD_GRAYSCALE)
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
    sess.run(
        iterator.initializer, feed_dict={
            data_placeholder: data, label_placeholder: label, names_placeholder: names})
    return iterator


main()
