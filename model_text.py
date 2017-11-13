from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
import ops
import tensorflow.contrib.slim as slim
# from keras.backend.tensorflow_backend import set_session
from datetime import datetime
import os.path
import time
import numpy as np
from flip_gradient import flip_gradient
import math
import params


class DANN(object):
    def __init__(self, config):
        self.gpu_fraction = config.fraction
        self.is_dan = config.is_dan
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.source_data_dir = config.source_data_dir
        self.target_data_dir = config.target_data_dir
        self.saver_path = config.saver_path
        self.image_save_dir = config.image_save_dir
        self.max_steps = config.max_steps
        self.decay_steps = config.decay_steps
        self.decay_factor = config.decay_factor
        self.beta = config.beta
        self.alpha = config.alpha

    def dan_s(self, inputs, global_step):
        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.02), activation_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.0001)):
            with slim.arg_scope([slim.batch_norm], decay=0.80, center=True, scale=True, epsilon=1e-5,
                                activation_fn=tf.nn.relu, is_training=True):
                p = tf.cast(global_step, tf.float32) / self.max_steps
                # decay = 0.17 * (2 / (1. + tf.exp(-2.5 * p)) - 1) + 0.1
                decay = 0.2 * (2 / (1. + tf.exp(-10 * p)) - 1) + 0.2
                feature = flip_gradient(inputs, decay)

                avg_0 = tf.reduce_mean(feature[:int(self.batch_size)], axis=0, keep_dims=True)
                avg_1 = tf.reduce_mean(feature[int(self.batch_size):], axis=0, keep_dims=True)
                avg = tf.concat([avg_0, avg_1], 0)
                # [2500]---[128]
                net = slim.fully_connected(avg, 128, scope="fc0")
                net = slim.batch_norm(net, scope='bn0')

                # [128]---[1]
                logits = slim.fully_connected(net, 1, scope="fc1")
                source_logits = tf.slice(logits, [0, 0], [1, 1])
                target_logits = tf.slice(logits, [1, 0], [1, 1])
                # compute domain_loss
                s_source_domain_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=source_logits, labels=tf.ones_like(source_logits)
                    ))
                s_target_domain_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=target_logits, labels=tf.zeros_like(target_logits)
                    ))
                s_domain_loss = (s_source_domain_loss + s_target_domain_loss) / 2.0
                return s_domain_loss

    def dan_s2(self, inputs, global_step):
        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.02), activation_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.0001)):
            with slim.arg_scope([slim.batch_norm], decay=0.80, center=True, scale=True, epsilon=1e-5,
                                activation_fn=tf.nn.relu, is_training=True):
                p = tf.cast(global_step, tf.float32) / self.max_steps
                decay = 0.1 * (2 / (1. + tf.exp(-10 * p)) - 1) + 0.1
                feature = flip_gradient(inputs, decay)

                # avg_0 = tf.reduce_mean(feature[:int(self.batch_size)], axis=0, keep_dims=True)
                # avg_1 = tf.reduce_mean(feature[int(self.batch_size):], axis=0, keep_dims=True)
                # avg = tf.concat([avg_0, avg_1], 0)
                avgs = []
                for i in range(4):
                    avgs.append(tf.reduce_mean(
                        feature[int(i * self.batch_size / 2):int((i + 1) * self.batch_size / 2)],
                        axis=0, keep_dims=True))
                avg_00 = tf.abs(avgs[0] - avgs[1])
                avg_11 = tf.abs(avgs[2] - avgs[3])
                avg_01 = tf.abs(avgs[0] - avgs[2])
                avg_10 = tf.abs(avgs[1] - avgs[3])
                avg = tf.concat([avg_00, avg_11, avg_01, avg_10], 0)
                # [2500]---[128]
                net = slim.fully_connected(avg, 128, scope="fc0")
                net = slim.batch_norm(net, scope='bn0')

                # [128]---[1]
                logits = slim.fully_connected(net, 1, scope="fc1")
                source_logits = tf.slice(logits, [0, 0], [1, 1])
                target_logits = tf.slice(logits, [1, 0], [1, 1])

                # compute domain_loss
                s_source_domain_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=source_logits, labels=tf.ones_like(source_logits)
                    ))
                s_target_domain_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=target_logits, labels=tf.zeros_like(target_logits)
                    ))
                s_domain_loss = (s_source_domain_loss + s_target_domain_loss) / 2.0
                return s_domain_loss

    def kron(self, f_1, dim_1, f_2, dim_2):
        stack_1 = tf.stack([f_1 for _ in range(dim_1)], axis=2)
        stack_1 = tf.reshape(stack_1, [2 * self.batch_size, dim_1 * dim_2])
        stack_2 = tf.stack([f_2 for _ in range(dim_2)], axis=1)
        stack_2 = tf.reshape(stack_2, [2 * self.batch_size, dim_1 * dim_2])
        joint_layer_feature = tf.multiply(stack_1, stack_2)
        return joint_layer_feature

    def generator(self, inputs, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.02), activation_fn=None,
                                weights_regularizer=slim.l2_regularizer(0.0001)):
                with slim.arg_scope([slim.conv2d], kernel_size=[5, 5], stride=1, padding="SAME"):
                    with slim.arg_scope([slim.max_pool2d], kernel_size=[2, 2], stride=2, padding="SAME"):
                        with slim.arg_scope([slim.batch_norm], decay=0.80, center=True, scale=True, epsilon=1e-5,
                                            activation_fn=tf.nn.relu, is_training=True):
                            net = slim.fully_connected(inputs, 1000, scope="fc0")
                            net = slim.batch_norm(net, scope='bn0')
                            # [1000]---[100]
                            net = slim.fully_connected(net, 100, scope="fc1")
                            net = slim.batch_norm(net, scope='bn1')
                            # [100]---[100] no activation function
                            net = slim.fully_connected(net, 2, scope="fc2")

        return net

    def discriminator(self, inputs, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            with slim.arg_scope([slim.fully_connected],
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.02), activation_fn=None,
                                weights_regularizer=slim.l2_regularizer(0.0001)):
                with slim.arg_scope([slim.batch_norm], decay=0.80, center=True, scale=True, epsilon=1e-5,
                                    activation_fn=tf.nn.relu, is_training=True):
                    net = slim.fully_connected(inputs, params.d_f0, scope="fc0")
                    net = slim.batch_norm(net, scope='bn0')
                    net = slim.fully_connected(net, params.d_f1, scope="fc00")
                    net = slim.batch_norm(net, scope='bn00')
                    # [100]---[1]
                    logits = slim.fully_connected(net, 1, scope="fc1")

                    d_source_logits = tf.slice(logits, [0, 0], [self.batch_size, 1])
                    d_target_logits = tf.slice(logits, [self.batch_size, 0], [self.batch_size, 1])

        return d_source_logits, d_target_logits

    def classifier(self, inputs, reuse=False):

        with tf.variable_scope("classifier", reuse=reuse):
            with slim.arg_scope([slim.fully_connected],
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.02), activation_fn=None,
                                weights_regularizer=slim.l2_regularizer(0.001)):
                with slim.arg_scope([slim.batch_norm], decay=0.80, center=True, scale=True, epsilon=1e-5,
                                    activation_fn=tf.nn.relu, is_training=True):
                    source_inputs = tf.slice(inputs, [0, 0], [self.batch_size, params.input_dim])
                    target_inputs = tf.slice(inputs, [self.batch_size, 0], [self.batch_size, params.input_dim])

                    net = slim.fully_connected(source_inputs, 2, scope="fc0")
                    net = slim.batch_norm(net, scope='bn0')
                    # fc1 [10] --> [2]
                    source_logits = slim.fully_connected(net, 2, scope="fc1")
                    source_logits = tf.add(source_inputs, source_logits)

                    target_logits = target_inputs
                    delta_loss = tf.nn.l2_loss(source_logits)
                    loss = tf.nn.l2_loss(source_inputs)
                    ratio = delta_loss / loss

                return source_logits, target_logits, ratio

    def build_model(self, source_images, target_images, source_labels, target_labels, step, model):
        merged_images = tf.concat([source_images, target_images], 0)
        single_feature = self.generator(merged_images)
        source_logits, target_logits, ratio = self.classifier(single_feature)
        # Kronecker product of multi_feature and label
        # for Unsupervisied domain adaption
        merged_labels = tf.concat([source_labels, tf.nn.softmax(target_logits)], 0)
        joint_layer_feature = self.kron(single_feature, 2, merged_labels, 2)
        d_source_logits, d_target_logits = self.discriminator(single_feature)
        ############
        # compute loss
        ############
        source_domain_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_source_logits, labels=tf.ones_like(d_source_logits)
            ))
        target_domain_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_target_logits, labels=tf.zeros_like(d_target_logits)
            ))

        # source data labeled
        source_label_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=source_logits, labels=source_labels))

        # target data not labeled(cross entropy with its self)
        target_label_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=target_logits, labels=tf.nn.softmax(target_logits)))

        # source_acc and target acc
        source_correct = tf.equal(tf.argmax(source_logits, 1), tf.argmax(source_labels, 1))
        source_acc = tf.reduce_mean(tf.cast(source_correct, tf.float32))

        target_correct = tf.equal(tf.argmax(target_logits, 1), tf.argmax(target_labels, 1))
        target_acc = tf.reduce_mean(tf.cast(target_correct, tf.float32))
        # regularization_loss = tf.reduce_mean(tf.losses.get_regularization_loss())
        c_loss = source_label_loss + self.beta * target_label_loss + tf.losses.get_regularization_loss()
        d_loss = source_domain_loss + target_domain_loss
        # g_loss = -d_loss + c_loss
        ##################
        # train_op
        ##################
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]
        c_vars = [var for var in t_vars if 'classifier' in var.name]

        lr = tf.train.exponential_decay(self.learning_rate, step, self.decay_steps,
                                        self.decay_factor,
                                        staircase=True)

        p = tf.cast(step, tf.float32) / self.max_steps
        decay = 0.5 * (2 / (1. + tf.exp(-2.5 * p)) - 1) + 0.1

        d_train_op = tf.train.MomentumOptimizer(lr, 0.9).minimize(d_loss, step, d_vars)
        g_train_op = tf.train.MomentumOptimizer(lr, 0.9).minimize(-decay * d_loss + c_loss, step, g_vars)
        c_train_op = tf.train.MomentumOptimizer(lr, 0.9).minimize(c_loss, step, c_vars)

        # train_op = tf.train.AdagradOptimizer(self.learning_rate).minimize(total_loss, step)
        pack_op_loss = [d_train_op, g_train_op, c_train_op, d_loss, c_loss]
        pack_acc = [source_acc, target_acc, decay, lr]

        return pack_op_loss, pack_acc

    def train_model(self):
        with tf.Graph().as_default():
            global_step = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
            source_images, source_labels = ops.load_batch_amazon_review(batch_size=self.batch_size,
                                                                        data_dir=self.source_data_dir,
                                                                        dataname='train.books')
            target_images, target_labels = ops.load_batch_amazon_review(batch_size=self.batch_size,
                                                                        data_dir=self.target_data_dir,
                                                                        dataname='train.kitchen')
            pack_op_loss, pack_acc = self.build_model(source_images, target_images,
                                                      source_labels, target_labels,
                                                      global_step, self.is_dan)

            d_train_op = pack_op_loss[0]
            g_train_op = pack_op_loss[1]
            c_train_op = pack_op_loss[2]

            d_loss = pack_op_loss[3]
            c_loss = pack_op_loss[4]

            source_acc = pack_acc[0]
            target_acc = pack_acc[1]
            decay = pack_acc[2]
            lr = pack_acc[3]

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=10000)

            # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = self.gpu_fraction
            sess = tf.Session(config=config)
            # set_session(sess)
            init = tf.global_variables_initializer()
            sess.run(init)

            tf.train.start_queue_runners(sess=sess)
            for step in xrange(self.max_steps + 1):
                start_time = time.time()
                _, _, _ = \
                    sess.run([d_train_op, g_train_op, c_train_op])
                duration = time.time() - start_time
                # assert not np.isnan(t_loss), 'Model diverged with loss = NaN'
                if step % params.train_show == 0:
                    d_l, c_l = sess.run([d_loss, c_loss])
                    acc_s, acc_t = sess.run([source_acc, target_acc])
                    examples_per_sec = self.batch_size / duration

                    format_str = (
                        '%s: step %d, d_loss = %.3f c_loss=%.3f [%.1f examples/sec]'
                    )
                    print(format_str % (datetime.now(), step, d_l, c_l,
                                        examples_per_sec))

                    format_str_1 = 'acc_s: {:.3f}, acc_t: {:.3f}' \
                        .format(acc_s, acc_t)
                    print(format_str_1)
                    print('decay:', sess.run(decay))
                    print('lr:', lr)
                    # print(sess.run(self.global_step))

                if step % params.train_save == 0:
                    save_path = os.path.join(self.saver_path, params.save_model)
                    saver.save(sess, save_path, global_step=step)

    def test_model(self, step):
        with tf.Graph().as_default():
            global_step = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
            test_source_images, test_source_labels = ops.load_batch_amazon_review(batch_size=self.batch_size,
                                                                                  data_dir=self.source_data_dir,
                                                                                  dataname='test.books')
            test_target_images, test_target_labels = ops.load_batch_amazon_review(batch_size=self.batch_size,
                                                                                  data_dir=self.source_data_dir,
                                                                                  dataname='test.kitchen')

            _, pack = self.build_model(test_source_images, test_target_images,
                                       test_source_labels, test_target_labels,
                                       global_step, self.is_dan)

            target_acc = pack[1]
            saver = tf.train.Saver(tf.global_variables())
            sess = tf.Session()
            restore_path = os.path.join(self.saver_path, params.save_model + '-' + str(step))
            print(restore_path)
            saver.restore(sess, restore_path)
            precision = self.eval_once(sess, target_acc)

        return precision

    def eval_once(self, sess, accuracy):
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
            num_iter = int(math.ceil(10000 / self.batch_size))
            true_count = 0
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([accuracy])
                true_count += np.sum(predictions)
                step += 1

            precision = (1.0 * true_count) / num_iter
            print('precision:', precision)
        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=1)

        return precision
