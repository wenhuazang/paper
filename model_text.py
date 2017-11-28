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
import params


class DANN(object):
    def __init__(self, config):
        self.gpu_fraction = config.fraction
        self.src_name = config.source_name
        self.trg_name = config.target_name
        self.is_dan = config.is_dan
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.source_data_dir = config.source_data_dir
        self.target_data_dir = config.target_data_dir
        self.saver_path = config.saver_path
        self.max_steps = config.max_steps
        self.decay_steps = config.decay_steps
        self.decay_factor = config.decay_factor
        self.beta = config.beta
        self.alpha = config.alpha
        self.mode = config.mode

    def kron(self, f_1, dim_1, f_2, dim_2):
        stack_1 = tf.stack([f_1 for _ in range(dim_1)], axis=2)
        stack_1 = tf.reshape(stack_1, [-1, dim_1 * dim_2])
        stack_2 = tf.stack([f_2 for _ in range(dim_2)], axis=1)
        stack_2 = tf.reshape(stack_2, [-1, dim_1 * dim_2])
        joint_layer_feature = tf.multiply(stack_1, stack_2)
        return joint_layer_feature

    def generator(self, inputs, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            with slim.arg_scope([slim.fully_connected],
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.02), activation_fn=None,
                                weights_regularizer=slim.l2_regularizer(0.001)):
                            net = slim.fully_connected(inputs, 50, scope="fc0")
                            net = tf.nn.relu(net)
                            return net

    def classifier(self, inputs, reuse=False):
        with tf.variable_scope("classifier", reuse=reuse):
            with slim.arg_scope([slim.fully_connected],
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.02), activation_fn=None,
                                weights_regularizer=slim.l2_regularizer(0.001)):
                    net = slim.fully_connected(inputs, 2, scope="fc0")
                    return net

    def discriminator(self, inputs, labels=None, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            with slim.arg_scope([slim.fully_connected],
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.02), activation_fn=None,
                                weights_regularizer=slim.l2_regularizer(0.0001)):
                    if labels is None:
                        logits = slim.fully_connected(inputs, 1, scope="fc0")
                    else:
                        logits = slim.fully_connected(tf.concat([inputs, labels], axis=1), 1, scope="fc")
                    return logits

    def build_model(self, source_images, target_images, source_labels,
                    target_labels, step, mode, is_train=False):
        if not is_train:
            target_feature = self.generator(target_images)
            target_logits = self.classifier(target_feature)
            target_correct = tf.equal(tf.argmax(target_logits, 1), tf.argmax(target_labels, 1))
            target_acc = tf.reduce_mean(tf.cast(target_correct, tf.float32))
            return target_acc
        # if seperate generator with reuse, acc down!!! So, merge it... maybe the parameter step?
        merged_images = tf.concat([source_images, target_images], 0)
        merge_features = self.generator(merged_images)
        # source_feature = tf.slice(merge_features, [0, 0], [self.batch_size, params.g_last])
        # target_feature = tf.slice(merge_features, [self.batch_size, 0], [self.batch_size, params.g_last])
        # source_logits = self.classifier(source_feature)
        # target_logits = target_feature
        c_logits = self.classifier(merge_features)
        source_logits = tf.slice(c_logits, [0, 0], [self.batch_size, 2])
        target_logits = tf.slice(c_logits, [self.batch_size, 0], [self.batch_size, 2])

        merged_labels = tf.concat([source_labels, tf.one_hot(tf.argmax(tf.nn.softmax(target_logits), 1), 2)], 0)
        # merged_labels = tf.concat([source_labels, tf.nn.softmax(target_logits)], 0)
        # merge_features = self.kron(merge_features, 2, merged_labels, 50)

        d_logits = self.discriminator(merge_features, merged_labels)
        d_source_logits = tf.slice(d_logits, [0, 0], [self.batch_size, 1])
        d_target_logits = tf.slice(d_logits, [self.batch_size, 0], [self.batch_size, 1])

        lr = tf.train.exponential_decay(self.learning_rate, step, self.decay_steps,
                                        self.decay_factor,
                                        staircase=True)

        p = tf.cast(step, tf.float32) / self.max_steps
        decay = params.coeff * (2 / (1. + tf.exp(-2.5 * p)) - 1) + 0.1
        ############
        # compute loss
        ############
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

        regularization_loss = tf.add_n(tf.losses.get_regularization_losses())
        c_loss = source_label_loss + self.beta * target_label_loss + regularization_loss
        clip_disc_weights = None
        if mode == 'gan':
            print('########## mode: gan ############')
            source_domain_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=d_source_logits, labels=tf.ones_like(d_source_logits)
                ))
            target_domain_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=d_target_logits, labels=tf.zeros_like(d_target_logits)
                ))
            disc_loss = source_domain_loss + target_domain_loss + regularization_loss
            d_loss = disc_loss
            g_loss = -decay * disc_loss + c_loss
        elif mode == 'wgan':
            print('########## mode: wgan ############')
            # gen_cost = -tf.reduce_mean(disc_fake)
            disc_cost = tf.reduce_mean(d_target_logits) - tf.reduce_mean(d_source_logits)
            t_vars = tf.trainable_variables()
            d_vars = [var for var in t_vars if 'discriminator' in var.name]
            clip_disc_weights = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]
            d_loss = disc_cost + regularization_loss
            g_loss = -disc_cost + c_loss + regularization_loss

        # wgan-gp
        else:
            print('########## mode: wgan-gp ############')
            disc_loss = tf.reduce_mean(d_target_logits) - tf.reduce_mean(d_source_logits)
            alpha = tf.random_uniform(shape=[self.batch_size, 1],
                                      minval=0., maxval=1., name='alpha')
            differences = g_target_outputs - g_source_outputs
            interpolates = g_source_outputs + (alpha * differences)
            print(tf.shape(interpolates))  # [128, 4]
            gradients = tf.gradients(self.discriminator(interpolates, None,
                                                        reuse=True), [interpolates])[0]
            # reduction_indices = 1, considering batch sample
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            d_loss = disc_loss + params.LAMBDA * gradient_penalty + regularization_loss
            # d_loss = disc_loss
            g_loss = -disc_loss + c_loss + regularization_loss

        ##################
        # train_op
        ##################
        # c_loss += disc_loss
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]
        c_vars = [var for var in t_vars if 'classifier' in var.name]
        algorithm = 1

        if algorithm == 1:
            d_train_op = tf.train.MomentumOptimizer(lr, 0.9).minimize(d_loss, step, d_vars)
            g_train_op = tf.train.MomentumOptimizer(lr, 0.9).minimize(g_loss, step, g_vars)
            c_train_op = tf.train.MomentumOptimizer(lr, 0.9).minimize(c_loss, step, c_vars)
        elif algorithm == 2:
            d_train_op = tf.train.AdagradOptimizer(0.001).minimize(d_loss, step, d_vars)
            g_train_op = tf.train.AdagradOptimizer(0.001).minimize(g_loss, step, g_vars)
            c_train_op = tf.train.AdagradOptimizer(0.001).minimize(c_loss, step, c_vars)
        else:
            d_train_op = tf.train.RMSPropOptimizer(lr).minimize(d_loss, step, d_vars)
            g_train_op = tf.train.RMSPropOptimizer(lr).minimize(g_loss, step, g_vars)
            c_train_op = tf.train.RMSPropOptimizer(lr).minimize(c_loss, step, c_vars)

        pack_op_loss = [d_train_op, g_train_op, c_train_op, d_loss, c_loss]
        pack_acc = [source_acc, target_acc, decay, lr, clip_disc_weights]
        return pack_op_loss, pack_acc

    def train_model(self):
        with tf.Graph().as_default():
            global_step = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
            source_images, source_labels = ops.load_batch_amazon_review(batch_size=self.batch_size,
                                                                        data_dir=self.source_data_dir,
                                                                        dataname='train.' + self.src_name,
                                                                        is_train=True)
            target_images, target_labels = ops.load_batch_amazon_review(batch_size=self.batch_size,
                                                                        data_dir=self.target_data_dir,
                                                                        dataname='train.' + self.trg_name,
                                                                        is_train=True)
            pack_op_loss, pack_acc = self.build_model(source_images, target_images,
                                                      source_labels, target_labels,
                                                      global_step, self.mode,
                                                      is_train=True)

            d_train_op = pack_op_loss[0]
            g_train_op = pack_op_loss[1]
            c_train_op = pack_op_loss[2]

            d_loss = pack_op_loss[3]
            c_loss = pack_op_loss[4]

            source_acc = pack_acc[0]
            target_acc = pack_acc[1]
            decay = pack_acc[2]
            lr = pack_acc[3]
            clip_disc_weights = pack_acc[4]
            # gradient_penalty = pack_acc[6]
            # slopes = pack_acc[7]

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=10000)

            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = self.gpu_fraction
            sess = tf.Session(config=config)
            # set_session(sess)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            tf.train.start_queue_runners(sess=sess)
            for step in xrange(self.max_steps + 1):
                start_time = time.time()
                _, _, _ = \
                    sess.run([d_train_op, g_train_op, c_train_op])
                if clip_disc_weights is not None:
                    _ = sess.run(clip_disc_weights)
                duration = time.time() - start_time
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
                    print('lr:', sess.run(lr))
                    # print('ratio:', sess.run(ratio))
                    # print(sess.run(self.global_step))

                if step % params.train_save == 0:
                    save_path = os.path.join(self.saver_path, params.save_model)
                    saver.save(sess, save_path, global_step=step)

    def test_model(self, step):
        with tf.Graph().as_default():
            global_step = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)

            test_target_images, test_target_labels = ops.load_batch_amazon_review(batch_size=1,
                                                                                  data_dir=self.target_data_dir,
                                                                                  dataname='test.' + self.trg_name,
                                                                                  is_train=False)

            target_acc = self.build_model(None, test_target_images,
                                          None, test_target_labels,
                                          global_step, self.mode,
                                          is_train=False)

            saver = tf.train.Saver(tf.global_variables())
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = self.gpu_fraction
            sess = tf.Session(config=config)
            restore_path = os.path.join(self.saver_path, params.save_model + '-' + str(step))
            print(restore_path)
            # init local vars
            sess.run(tf.local_variables_initializer())
            saver.restore(sess, restore_path)
            precision = self.eval_once(sess, target_acc)

        return precision

    def eval_once(self, sess, accuracy):
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
            total_acc = 0
            step = 0
            while not coord.should_stop():
                acc = sess.run(accuracy)
                total_acc += acc
                step += 1
        except Exception as e:
            print ('num: ', step)
            avg_acc = total_acc / step
            print('precision:', avg_acc)
            coord.request_stop(e)
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=1)
            return avg_acc