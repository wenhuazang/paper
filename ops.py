import os
from six.moves import xrange
import tensorflow as tf
import scipy.io as sio
import numpy as np
from skimage.io import imsave


def read_and_decode_usps(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'labels': tf.FixedLenFeature([], tf.int64),
                                           'images': tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['images'], tf.uint8)
    img = tf.reshape(img, [28, 28, 1])
    label = tf.cast(features['labels'], tf.int32) - 1
    return img, label


def read_and_decode_mnist(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'labels': tf.FixedLenFeature([], tf.int64),
                                           'images': tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['images'], tf.uint8)
    img = tf.reshape(img, [784])
    label = tf.cast(features['labels'], tf.int32)
    return img, label


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    num_preprocess_threads = 3
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)
    return images, tf.reshape(label_batch, [batch_size])


def load_usps_images_labels(filename, batch_size, shuffle):
    read_input_image, read_input_label = read_and_decode_usps(filename)
    reshaped_image = tf.cast(read_input_image, tf.float32)
    min_queue_examples = 5000
    return _generate_image_and_label_batch(reshaped_image, read_input_label,
                                           min_queue_examples, batch_size,
                                           shuffle=shuffle)


def load_mnist_images_labels(filename, batch_size, shuffle):
    read_input_image, read_input_label = read_and_decode_mnist(filename)
    reshaped_image = tf.cast(read_input_image, tf.float32)
    min_queue_examples = 5000
    return _generate_image_and_label_batch(reshaped_image, read_input_label,
                                           min_queue_examples, batch_size,
                                           shuffle=shuffle)


def read_and_decode_mnistm(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'labels': tf.FixedLenFeature([], tf.int64),
                                           'images': tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['images'], tf.uint8)
    img = tf.reshape(img, [28, 28, 3])
    label = tf.cast(features['labels'], tf.int32)
    return img, label


def load_mnistm_images_labels(filename, batch_size, shuffle):
    read_input_image, read_input_label = read_and_decode_mnistm(filename)
    reshaped_image = tf.cast(read_input_image, tf.float32)
    min_queue_examples = 5000
    return _generate_image_and_label_batch(reshaped_image, read_input_label,
                                           min_queue_examples, batch_size,
                                           shuffle=shuffle)


def load_batch_usps(batch_size, data_dir, add_noise=False, dataname='train.usps'):
    filename = os.path.join(data_dir, dataname)
    images, labels = load_usps_images_labels(filename, batch_size, shuffle=True)
    images = tf.cast(images, tf.float32)
    images = 2 * (images / 255) - 1
    if add_noise:
        eps = tf.random_normal([batch_size, 28, 28, 1], 0, 0.01, dtype=tf.float32)
        images = images + eps

    images = tf.reshape(images, [batch_size, 28, 28, 1])
    labels = tf.cast(labels, tf.int32)
    labels = tf.reshape(labels, [batch_size])
    labels = tf.one_hot(labels, 10)
    return images, labels


def load_batch_mnistm(batch_size, data_dir, dataname):
    filename = os.path.join(data_dir, dataname)
    images, labels = load_mnistm_images_labels(filename, batch_size, shuffle=True)
    images = tf.cast(images, tf.float32)
    images = 2 * (images / 255) - 1
    labels = tf.cast(labels, tf.int32)
    labels = tf.reshape(labels, [batch_size])
    labels = tf.one_hot(labels, 10)
    return images, labels


def load_batch_mnist(batch_size, data_dir, dataname, add_noise=False, is_RGB=False):
    filename = os.path.join(data_dir, dataname)
    images, labels = load_mnist_images_labels(filename, batch_size, shuffle=False)
    images = tf.cast(images, tf.float32)
    images = 2 * (images / 255) - 1
    if add_noise:
        eps = tf.random_normal([batch_size, 784], 0, 0.05, dtype=tf.float32)
        images = images + eps

    images = tf.reshape(images, [batch_size, 28, 28, 1])
    if is_RGB:
        images = tf.concat([images, images, images], axis=3)
    labels = tf.cast(labels, tf.int32)
    labels = tf.reshape(labels, [batch_size])
    labels = tf.one_hot(labels, 10)
    return images, labels

