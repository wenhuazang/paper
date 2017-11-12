import tensorflow as tf
import numpy as np
from model import DANN
import scipy.io as sio
import os
import pprint
import params


flags = tf.app.flags
flags.DEFINE_integer("max_steps",   3000, "maximum step to train model")
flags.DEFINE_integer("decay_steps", 4000, "steps to change learning rate")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("decay_factor", 0.92, "the change of ration of learning rate")
flags.DEFINE_float("alpha", 0.5, "the regularized of dan_2s")
flags.DEFINE_float("beta", 0.8, "the regularized of target loss")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [64]")
flags.DEFINE_integer("test_point", 0, "The beginning of point of test")
flags.DEFINE_integer("save_size", 100, "The size of save parameter [100]")
flags.DEFINE_float("fraction", 0.4, "per process gpu memory fraction")
flags.DEFINE_string("is_dan", "dan", "whether using distributional adversarial network")
flags.DEFINE_string("source_data_dir", "./data/mnist", "the path of source data")
flags.DEFINE_string("target_data_dir", "./data/usps", "the path of target data")
flags.DEFINE_string("saver_path", "./saver_mnist_usps_me2",
                    "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("image_save_dir", "./data/DANN_usps2mnist/output",
                    "Directory name to save the image samples [samples]")

FLAGS = flags.FLAGS
pp = pprint.PrettyPrinter()


def main(_):
    pp.pprint(flags.FLAGS.__flags)
    if not os.path.exists(FLAGS.saver_path):
        os.mkdir(FLAGS.saver_path)
    model = DANN(FLAGS)
    result = []
    best_result = 0
    # model.train_model()
    for step in range((FLAGS.max_steps-FLAGS.test_point)//params.train_save):
        precision = model.test_model(FLAGS.test_point+step * params.train_save)
        result.append(precision)
        if precision > best_result:
            best_result = precision
        print('step:', step)
        print('precision:', precision)
        print('best_result:', best_result)

    sio.savemat('result.mat', {'prec': result})


if __name__ == '__main__':
    tf.app.run()
