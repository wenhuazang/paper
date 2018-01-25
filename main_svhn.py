import tensorflow as tf
from model_svhn import DANN
import scipy.io as sio
import os
import pprint
import params

flags = tf.app.flags
flags.DEFINE_integer("max_steps", 40000, "maximum step to train model")
flags.DEFINE_integer("d_iter", 1, "d iter time each g iter")
flags.DEFINE_integer("decay_steps", 4000, "steps to change learning rate")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("decay_factor", 0.92, "the change of ration of learning rate")
flags.DEFINE_float("alpha", 0.5, "the regularized of dan_2s")
flags.DEFINE_float("beta", .8, "the regularized of target loss")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [64]")
flags.DEFINE_integer("test_point", 0, "The beginning of point of test")
flags.DEFINE_integer("save_size", 100, "The size of save parameter [100]")
flags.DEFINE_float("fraction", 0.3, "per process gpu memory fraction")
flags.DEFINE_string("gpu_num", "0", "number of gpu")
flags.DEFINE_string("source_name", "svhn_32", "the filename of source data[books, electronics, kitchen]")
flags.DEFINE_string("target_name", "mnist_32", "the filename of target data")
flags.DEFINE_string("is_dan", "dan", "whether using distributional adversarial network")
flags.DEFINE_string("source_data_dir", "./data/svhn", "the path of source data")
flags.DEFINE_string("target_data_dir", "./data/mnist", "the path of target data")
flags.DEFINE_string("saver_path", "./saver_svhn_mnist_32_1000",
                    "Directory name to save the checkpoints [checkpoint]")

FLAGS = flags.FLAGS
pp = pprint.PrettyPrinter()


def main(_):
    pp.pprint(flags.FLAGS.__flags)
    if not os.path.exists(FLAGS.saver_path):
        os.mkdir(FLAGS.saver_path)
    model = DANN(FLAGS)
    result = []
    best_result = 0
    best_step = 0
    # model.train_model()
    for step in range((FLAGS.max_steps - FLAGS.test_point) // params.train_save):
        precision = model.test_model(FLAGS.test_point + step * params.train_save)
        result.append(precision)
        if precision > best_result:
            best_result = precision
            best_step = FLAGS.test_point + step * params.train_save
        print('best_step:', best_step)
        print('precision:', precision)
        print('best_result:', best_result)

    sio.savemat('svhn_result.mat', {'svhn_prec': result})


if __name__ == '__main__':
    tf.app.run()
