import os
import gym
import inspect
import logging
import tensorflow as tf
from ac.utils import mkdir_p
FLAGS = tf.flags.FLAGS

# Disable gym default logger
gym.configuration.undo_logger_setup()

# Turn on TensorFlow logging system
tf.logging.set_verbosity(tf.logging.INFO)

# Set format
fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt="%m-%d %H:%M:%S")
tf.logging._handler.setFormatter(fmt)

# Use the same format and dump it to log file
if FLAGS.log_file is not None:
    FLAGS.log_file = FLAGS.log_dir + FLAGS.log_file

    fh = logging.FileHandler(FLAGS.log_file)
    fmt = logging.Formatter('[%(asctime)s %(file)s:%(line)s] %(message)s')
    fh.setFormatter(fmt)
    tf.logging._logger.addHandler(fh)

    FLAGS.stats_file = FLAGS.log_dir + FLAGS.stats_file

cwd = os.getcwd() + "/"

def my_logger_factory(level):

    def log(msg):
        _, filename, line, function_name, _, _ = inspect.getouterframes(
            inspect.currentframe())[1]
        filename = filename.replace(cwd, "")
        tf.logging._logger.log(level, msg, extra={ 'file': filename, 'line': line })

    return log

tf.logging.info = my_logger_factory(tf.logging.INFO)
tf.logging.warn = my_logger_factory(tf.logging.WARN)
tf.logging.error = my_logger_factory(tf.logging.ERROR)
