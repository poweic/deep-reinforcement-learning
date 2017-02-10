import os
import gym
import inspect
import logging
import tensorflow as tf

gym.configuration.undo_logger_setup()

tf.logging.set_verbosity(tf.logging.INFO)
tf.logging._handler.setFormatter(
    logging.Formatter('[%(asctime)s %(file)s:%(line)s] %(message)s')
)

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
