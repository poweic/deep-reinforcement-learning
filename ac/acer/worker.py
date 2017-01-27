import itertools
import tensorflow as tf
import ac.acer.estimators
from ac.utils import *

class Worker(object):
    """
    An A3C worker thread. Runs episodes locally and updates global shared value and policy nets.

    Args:
    name: A unique name for this worker
    env: The Gym environment used by this worker
    global_net: Instance of the globally shared network
    global_counter: Iterator that holds the global step
    discount_factor: Reward discount factor
    summary_writer: A tf.train.SummaryWriter for Tensorboard summaries
    max_global_steps: If set, stop coordinator when global_counter > max_global_steps
    """
    def __init__(self, name, env, global_counter, global_net, add_summaries, n_agents=1, discount_factor=0.99, max_global_steps=None):
        self.name = name
        self.discount_factor = discount_factor
        self.max_global_steps = max_global_steps
        self.global_step = tf.contrib.framework.get_global_step()
        self.global_counter = global_counter
        self.local_counter = itertools.count()
        self.n_agents = n_agents
        self.env = env

        # Create local policy/value nets that are not updated asynchronously
        with tf.variable_scope(name):
            self.local_net = ac.acer.estimators.AcerEstimator(add_summaries)

        self.set_global_net(global_net)

        # Initialize counter, maximum return of this worker, and summary_writer
        self.counter = 0
        self.max_return = 0
        self.summary_writer = None

        self.reset_env()

    def set_global_net(self, global_net):
        # Get global, local, and the average net var_list
        global_vars = global_net.var_list
        local_vars = self.local_net.var_list

        # Operation to copy params from global net to local net
        self.copy_params_op = make_copy_params_op(global_vars, local_vars)

        self.global_net = global_net

        avg_vars = ac.acer.estimators.AcerEstimator.average_net.var_list
        copy_global_to_avg = make_copy_params_op(global_vars, avg_vars)

        init_avg_net = tf.cond(
            tf.equal(self.global_step, 0),
            lambda: copy_global_to_avg,
            lambda: tf.no_op()
        )

        init_avg_net = tf.Print(init_avg_net, [tf.constant(0)], message="init_avg_net called")

        with tf.control_dependencies([init_avg_net]):
            self.train_op = make_train_op(self.local_net, global_net)

    def reset_env(self):
        # TODO
        pass

    def run(self, sess, coord):

        self.sess = sess

        with sess.as_default(), sess.graph.as_default():

            try:
                while not coord.should_stop():
                    # Copy Parameters from the global networks
                    sess.run(self.copy_params_op)

                    # TODO
                    # Resample experiences from memory

                    if self.max_global_steps is not None and global_t >= self.max_global_steps:
                        tf.logging.info("Reached global step {}. Stopping.".format(global_t))
                        coord.request_stop()
                        return

                    # Update the global networks
                    self.update()

            except tf.errors.CancelledError:
                return
            except:
                traceback.print_exc()

    def update(self):

        # TODO
        # update the network

        self.counter += 1
