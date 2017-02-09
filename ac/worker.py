import traceback
import itertools
import tensorflow as tf
FLAGS = tf.flags.FLAGS

class Worker(object):
    """
    An A3C worker thread. Runs episodes locally and updates global shared value and policy nets.

    Args:
    name: A unique name for this worker
    env: The Gym environment used by this worker
    global_net: Instance of the globally shared network
    global_counter: Iterator that holds the global step
    summary_writer: A tf.train.SummaryWriter for Tensorboard summaries
    """
    def __init__(self, name, env, global_counter, global_net, add_summaries,
                 n_agents=1):

        self.name = name
        self.env = env
        self.global_counter = global_counter
        self.global_net = global_net
        self.add_summaries = add_summaries
        self.n_agents = n_agents

        # Get global variables and flags
        self.global_step = tf.contrib.framework.get_global_step()
        self.discount_factor = FLAGS.discount_factor
        self.max_global_steps = FLAGS.max_global_steps

        # Create local policy/value nets that are not updated asynchronously
        with tf.variable_scope(name):
            self.local_net = self.Estimator(add_summaries)
        self.set_global_net(global_net)

        # Initialize counter, maximum return of this worker, and summary_writer
        self.counter = 0
        self.max_return = 0
        self.summary_writer = None

        self.local_counter = itertools.count()

        self.reset_env()

    def run(self, sess, coord):

        self.sess = sess
        self.coord = coord

        with sess.as_default(), sess.graph.as_default():
            try:
                while not coord.should_stop() and not Worker.stop:
                    self._run()
            except tf.errors.CancelledError:
                return
            except:
                print "\33[91m"
                traceback.print_exc()

Worker.stop = False
