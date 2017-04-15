import numpy as np
import tensorflow as tf
import scipy.io
import traceback
import time
import drl.ac.a3c.estimators
from drl.ac.worker import Worker
from drl.ac.utils import *

FLAGS = tf.flags.FLAGS

class A3CWorker(Worker):
    """
    An A3C worker thread. Runs episodes locally and updates global shared value and policy nets.

    Args:
    name: A unique name for this worker
    env: The Gym environment used by this worker
    global_net: Instance of the globally shared network
    global_counter: Iterator that holds the global step
    summary_writer: A tf.train.SummaryWriter for Tensorboard summaries
    """
    def __init__(self, **kwargs):
        self.Estimator = drl.ac.a3c.estimators.A3CEstimator
        super(A3CWorker, self).__init__(**kwargs)

    def set_global_net(self, global_net):
        global_vars = global_net.var_list
        local_vars = self.local_net.var_list
        self.copy_params_op = make_copy_params_op(global_vars, local_vars)

        self.global_net = global_net
        self.gstep = 0

        self.train_op = make_train_op(self.local_net, self.global_net)
        self.inc_global_step = tf.assign_add(self.global_step, 1)

        net = self.local_net
        self.step_op = [
            {
                'total': net.loss,
                'pi': net.pi_loss,
                'vf': net.vf_loss,
                'entropy': net.entropy_loss,
            },
            net.summaries,
            self.train_op,
            self.inc_global_step
        ]

    def _run(self):
        self.copy_params_from_global()

        # Collect rollout {(s_0, a_0, r_0, mu_0), (s_1, ...), ... }
        rollout = self.run_n_steps()

        # Even though A3C can't use experience replay, we still need store
        # experiences for playback visualization and statistics
        self.store_experience(rollout)

        # Update the global networks
        self.update(rollout)

        """
        mean, std, msg = self.global_episode_stats.last_n_stats()
        tf.logging.info("\33[93m" + msg + "\33[0m")
        """

    def update(self, rollout):

        if rollout.seq_length == 0:
            return

        rollout = self.get_partial_rollout(rollout, FLAGS.max_seq_length)

        """
        print "rollout.keys = {}".format(rollout.keys())
        for key in rollout.states:
            print "rollout.states[{}].shape = {}".format(key, rollout.states[key].shape)
        """

        net = self.local_net
        net.reset_lstm_state()
        gamma = self.discount_factor

        # Compute values and bootstrap from last state if not terminal (~done)
        # tf.logging.info("rollout.seq_length = {}, rollout.states.keys() = {}".format(rollout.seq_length, rollout.states.keys()))
        values = net.predict_values(rollout.states, self.sess)
        values[-1, rollout.done[-1]] = 0

        # Compute discounted total returns from rewards and values[-1]
        returns = discount(np.vstack([rollout.reward, values[-1:]]), gamma)[:-1]

        # Compute 1-step TD target
        delta_t = rollout.reward + gamma * values[1:] - values[:-1]

        # Compute Generalized Advantage Estimation (GAE) from 1-step TD target
        advantages = discount(delta_t, gamma * FLAGS.lambda_)

        # Fill feed_dict with values we just computed
        feed_dict = {
            net.advantages: advantages,
            net.returns: returns,
            net.actions_ext: rollout.action,
        }
        feed_dict.update({net.state[k]: v[:-1] for k, v in rollout.states.iteritems()})

        # Reset LSTM state and update
        net.reset_lstm_state()
        loss, summaries, _, self.gstep = net.predict(self.step_op, feed_dict, self.sess)
        loss = AttrDict(loss)

        tf.logging.info(pretty_float(
            "#{:6d}: pi_loss = %f, vf_loss = %f, entropy_loss = %f, total = %f [S = {}]"
        ).format(
            self.gstep, loss.pi, loss.vf, loss.entropy, loss.total, rollout.seq_length
        ))
