import numpy as np
import tensorflow as tf
from drl.ac.utils import *
from drl.ac.models import *
from drl.ac.policies import build_policy
import drl.ac.a3c.worker
FLAGS = tf.flags.FLAGS
batch_size = FLAGS.batch_size
seq_length = FLAGS.seq_length

class A3CEstimator():
    """
    Policy Function approximator. 
    """

    def __init__(self, add_summaries=False, trainable=True):

        with tf.name_scope("inputs"):
            self.state = get_state_placeholder()

        with tf.name_scope("outputs"):
            self.advantages  = tf.placeholder(FLAGS.dtype, [seq_length, batch_size, 1], "advantages")
            self.returns     = tf.placeholder(FLAGS.dtype, [seq_length, batch_size, 1], "returns")
            self.actions_ext = tf.placeholder(FLAGS.dtype, [seq_length, batch_size, FLAGS.num_actions], "actions_ext")

        with tf.variable_scope("shared"):
            shared, self.lstm = build_network(self.state, add_summaries=add_summaries)
            shared = tf_check_numerics(shared)

        with tf.name_scope("policy"):
            self.pi, _ = build_policy(shared, FLAGS.policy_dist)
            actions = tf.squeeze(self.pi.sample_n(1), 0)
            self.actions = tf_print(actions)

            self.action_and_stats = [self.actions, self.pi.stats]

        with tf.name_scope("state_value_network"):
            self.value = state_value_network(shared)

        with tf.name_scope("losses"):
            self.pi_loss = self.get_policy_loss(self.pi)
            entropy = self.get_exploration_loss(self.pi)
            self.entropy_loss = -entropy * FLAGS.entropy_cost_mult
            self.vf_loss = self.get_value_loss(self.value, self.returns)

            for loss in [self.pi_loss, self.vf_loss, self.entropy_loss]:
                assert len(loss.get_shape()) == 0

            self.loss = (
                self.pi_loss +
                self.vf_loss * FLAGS.lr_vp_ratio +
                self.entropy_loss
            )

        with tf.name_scope("regularization"):
            self.reg_loss = self.get_reg_loss()
            self.loss += FLAGS.l2_reg * self.reg_loss

        with tf.name_scope("grads_and_optimizer"):
            self.optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate)
            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            check_none_grads(grads_and_vars)
            self.grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]

            # Collect all trainable variables initialized here
            self.var_list = [v for g, v in self.grads_and_vars]

        self.summaries = self.summarize(add_summaries)

    def get_policy_loss(self, pi):
        # policy loss is the negative of log_prob times advantages
        with tf.name_scope("policy_loss"):
            actions = tf_print(self.actions_ext)
            self.log_prob = pi.log_prob(actions)[..., None]
            pi_loss = self.log_prob * self.advantages
            pi_loss = -tf.reduce_sum(tf.reduce_mean(pi_loss, axis=[1, 2]), axis=0)

        return pi_loss

    def get_value_loss(self, value, returns):
        # loss of value function (L2-norm)
        with tf.name_scope("value_loss"):
            vf_loss = tf.square(value - returns)
            vf_loss = 0.5 * tf.reduce_sum(tf.reduce_mean(vf_loss, axis=[1, 2]), axis=0)

        return vf_loss

    def get_exploration_loss(self, pi):
        return tf.reduce_sum(tf.reduce_mean(self.pi.entropy(), axis=1), axis=0)

    def get_reg_loss(self):
        vscope = tf.get_variable_scope().name
        weights = [
            v for v in tf.trainable_variables()
            if vscope in v.name and ("W" in v.name or "weights" in v.name)
        ]
        reg_losses = tf.add_n([tf.reduce_sum(w * w) for w in weights])
        return reg_losses

    def to_feed_dict(self, state):
        rank_a = len(self.state.prev_reward.get_shape())
        rank_b = state.prev_reward.ndim

        feed_dict = {
            self.state[k]: state[k] if rank_a == rank_b else state[k][None, ...]
            for k in state.keys()
        }

        return feed_dict

    def reset_lstm_state(self):
        self.lstm.prev_state_out = None

    def predict(self, tensors, feed_dict, sess=None):
        sess = sess or tf.get_default_session()

        B = feed_dict[self.state.prev_reward].shape[1]
        fill_lstm_state_placeholder(self.lstm, feed_dict, B)

        output, self.lstm.prev_state_out = sess.run([
            tensors, self.lstm.state_out
        ], feed_dict)

        return output

    def predict_values(self, state, sess=None):
        feed_dict = self.to_feed_dict(state)
        return self.predict(self.value, feed_dict, sess)

    def predict_actions(self, state, sess=None):
        feed_dict = self.to_feed_dict(state)
        actions, stats = self.predict(self.action_and_stats, feed_dict, sess)
        actions = actions[0, ...].T
        return actions, stats

    def summarize(self, add_summaries):
        return tf.no_op()

        if not add_summaries:
            return tf.no_op()

        with tf.name_scope("summaries"):
            self.summarize_gradient_norm()
            # self.summarize_policy_estimator()
            self.summarize_value_estimator()
            tf.summary.scalar("total_loss", self.loss)

        return tf.summary.merge_all()

    def summarize_policy_estimator(self):
        tf.summary.scalar("pi_loss", self.pi_loss)

        tf.summary.scalar("entropy", self.entropy)

        tf.summary.scalar("mean_mu_vf", tf.reduce_mean(self.pi.mu[..., 0]) / 3.6)
        tf.summary.scalar("mean_mu_steer", tf.reduce_mean(self.pi.mu[..., 1]) / np.pi * 180)

        tf.summary.scalar("mean_sigma_vf", tf.reduce_mean(self.pi.sigma[..., 0]) / 3.6)
        tf.summary.scalar("mean_sigma_steer", tf.reduce_mean(self.pi.sigma[..., 1]) / np.pi * 180)

    def summarize_gradient_norm(self):
        with tf.variable_scope("gradient"):
            for g, v in self.grads_and_vars:
                tf.summary.scalar("gradient/" + v.name, tf.reduce_mean(g * g))

    def summarize_value_estimator(self):
        tf.summary.scalar("vf_loss", self.vf_loss)

        tf.summary.scalar("max_value", tf.reduce_max(self.value))
        tf.summary.scalar("min_value", tf.reduce_min(self.value))
        tf.summary.scalar("mean_value", tf.reduce_mean(self.value))

        tf.summary.scalar("max_advantage", tf.reduce_max(self.advantages))
        tf.summary.scalar("min_advantage", tf.reduce_min(self.advantages))
        tf.summary.scalar("mean_advantage", tf.reduce_mean(self.advantages))

A3CEstimator.Worker = drl.ac.a3c.worker.A3CWorker
