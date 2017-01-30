import numpy as np
import tensorflow as tf
from ac.utils import *
from ac.models import *
import ac.a3c.worker
# from ac.a3c.worker import Worker
FLAGS = tf.flags.FLAGS
batch_size = FLAGS.batch_size

class A3CEstimator():
    """
    Policy Function approximator. 
    """

    def __init__(self, add_summaries=False, trainable=True):

        with tf.name_scope("inputs"):
            self.state = get_state_placeholder()

        with tf.name_scope("outputs"):
            self.advantages  = tf.placeholder(tf.float32, [seq_length, batch_size, 1], "advantages")
            self.returns     = tf.placeholder(tf.float32, [seq_length, batch_size, 1], "returns")
            self.actions_ext = tf.placeholder(tf.float32, [seq_length, batch_size, 2], "actions_ext")

        with tf.variable_scope("shared"):
            shared = build_shared_network(self.state, add_summaries=add_summaries)

        with tf.name_scope("policy_network"):
            self.pi = self.policy_network(shared, 2)
            self.actions = tf.squeeze(self.pi.sample(1), 0)

        with tf.name_scope("state_value_network"):
            self.logits = state_value_network(shared)

        with tf.name_scope("losses"):
            self.pi_loss = self.get_policy_loss(self.pi)
            self.entropy = self.get_exploration_loss(self.pi)
            self.vf_loss = self.get_value_loss(self.logits, self.returns)

            self.loss = self.pi_loss + self.vf_loss + FLAGS.entropy_cost_mult * self.entropy

        with tf.name_scope("regularization"):
            self.reg_loss = self.get_reg_loss()
            self.loss += FLAGS.l2_reg * self.reg_loss

        with tf.name_scope("grads_and_optimizer"):
            # self.optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            self.optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate)
            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            self.grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]

            # Collect all trainable variables initialized here
            self.var_list = [v for g, v in self.grads_and_vars]
            # self.g_mu = tf.gradients(self.pi_loss, [self.pi.mu])[0]

        if add_summaries:

            # ====== DEBUG ======
            '''
            g_mu_mean = tf.reduce_mean(tf.reshape(self.g_mu[:, 1], (FLAGS.n_agents_per_worker, -1)), axis=0)
            mu_mean = tf.reduce_mean(tf.reshape(self.pi.mu[:, 1], (FLAGS.n_agents_per_worker, -1)), axis=0)

            tf.summary.scalar("g_mu_mean_0", tf.reduce_sum(g_mu_mean[0]))
            tf.summary.scalar("g_mu_mean_1", tf.reduce_sum(g_mu_mean[1]))
            tf.summary.scalar("g_mu_mean_2", tf.reduce_sum(g_mu_mean[2]))
            tf.summary.scalar("g_mu_mean_3", tf.reduce_sum(g_mu_mean[3]))

            tf.summary.scalar("mu_mean_0", tf.reduce_sum(mu_mean[0]))
            tf.summary.scalar("mu_mean_1", tf.reduce_sum(mu_mean[1]))
            tf.summary.scalar("mu_mean_2", tf.reduce_sum(mu_mean[2]))
            tf.summary.scalar("mu_mean_3", tf.reduce_sum(mu_mean[3]))
            '''
            # ====== DEBUG ======

            self.summaries = self.summarize()

    def get_policy_loss(self, pi):
        # policy loss is the negative of log_prob times advantages
        with tf.name_scope("policy_loss"):
            self.log_prob = self.compute_log_prob(pi, self.actions_ext)
            pi_loss = -tf.reduce_mean(self.log_prob * self.advantages)

        return pi_loss

    def get_value_loss(self, logits, returns):
        # loss of value function (L2-norm)
        with tf.name_scope("value_loss"):
            # logits = tf_print(logits, "logits = ")
            # returns = tf_print(returns, "returns = ")
            vf_loss = 0.5 * tf.reduce_mean(tf.square(logits - returns))

        return vf_loss

    def get_exploration_loss(self, pi):
        with tf.name_scope("exploration_entropy"):
            # Add entropy cost to encourage exploration
            entropy = tf.reduce_mean(pi.dist.entropy())

        return entropy

    def get_reg_loss(self):
        vscope = tf.get_variable_scope().name
        weights = [
            v for v in tf.trainable_variables()
            if vscope in v.name and ("W" in v.name or "weights" in v.name)
        ]
        reg_losses = tf.add_n([tf.reduce_sum(w * w) for w in weights])
        return reg_losses

    def get_forward_velocity(self):
        return self.state.vehicle_state[..., 4:5]

    def policy_network(self, input, num_outputs):

        # mu: [B, 2], sigma: [B, 2], phi is just a syntatic sugar
        mu, sigma = policy_network(input, num_outputs)

        # Add naive policy as baseline bias
        naive_mu = naive_mean_steer_policy(self.state.front_view)
        mu = tf.pack([mu[..., 0], mu[..., 1] + naive_mu], axis=-1)

        # Convert mu_steer (mu[..., 1]) to mu_yawrate
        mu = s2y(mu, self.get_forward_velocity())

        # mu = tf_print(mu, "mu = ")
        # sigma = tf_print(sigma, "sigma = ")

        pi = create_distribution(mu, sigma)

        return pi

    def compute_log_prob(self, pi, actions):

        with tf.name_scope("compute_log_prob"):
            log_prob = pi.log_prob(actions)[..., None]

        # Compute z-score for debugging
        z = (actions - pi.mu) / pi.sigma
        # self.z = z = tf_print(z, "z = ")
        z_mean = tf.reduce_mean(z)
        z_rms = tf.reduce_mean(tf.square(z))
        # z_mean = tf_print(z_mean, "z_mean = ")

        const = -0.5 * np.log(2 * np.pi)
        log_prob2 = const - tf.log(pi.sigma) - 0.5 * tf.square(z)
        log_prob2 = tf.reduce_sum(log_prob2, axis=2, keep_dims=True)
        # log_prob2 = tf_print(log_prob2, "log_prob2 = ")

        log_prob_diff = tf.reduce_sum(tf.square(log_prob2 - log_prob))
        # log_prob_diff = tf_print(log_prob_diff, "log_prob diff = ")

        return log_prob + 0 * log_prob_diff + 0 * z_mean

    def to_feed_dict(self, state):
        rank_a = len(self.state.prev_reward.get_shape())
        rank_b = state.prev_reward.ndim

        feed_dict = {
            self.state[k]: state[k] if rank_a == rank_b else state[k][None, ...]
            for k in state.keys()
        }

        return feed_dict

    def predict(self, state, tensors, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = self.to_feed_dict(state)
        result = sess.run(tensors, feed_dict)

        if len(result) == 1:
            result = result[0]
            if result.size == 1:
                return np.asscalar(result)
            else:
                return result
        else:
            return result

    def predict_values(self, state, sess=None, debug=False):
        tensors = [self.logits]
        if debug:
            tensors += [self.pi.mu, self.pi.sigma]
        return self.predict(state, tensors, sess)

    def predict_actions(self, state, sess=None):
        actions = self.predict(state, [self.actions], sess)[0, ...]
        return actions

    def summarize(self):
        with tf.name_scope("summaries"):
            self.summarize_gradient_norm()
            self.summarize_policy_estimator()
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

            # tf.summary.scalar("gradient/clipped_mu", tf.reduce_mean(self.g_mu))

    def summarize_value_estimator(self):
        tf.summary.scalar("vf_loss", self.vf_loss)

        tf.summary.scalar("max_value", tf.reduce_max(self.logits))
        tf.summary.scalar("min_value", tf.reduce_min(self.logits))
        tf.summary.scalar("mean_value", tf.reduce_mean(self.logits))

        tf.summary.scalar("max_advantage", tf.reduce_max(self.advantages))
        tf.summary.scalar("min_advantage", tf.reduce_min(self.advantages))
        tf.summary.scalar("mean_advantage", tf.reduce_mean(self.advantages))

A3CEstimator.Worker = ac.a3c.worker.A3CWorker
