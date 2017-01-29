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
            self.advantages = tf.placeholder(tf.float32, [batch_size, 1], "advantages")
            self.returns = tf.placeholder(tf.float32, [batch_size, 1], "returns")
            self.actions_ext = tf.placeholder(tf.float32, [batch_size, 2], "actions_ext")

        with tf.variable_scope("shared"):
            shared = build_shared_network(self.state, add_summaries=add_summaries)

        with tf.name_scope("policy_network"):
            self.mu, self.sigma = policy_network(shared, 2)
            normal_dist = self.get_normal_dist(self.mu, self.sigma)
            self.actions = self.sample_actions(normal_dist)

        with tf.name_scope("state_value_network"):
            self.logits = state_value_network(shared)

        with tf.name_scope("losses"):
            self.pi_loss = self.get_policy_loss(normal_dist)
            self.entropy = self.get_exploration_loss(normal_dist)
            self.vf_loss = self.get_value_loss()

            self.loss = self.pi_loss + self.vf_loss + FLAGS.entropy_cost_mult * self.entropy

        with tf.name_scope("regularization"):
            self.reg_loss = self.get_reg_loss()
            self.loss += FLAGS.l2_reg * self.reg_loss

        with tf.name_scope("grads_and_optimizer"):
            # self.optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            self.optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate)
            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            self.grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
            self.g_mu = tf.gradients(self.pi_loss, [self.mu])[0]

        if add_summaries:

            # ====== DEBUG ======
            '''
            g_mu_mean = tf.reduce_mean(tf.reshape(self.g_mu[:, 1], (FLAGS.n_agents_per_worker, -1)), axis=0)
            mu_mean = tf.reduce_mean(tf.reshape(self.mu[:, 1], (FLAGS.n_agents_per_worker, -1)), axis=0)

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

    def get_policy_loss(self, normal_dist):
        # policy loss is the negative of log_prob times advantages
        with tf.name_scope("policy_loss"):
            self.log_prob = self.compute_log_prob(normal_dist, self.actions_ext)
            pi_loss = -tf.reduce_mean(self.log_prob * self.advantages)

        return pi_loss

    def get_value_loss(self):
        # loss of value function (L2-norm)
        with tf.name_scope("value_loss"):
            logits = tf_print(self.logits, "logits = ")
            returns = tf_print(self.returns, "returns = ")
            vf_loss = 0.5 * tf.reduce_mean(tf.square(logits - returns))

        return vf_loss

    def get_exploration_loss(self, normal_dist):
        with tf.name_scope("exploration_entropy"):
            # Add entropy cost to encourage exploration
            entropy = tf.reduce_mean(normal_dist.entropy())

        return entropy

    def get_reg_loss(self):
        vscope = tf.get_variable_scope().name
        weights = [
            v for v in tf.trainable_variables()
            if vscope in v.name and ("W" in v.name or "weights" in v.name)
        ]
        reg_losses = tf.add_n([tf.reduce_sum(w * w) for w in weights])
        return reg_losses

    def get_normal_dist(self, mu, sigma):

        # Add some initial bias for debugging (to see whether it can recover from it)
        '''
        mu_steer = mu[:, 1:2] + 15. * np.pi / 180 * np.sign(np.random.rand() - 0.5)
        # mu_steer = tf_print(mu_steer, "mu_steer = ")
        mu = tf.concat(1, [mu[:, 0:1], mu_steer])
        self.mu = mu
        '''

        # Reshape and sample
        mu = tf.reshape(mu, [-1])
        sigma = tf.reshape(sigma, [-1])

        '''
        # For debugging
        mu = tf_print(mu, "mu = ")
        sigma = tf_print(sigma, "sigma = ")
        '''

        # Create normal distribution and sample some actions
        normal_dist = tf.contrib.distributions.Normal(mu, sigma)

        return normal_dist

    def sample_actions(self, dist):
        actions = tf.reshape(dist.sample_n(1), [-1, 2])

        # Extract steer from actions (2rd column), turn it to yawrate, and
        # concatenate it back
        with tf.name_scope("steer_to_yawrate"):
            vf = actions[:, 0:1]
            steer = actions[:, 1:2]
            # steer = tf_print(steer, "steer = ")
            yawrate = steer_to_yawrate(steer, self.state["vehicle_state"][:, 4:5])
            # yawrate = tf_print(yawrate, "yawrate = ")
            actions = tf.concat(1, [vf, yawrate])

        return actions

    def compute_log_prob(self, dist, actions):
        # Extract yawrate from actions (2rd column), turn it to steer, and
        # concatenate it back
        with tf.name_scope("yawrate_to_steer"):
            vf = actions[:, 0:1]
            yawrate = actions[:, 1:2]
            # yawrate = tf_print(yawrate, "yawrate = ")
            steer = yawrate_to_steer(yawrate, self.state["vehicle_state"][:, 4:5])
            # steer = tf_print(steer, "steer = ")
            actions = tf.concat(1, [vf, steer])

        with tf.name_scope("compute_log_prob"):
            # actions = tf_print(actions, "actions = ")
            reshaped_actions = tf.reshape(actions, [-1])
            log_prob = dist.log_prob(reshaped_actions)
            # log_prob = tf_print(log_prob, "log_prob = ")
            log_prob = tf.reshape(log_prob, [-1, 2])

        # Compute z-score for debugging
        '''
        self.z = (actions - self.mu) / self.sigma
        self.z_mean = tf.reduce_mean(self.z)
        self.z_rms = tf.reduce_mean(tf.square(self.z))

        const = -0.5 * np.log(2 * np.pi)
        self.log_prob2 = const - tf.log(self.sigma) - 0.5 * tf.square(self.z)
        self.log_prob_diff = tf.reduce_sum(tf.square(self.log_prob2 - log_prob))
        '''

        return log_prob

    def predict(self, state, tensors, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.state[k]: state[k] for k in state.keys() }
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
            tensors += [self.mu, self.sigma]
        return self.predict(state, tensors, sess)

    def predict_actions(self, state, sess=None):
        tensors = [self.actions]
        return self.predict(state, tensors, sess)

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

        tf.summary.scalar("mean_mu_vf", tf.reduce_mean(self.mu[:, 0]) / 3.6)
        tf.summary.scalar("mean_mu_steer", tf.reduce_mean(self.mu[:, 1]) / np.pi * 180)

        tf.summary.scalar("mean_sigma_vf", tf.reduce_mean(self.sigma[:, 0]) / 3.6)
        tf.summary.scalar("mean_sigma_steer", tf.reduce_mean(self.sigma[:, 1]) / np.pi * 180)

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
