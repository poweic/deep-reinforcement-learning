import sys
import tensorflow as tf
from drl.ac.utils import *
from drl.ac.distributions import *
from drl.ac.models import *
from drl.ac.policies import build_policy
from drl.ac.estimators import *
from drl.ac.worker import Worker
from drl.ac.qprop.worker import QPropWorker
import threading

FLAGS = tf.flags.FLAGS
batch_size = FLAGS.batch_size
seq_length = FLAGS.seq_length

class QPropEstimator():
    def __init__(self, add_summaries=False, trainable=True, use_naive_policy=True):

        self.trainable = trainable

        self.avg_net = getattr(QPropEstimator, "average_net", self)

        scope_name = tf.get_variable_scope().name + '/'

        with tf.name_scope("inputs"):
            self.seq_length = tf.placeholder(tf.int32, [], "seq_length")
            self.state = get_state_placeholder()
            self.a = tf.placeholder(FLAGS.dtype, [seq_length, batch_size, FLAGS.num_actions], "actions")
            self.r = tf.placeholder(FLAGS.dtype, [seq_length, batch_size, 1], "rewards")
            self.adv = tf.placeholder(FLAGS.dtype, [seq_length, batch_size, 1], "advantages")
            self.done = tf.placeholder(tf.bool, [batch_size, 1], "done")

        with tf.variable_scope("shared"):
            shared, self.lstm = build_network(self.state, scope_name, add_summaries)

        # For k-step rollout s_i, i = 0, 1, ..., k-1, we need one additional
        # state s_k s.t. we can bootstrap value from it, i.e. we need V(s_k)
        with tf.variable_scope("value-network"):
            self.value_all = value = state_value_network(shared)
            value *= tf.Variable(1, dtype=FLAGS.dtype, name="value_scale", trainable=FLAGS.train_value_scale)
            self.value_last = value[-1:, ...] * tf.cast(~self.done, FLAGS.dtype)[None, ...]
            self.value = value[:self.seq_length, ...]

        with tf.variable_scope("shared-policy"):
            if not FLAGS.share_network:
                # FIXME right now this only works for non-lstm version
                shared, lstm2 = build_network(self.state, scope_name, add_summaries)
                self.lstm.inputs.update(lstm2.inputs)
                self.lstm.outputs.update(lstm2.outputs)

            shared = shared[:self.seq_length, ...]

        self.state.update(self.lstm.inputs)

        with tf.variable_scope("policy-network"):
            self.pi, self.pi_behavior = build_policy(shared, FLAGS.policy_dist)
            self.pi_mean = self.pi.dist.mean()

        with tf.name_scope("output"):
            self.a_prime = tf.squeeze(self.pi.sample_n(1), 0)
            self.action_and_stats = [self.a_prime, self.pi.stats]

        with tf.variable_scope("A"):
            Q_tilt = stochastic_dueling_network(shared, self.value, self.pi)

        with tf.variable_scope("Q"):
            self.Q_tilt_a = Q_tilt(self.a, name="Q_tilt_a")
            self.Q_tilt_mu = Q_tilt(self.pi_mean, name="Q_w_at_mu")

            # Compute the importance sampling weight \rho and \rho^{'}
            with tf.name_scope("rho"):
                self.rho = compute_rho(self.a, self.pi, self.pi_behavior)

            with tf.name_scope("c_i"):
                self.c = tf.minimum(tf_const(1.), self.rho ** (1. / FLAGS.num_actions), "c_i")
                tf.logging.info("c.shape = {}".format(tf_shape(self.c)))

            with tf.name_scope("Q_Retrace"):
                self.Q_ret, self.Q_opc = compute_Q_ret_Q_opc(
                    self.value, self.value_last, self.c, self.r, self.Q_tilt_a
                )

        with tf.name_scope("losses"):
            # Surrogate loss is the loss tensor we passed to optimizer for
            # automatic gradient computation, it uses lots of stop_gradient.
            # Therefore it's different from the true loss (self.loss)

            self.pi_loss, self.pi_loss_sur = self.get_policy_loss(
                self.rho, self.pi, self.a, self.Q_opc, self.value, self.Q_tilt_mu
            )

            self.vf_loss, self.vf_loss_sur = self.get_value_loss(
                self.Q_ret, self.Q_tilt_a, self.rho, self.value
            )

            self.entropy, self.entropy_loss = exploration_loss(self.pi)

            for loss in [self.pi_loss_sur, self.vf_loss_sur, self.entropy_loss]:
                assert len(loss.get_shape()) == 0

            self.loss = self.pi_loss + self.vf_loss + self.entropy_loss

        with tf.name_scope("grads_and_optimizer"):

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):

                global_step = FLAGS.global_step

                self.lr = tf.train.exponential_decay(
                    tf_const(FLAGS.learning_rate), FLAGS.global_timestep,
                    FLAGS.decay_steps, FLAGS.decay_rate, staircase=FLAGS.staircase
                )

                tvars = tf.trainable_variables()

                # optimizer for on-policy actor update
                self.optimizer = tf.train.AdamOptimizer(self.lr)
                pi_var_list = [v for v in tvars if "shared" in v.name or "policy-network" in v.name]
                self.grads_and_vars, self.global_norm = compute_gradients_with_checks(
                    self.optimizer, self.pi_loss_sur + self.entropy_loss, pi_var_list)

                # optimizer for off-policy critic update
                self.vf_optimizer = tf.train.AdamOptimizer(self.lr)
                vf_var_list = [v for v in tvars if "value-network" in v.name or "advantage" in v.name]
                self.grads_and_vars, self.global_norm = compute_gradients_with_checks(
                    self.vf_optimizer, self.vf_loss_sur, vf_var_list)

                """
                unused = set(tvars) - set(pi_var_list) - set(vf_var_list)
                print "\n".join([v.name for v in unused]), "\n============="
                """

            # Collect all trainable variables initialized here
            self.var_list = [v for g, v in self.grads_and_vars]

        self.summaries = self.summarize(add_summaries)

    def predict_values(self, state, seq_length, sess=None):
        feed_dict = to_feed_dict(self, state)
        feed_dict[self.seq_length] = seq_length
        return self.predict(self.value_all, feed_dict, sess)[0]

    def get_initial_hidden_states(self, batch_size):
        return get_lstm_initial_states(self.lstm.inputs, batch_size)

    def predict(self, tensors, feed_dict, sess=None):
        sess = sess or tf.get_default_session()

        B = feed_dict[self.state.prev_reward].shape[1]

        output, hidden_states = sess.run([
            tensors, self.lstm.outputs
        ], feed_dict)

        return output, hidden_states

    def update(self, tensors, feed_dict, sess=None):
        sess = sess or tf.get_default_session()

        with Worker.lock:
            output = sess.run(tensors, feed_dict)

        return output

    def predict_actions(self, state, sess=None):

        feed_dict = to_feed_dict(self, state)
        feed_dict[self.seq_length] = 1

        (a_prime, stats), hidden_states = self.predict(self.action_and_stats, feed_dict, sess)

        a_prime = a_prime[0, ...].T

        return a_prime, stats, hidden_states

    def get_policy_loss(self, rho, pi, a, Q_opc, value, Q_tilt_mu):

        tf.logging.info("Computing policy loss ...")

        # mu is the average action under current policy
        mu = self.pi_mean
        g_Qw_at_mu = tf.gradients(Q_tilt_mu, mu)[0]
        A_bar = tf.reduce_sum(g_Qw_at_mu * (a - mu), axis=-1, keep_dims=True)

        self.log_a = log_a = pi.log_prob(a)[..., None]

        Cov_A_Abar = self.adv * A_bar

        # assuming covariance matrix of action is identity matrix
        # Var_A_bar = tf.reduce_sum(g_Qw_at_mu ** 2, axis=-1, keep_dims=True)
        Var_A_bar = A_bar ** 2

        ETA = {
            "adaptive":     lambda C, V: C / V,
            "conservative": lambda C, V: (tf.sign(C) + 1) / 2,
            "aggressive":   lambda C, V: tf.sign(C)
        }
        eta = ETA[FLAGS.qprop_type](Cov_A_Abar, Var_A_bar)

        pi_obj = tf.stop_gradient(self.adv - eta * A_bar) * log_a + \
            tf.stop_gradient(eta) * tf.reduce_sum(g_Qw_at_mu * mu, axis=-1, keep_dims=True)

        pi_obj_sur, self.mean_KL = add_fast_TRPO_regularization(
            pi, self.avg_net.pi, pi_obj)

        # loss is the negative of objective function
        loss, loss_sur = -pi_obj, -pi_obj_sur

        return reduce_seq_batch_dim(loss, loss_sur)

    def get_value_loss(self, Q_ret, Q_tilt_a, rho, value):

        tf.logging.info("Computing value loss ...")

        Q_diff = tf.stop_gradient(Q_ret - Q_tilt_a)

        # L2 norm as loss function
        Q_l2_loss = 0.5 * tf.square(Q_diff)

        # surrogate loss function for L2-norm of Q and V, the derivatives of
        # (-Q_diff * Q_tilt_a) is the same as that of (0.5 * tf.square(Q_diff))
        Q_l2_loss_sur = -Q_diff * Q_tilt_a
        V_l2_loss_sur = -Q_diff * value * tf.minimum(tf_const(1.), rho)

        # Compute the objective function (obj) we try to maximize
        loss     = Q_l2_loss
        loss_sur = Q_l2_loss_sur + V_l2_loss_sur

        return reduce_seq_batch_dim(loss, loss_sur)

    def summarize(self, add_summaries):

        if not add_summaries:
            return tf.no_op()

        # sum over rewards along the sequence dimension to get total return
        # and take mean along the batch dimension
        self.total_return = tf.reduce_mean(tf.reduce_sum(self.r, axis=0))

        keys_to_summarize = [
            "vf_loss", "pi_loss", "entropy", "mean_KL", "loss",
            "total_return", "seq_length"
        ]

        tf.logging.info("Adding summaries ...")
        with tf.name_scope("summaries"):
            for key in keys_to_summarize:
                tf.summary.scalar(key, getattr(self, key))

        return tf.summary.merge_all()

    @staticmethod
    def create_averge_network():
        if "average_net" not in QPropEstimator.__dict__:
            with tf.variable_scope("average_net"):
                QPropEstimator.average_net = QPropEstimator(add_summaries=False, trainable=False)

QPropEstimator.Worker = QPropWorker
