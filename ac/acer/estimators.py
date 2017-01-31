import tensorflow as tf
from tensorflow_helnet.layers import DenseLayer, Conv2DLayer, MaxPool2DLayer
from ac.utils import *
from ac.models import *
import ac.acer.worker

FLAGS = tf.flags.FLAGS
batch_size = FLAGS.batch_size
seq_length = FLAGS.seq_length

def flatten_all_leading_axes(x):
    return tf.reshape(x, [-1, x.get_shape()[-1].value])

class AcerEstimator():
    def __init__(self, add_summaries=False, trainable=True):

        self.trainable = trainable

        with tf.name_scope("inputs"):
            self.state = get_state_placeholder()
            self.a = tf.placeholder(tf.float32, [seq_length, batch_size, 2], "actions")
            self.r = tf.placeholder(tf.float32, [seq_length, batch_size, 1], "rewards")

        with tf.variable_scope("shared"):
            shared, self.lstm = build_shared_network(self.state, add_summaries=add_summaries)

        with tf.variable_scope("pi"):
            self.pi = self.policy_network(shared, 2)

        with tf.variable_scope("mu"):
            # Placeholder for behavior policy
            self.mu = create_distribution(
                mu    = tf.placeholder(tf.float32, [seq_length, batch_size, 2], "mu"),
                sigma = tf.placeholder(tf.float32, [seq_length, batch_size, 2], "sigma")
            )

        with tf.name_scope("output"):
            self.a_prime = tf.squeeze(self.pi.sample(1), 0)

        with tf.variable_scope("V"):
            self.value = state_value_network(shared)

        with tf.variable_scope("A"):
            adv = self.advantage_network(shared)
            Q_tilt = self.SDN_network(adv, self.value, self.pi)

        with tf.variable_scope("Q"):
            self.Q_tilt_a = Q_tilt(self.a, name="Q_tilt_a")
            self.Q_tilt_a_prime = Q_tilt(self.a_prime, name="Q_tilt_a_prime")

            # Compute the importance sampling weight \rho and \rho^{'}
            with tf.name_scope("rho"):
                self.rho, self.rho_prime = self.compute_rho(
                    self.a, self.a_prime, self.pi, self.mu
                )

            with tf.name_scope("c_i"):
                self.c = tf.minimum(1., self.rho ** (1. / 2), "c_i")
                print "c.shape = {}".format(tf_shape(self.c))

            with tf.name_scope("Q_Retrace"):
                self.Q_ret, self.Q_opc = self.compute_Q_ret_Q_opc_recursively(
                    self.value, self.c, self.r, self.Q_tilt_a
                )

        self.avg_net = getattr(AcerEstimator, "average_net", self)

        with tf.name_scope("ACER"):
            self.acer_g = self.compute_ACER_gradient(
                self.rho, self.pi, self.a, self.Q_opc, self.value,
                self.rho_prime, self.Q_tilt_a_prime, self.a_prime)

        with tf.name_scope("losses"):
            self.pi_loss = self.get_policy_loss(self.pi, self.acer_g)
            self.vf_loss = self.get_value_loss(
                self.Q_ret, self.Q_tilt_a, self.rho, self.value)

            z_mean = tf.reduce_mean((self.a - self.pi.mu) / self.pi.sigma)
            z_mean = tf_print(z_mean, "z_mean = ")

            self.loss = self.pi_loss + self.vf_loss # + 0 * z_mean

        self.optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate)
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]

        # Collect all trainable variables initialized here
        self.var_list = [v for g, v in self.grads_and_vars]

        # self.pi_var_list = get_var_list_wrt(self.pi.mu + self.pi.sigma)

    def compute_rho(self, a, a_prime, pi, mu):
        # rho = tf.zeros_like(self.value)
        # rho_prime = tf.zeros_like(self.value)
        # return rho, rho_prime

        # compute rho, rho_prime, and c
        with tf.name_scope("pi_a"):
            self.pi_a = pi_a = pi.prob(a)[..., None]
        with tf.name_scope("mu_a"):
            self.mu_a = mu_a = mu.prob(a)[..., None]

        with tf.name_scope("pi_a_prime"):
            self.pi_a_prime = pi_a_prime = pi.prob(a_prime)[..., None]
        with tf.name_scope("mu_a_prime"):
            self.mu_a_prime = mu_a_prime = mu.prob(a_prime)[..., None]

        # use tf.div instead of pi_a / mu_a to assign a name to the output
        rho = tf.div(pi_a, mu_a, name="rho")
        rho_prime = tf.div(pi_a_prime, mu_a_prime, name="rho_prime")

        print "pi_a.shape = {}".format(tf_shape(pi_a))
        print "mu_a.shape = {}".format(tf_shape(mu_a))
        print "pi_a_prime.shape = {}".format(tf_shape(pi_a_prime))
        print "mu_a_prime.shape = {}".format(tf_shape(mu_a_prime))
        print "rho.shape = {}".format(tf_shape(rho))
        print "rho_prime.shape = {}".format(tf_shape(rho_prime))

        return rho, rho_prime

    def compute_Q_ret_Q_opc_recursively(self, values, c, r, Q_tilt_a):
        """
        Use tf.while_loop to compute Q_ret, Q_opc
        """
        # Q_ret = tf.zeros_like(self.value)
        # Q_opc = tf.zeros_like(self.value)
        # return Q_ret, Q_opc

        gamma = tf.constant(FLAGS.discount_factor, dtype=tf.float32)
        # gamma = tf_print(gamma, "gamma = ")

        with tf.name_scope("initial_value"):
            # Use "done" to determine whether x_k is terminal state. If yes,
            # set initial Q_ret to 0. Otherwise, bootstrap initial Q_ret from V.
            Q_ret_0 = tf.zeros_like(values[0:1, ...]) # tf.zeros((1, batch_size, 1), dtype=tf.float32)
            Q_opc_0 = Q_ret_0

            Q_ret = Q_ret_0
            Q_opc = Q_opc_0

            '''
            print "Q_ret_0.shape = {}".format(tf_shape(Q_ret_0))
            print "Q_opc_0.shape = {}".format(tf_shape(Q_opc_0))
            print "Q_ret.shape = {}".format(tf_shape(Q_ret))
            print "Q_opc.shape = {}".format(tf_shape(Q_opc))
            '''

            k = tf.shape(values)[0] # if seq_length is None else seq_length
            i_0 = k - 1

        def cond(i, Q_ret_i, Q_opc_i, Q_ret, Q_opc):
            return i >= 0

        def body(i, Q_ret_i, Q_opc_i, Q_ret, Q_opc):

            # Q^{ret} \leftarrow r_i + \gamma Q^{ret}
            with tf.name_scope("r_i"):
                r_i = r[i:i+1, ...]

            with tf.name_scope("pre_update"):
                Q_ret_i = r_i + gamma * Q_ret_i
                Q_opc_i = r_i + gamma * Q_opc_i

            # TF equivalent of .prepend()
            with tf.name_scope("prepend"):
                Q_ret = tf.concat_v2([Q_ret_i, Q_ret], axis=0)
                Q_opc = tf.concat_v2([Q_opc_i, Q_opc], axis=0)

            '''
            print "Q_ret_i.shape = {}".format(tf_shape(Q_ret_i))
            print "Q_opc_i.shape = {}".format(tf_shape(Q_opc_i))
            print "Q_ret.shape = {}".format(tf_shape(Q_ret))
            print "Q_opc.shape = {}".format(tf_shape(Q_opc))
            '''

            # Q^{ret} \leftarrow c_i (Q^{ret} - Q(x_i, a_i)) + V(x_i)
            with tf.name_scope("post_update"):
                with tf.name_scope("c_i"): c_i = c[i:i+1, ...]
                with tf.name_scope("Q_i"): Q_i = Q_tilt_a[i:i+1, ...]
                with tf.name_scope("V_i"): V_i = values[i:i+1, ...]

                Q_ret_i = c_i * (Q_ret_i - Q_i) + V_i
                Q_opc_i = (Q_opc_i - Q_i) + V_i

            '''
            print "c_i.shape = {}".format(tf_shape(c_i))
            print "Q_i.shape = {}".format(tf_shape(Q_i))
            print "V_i.shape = {}".format(tf_shape(V_i))
            print "Q_ret_i.shape = {}".format(tf_shape(Q_ret_0))
            print "Q_opc_i.shape = {}".format(tf_shape(Q_opc_0))
            '''

            return i-1, Q_ret_i, Q_opc_i, Q_ret, Q_opc

        i, Q_ret_i, Q_opc_i, Q_ret, Q_opc = tf.while_loop(
            cond, body,
            loop_vars=[
                i_0, Q_ret_0, Q_opc_0, Q_ret, Q_opc
            ],
            shape_invariants=[
                i_0.get_shape(),
                Q_ret_0.get_shape(),
                Q_opc_0.get_shape(),
                tf.TensorShape([None, batch_size, 1]),
                tf.TensorShape([None, batch_size, 1])
            ]
        )

        Q_ret = Q_ret[:-1, ...]
        Q_opc = Q_opc[:-1, ...]

        return Q_ret, Q_opc

    def compute_trust_region_update(self, g, pi_avg, pi, delta=0.5):
        """
        In ACER's original paper, they use delta uniformly sampled from [0.1, 2]
        """
        # Compute the KL-divergence between the policy distribution of the
        # average policy network and those of this network, i.e. KL(avg || this)
        KL_divergence = tf.contrib.distributions.kl(
            pi_avg.dist, pi.dist, allow_nan=False)

        # Take the partial derivatives w.r.t. phi (i.e. mu and sigma)
        k = tf_concat(-1, tf.gradients(KL_divergence, pi.phi))

        # z* is the TRPO regularized gradient
        z_star = g - tf.maximum(0., (k * g - delta) / (g ** 2)) * k

        # By using stop_gradient, we make z_star being treated as a constant
        z_star = tf.stop_gradient(z_star)

        return z_star

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

    def fill_lstm_state_placeholder(self, feed_dict, B):
        # If previous LSTM state out is empty, then set it to zeros
        if self.lstm.prev_state_out is None:
            H = self.lstm.num_hidden

            self.lstm.prev_state_out = [
                np.zeros((B, H), np.float32) for s in self.lstm.state_in
            ]

        # Set placeholders with previous LSTM state output
        for sin, prev_sout in zip(self.lstm.state_in, self.lstm.prev_state_out):
            feed_dict[sin] = prev_sout

    def predict(self, tensors, feed_dict, sess=None):
        sess = sess or tf.get_default_session()

        B = feed_dict[self.state.prev_reward].shape[1]
        self.fill_lstm_state_placeholder(feed_dict, B)
        self.avg_net.fill_lstm_state_placeholder(feed_dict, B)

        output, self.lstm.prev_state_out = sess.run([
            tensors, self.lstm.state_out
        ], feed_dict)

        return output

    def predict_actions(self, state, sess=None):

        tensors = [self.a_prime, self.pi.mu, self.pi.sigma]
        feed_dict = self.to_feed_dict(state)

        a_prime, mu, sigma = self.predict(tensors, feed_dict, sess)

        a_prime = a_prime[0, ...]
        mu = mu[0, ...]
        sigma = sigma[0, ...]

        return a_prime, AttrDict(mu=mu, sigma=sigma)

    def get_policy_loss(self, pi, acer_g):

        with tf.name_scope("TRPO"):
            self.z_star = self.compute_trust_region_update(
                acer_g, self.avg_net.pi, self.pi)

        phi = tf_concat(-1, pi.phi)
        losses = -tf.reduce_sum(phi * self.z_star, axis=-1)

        return tf.reduce_mean(losses)

    def get_value_loss(self, Q_ret, Q_tilt_a, rho, value):

        with tf.name_scope("Q_target"):
            Q_target = tf.stop_gradient(Q_ret - Q_tilt_a)

        losses = - Q_target * (Q_tilt_a + tf.minimum(1., rho) * value)
        losses = tf.squeeze(losses, -1)

        return tf.reduce_mean(losses)

    def compute_ACER_gradient(self, rho, pi, a, O_opc, value, rho_prime,
                              Q_tilt_a_prime, a_prime):

        def d_log_prob(actions):
            print "actions.shape = {}".format(tf_shape(actions))
            log_prob = pi.log_prob(actions)
            g = tf.gradients(log_prob, pi.phi)
            return tf_concat(-1, g)

        # compute gradient with importance weight truncation using c = 10
        c = tf.constant(10, tf.float32, name="importance_weight_truncation_thres")

        with tf.name_scope("truncation"):
            with tf.name_scope("truncated_importance_weight"):
                rho_bar = tf.minimum(c, rho)

            with tf.name_scope("d_log_prob_a"):
                g_a = d_log_prob(a)

            with tf.name_scope("target_1"):
                target_1 = self.Q_opc - self.value

            truncation = (rho_bar * target_1) * g_a

        # compute bias correction term
        with tf.name_scope("bias_correction"):
            with tf.name_scope("bracket_plus"):
                plus = tf.nn.relu(1. - c / rho_prime)

            with tf.name_scope("d_log_prob_a_prime"):
                g_ap = d_log_prob(a_prime)

            with tf.name_scope("target_2"):
                target_2 = Q_tilt_a_prime - value

            bias_correction = (plus * target_2) * g_ap

        # g is called "truncation with bias correction" in ACER
        g = truncation + bias_correction

        return g

    def SDN_network(self, advantage, value, pi):
        """
        This function wrap advantage, value, policy pi within closure, so that
        the caller doesn't have to pass these as argument anymore
        """

        def Q_tilt(action, name, num_samples=15):
            with tf.name_scope(name):
                # See eq. 13 in ACER
                if len(action.get_shape()) != 4:
                    action = action[None, ...]

                adv = tf.squeeze(advantage(action, name="A_action"), 0)
                advs = advantage(pi.sample(num_samples), "A_sampled", num_samples)
                mean_adv = tf.reduce_mean(advs, axis=0)
                return value + adv - mean_adv

        return Q_tilt

    def policy_network(self, input, num_outputs):

        # mu: [B, 2], sigma: [B, 2], phi is just a syntatic sugar
        mu, sigma = policy_network(input, num_outputs)

        # Add naive policy as baseline bias
        naive_mu = naive_mean_steer_policy(self.state.front_view)
        mu = tf.pack([mu[..., 0], mu[..., 1] + naive_mu], axis=-1)

        # Convert mu_steer (mu[..., 1]) to mu_yawrate
        mu = s2y(mu, self.get_forward_velocity())

        pi = create_distribution(mu, sigma)

        return pi

    def get_forward_velocity(self):
        return self.state.vehicle_state[..., 4:5]

    def advantage_network(self, input):

        rank = get_rank(input)
        if rank == 3:
            S, B = get_seq_length_batch_size(input)

        # Given states
        def advantage(actions, name, num_samples=1):

            with tf.name_scope(name):
                ndims = len(actions.get_shape())
                broadcaster = tf.zeros([num_samples] + [1] * (ndims-1))
                input_ = input[None, ...] + broadcaster

                input_with_a = tf_concat(-1, [input_, actions])
                input_with_a = flatten_all_leading_axes(input_with_a)

                # 1st fully connected layer
                fc1 = DenseLayer(
                    input=input_with_a,
                    num_outputs=256,
                    nonlinearity="relu",
                    name="fc1")

                # 2nd fully connected layer that regresses the advantage
                fc2 = DenseLayer(
                    input=fc1,
                    num_outputs=1,
                    nonlinearity=None,
                    name="fc2")

                output = fc2

                output = tf.reshape(output, [num_samples, -1, 1])

                if rank == 3:
                    output = tf.reshape(output, [-1, S, B, 1])

            return output

        return tf.make_template('advantage', advantage)

    @staticmethod
    def create_averge_network():
        if "average_net" not in AcerEstimator.__dict__:
            with tf.variable_scope("average_net"):
                AcerEstimator.average_net = AcerEstimator(add_summaries=True, trainable=False)

# AcerEstimator.create_averge_network = create_averge_network

AcerEstimator.Worker = ac.acer.worker.AcerWorker
