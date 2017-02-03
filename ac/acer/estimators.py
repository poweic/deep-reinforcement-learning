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

def mixture_density(mu_vf, sigma_vf, logits_steer, sigma_steer, vf, steer_buckets, msg):

    # mu: [B, 2], sigma: [B, 2], phi is just a syntatic sugar
    with tf.variable_scope("speed_control_policy"):
        mu_vf = tf_print(mu_vf, msg + ", mu_vf = ")
        sigma_vf = tf_print(sigma_vf, msg + ", sigma_vf = ")
        dist_vf = tf.contrib.distributions.MultivariateNormalDiag(
            mu_vf, sigma_vf, allow_nan_stats=False
        )

    with tf.variable_scope("steer_control_policy"):
        # Create broadcaster to repeat over time, batch axes
        broadcaster = logits_steer[..., 0:1] * 0

        components = [
            tf.contrib.distributions.MultivariateNormalDiag(
                mu = steer_to_yawrate(s + broadcaster, vf),
                diag_stdev = sigma_steer[..., i:i+1],
                allow_nan_stats = False
            ) for i, s in enumerate(steer_buckets)
        ]

        cat = tf.contrib.distributions.Categorical(
            logits=logits_steer, allow_nan_stats=False
        )

        # Gaussian Mixture Model (GMM)
        dist_steer = gmm = tf.contrib.distributions.Mixture(
            cat, components, allow_nan_stats=False
        )

        # Just categorical
        # dist_steer = cat

    pi = AttrDict(
        phi = [mu_vf, sigma_vf, logits_steer, sigma_steer, vf], # for GMM
        mu_vf         = mu_vf,
        sigma_vf      = sigma_vf,
        logits_steer  = logits_steer,
        sigma_steer   = sigma_steer,
        vf            = vf,
        steer_buckets = steer_buckets,
        n_buckets     = len(steer_buckets)
    )

    def prob(x, msg):
        x = x[None, ...]

        x1 = tf_print(x[..., 0:1], msg + ", x1 = ")
        x2 = tf_print(x[..., 1:2], msg + ", x2 = ")

        p1 = dist_vf.prob(x1)[0, ...]
        # p2 = dist_steer.prob(tf.to_int32(x[..., 1]))[0, ...]
        p2 = dist_steer.prob(x2)[0, ...] # for GMM

        p1 = tf_print(p1, msg + ", p1 = ")
        p2 = tf_print(p2, msg + ", p2 = ")
        return p1 * p2

    def log_prob(x):
        x = x[None, ...]
        lp1 = dist_vf.log_prob(x[..., 0:1])[0, ...]
        # lp2 = dist_steer.log_prob(tf.to_int32(x[..., 1]))[0, ...]
        lp2 = dist_steer.log_prob(x[..., 1:2])[0, ...] # for GMM
        lp1 = tf_print(lp1)
        lp2 = tf_print(lp2)
        return lp1 + lp2

    def sample_n(n):
        s1 = dist_vf.sample_n(n)
        s1 = tf.maximum(s1, 0.)

        '''
        step = steer_buckets[1] - steer_buckets[0]
        begin = steer_buckets[0]
        s2 = tf.to_float(dist_steer.sample_n(n)[..., None]) * step + begin
        '''
        s2 = dist_steer.sample_n(n) # for GMM

        s = tf_concat(-1, [s1, s2])
        return s

    pi.prob     = prob
    pi.log_prob = log_prob
    pi.sample_n = sample_n

    return pi

class AcerEstimator():
    def __init__(self, add_summaries=False, trainable=True):

        self.trainable = trainable

        self.avg_net = getattr(AcerEstimator, "average_net", self)

        with tf.name_scope("inputs"):
            self.state = get_state_placeholder()
            # self.a = tf.to_float(tf.placeholder(tf.int32, [seq_length, batch_size, 2], "actions"))
            self.a = tf.placeholder(tf.float32, [seq_length, batch_size, 2], "actions")
            self.r = tf.placeholder(tf.float32, [seq_length, batch_size, 1], "rewards")

        with tf.variable_scope("shared"):
            shared, self.lstm = build_shared_network(self.state, add_summaries)

        with tf.variable_scope("pi"):
            # self.pi = self.gaussian_policy(shared, 2)
            self.pi = self.mixture_density_policy(shared, 2)

        with tf.variable_scope("mu"):
            # Placeholder for behavior policy
            self.mu = mixture_density(
                tf.placeholder(tf.float32, [seq_length, batch_size, 1], "mu_vf"),
                tf.placeholder(tf.float32, [seq_length, batch_size, 1], "sigma_vf"),
                tf.placeholder(tf.float32, [seq_length, batch_size, self.pi.n_buckets], "logits_steer"),
                tf.placeholder(tf.float32, [seq_length, batch_size, self.pi.n_buckets], "sigma_steer"),
                tf.placeholder(tf.float32, [seq_length, batch_size, 1], "vf"),
                self.pi.steer_buckets,
                msg="self.mu"
            )

            '''
            self.mu = create_distribution(
                mu    = tf.placeholder(tf.float32, [seq_length, batch_size, 2], "mu"),
                sigma = tf.placeholder(tf.float32, [seq_length, batch_size, 2], "sigma")
            )
            '''

        with tf.name_scope("output"):
            self.a_prime = tf.squeeze(self.pi.sample_n(1), 0)

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

        with tf.name_scope("losses"):
            self.pi_loss, self.pi_loss_sur = self.get_policy_loss(
                self.rho, self.pi, self.a, self.Q_opc, self.value,
                self.rho_prime, self.Q_tilt_a_prime, self.a_prime
            )

            self.vf_loss, self.vf_loss_sur = self.get_value_loss(
                self.Q_ret, self.Q_tilt_a, self.rho, self.value
            )

            # Surrogate loss is the loss tensor we passed to optimizer for
            # automatic gradient computation, it uses lots of stop_gradient.
            # Therefore it's different from the true loss (self.loss)
            self.loss_sur = self.pi_loss_sur + self.vf_loss_sur
            self.loss = self.pi_loss + self.vf_loss

        with tf.name_scope("grads_and_optimizer"):
            self.optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate)
            grads_and_vars = self.optimizer.compute_gradients(self.loss_sur)
            self.grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]

            # Collect all trainable variables initialized here
            self.var_list = [v for g, v in self.grads_and_vars]

        self.summaries = self.summarize(add_summaries)

    def compute_rho(self, a, a_prime, pi, mu):
        # rho = tf.zeros_like(self.value)
        # rho_prime = tf.zeros_like(self.value)
        # return rho, rho_prime

        # compute rho, rho_prime, and c
        with tf.name_scope("pi_a"):
            pi_a = pi.prob(a, "In pi_a: ")[..., None]
            pi_a = tf_print(pi_a)
        with tf.name_scope("mu_a"):
            mu_a = mu.prob(a, "In mu_a: ")[..., None]
            mu_a = tf_print(mu_a)

        with tf.name_scope("pi_a_prime"):
            pi_a_prime = pi.prob(a_prime, "In pi_a_prime: ")[..., None]
            pi_a_prime = tf_print(pi_a_prime)
        with tf.name_scope("mu_a_prime"):
            mu_a_prime = mu.prob(a_prime, "In mu_a_prime: ")[..., None]
            mu_a_prime = tf_print(mu_a_prime)

        # use tf.div instead of pi_a / mu_a to assign a name to the output
        rho = tf.div(pi_a, mu_a)
        rho_prime = tf.div(pi_a_prime, mu_a_prime)

        print "pi_a.shape = {}".format(tf_shape(pi_a))
        print "mu_a.shape = {}".format(tf_shape(mu_a))
        print "pi_a_prime.shape = {}".format(tf_shape(pi_a_prime))
        print "mu_a_prime.shape = {}".format(tf_shape(mu_a_prime))
        print "rho.shape = {}".format(tf_shape(rho))
        print "rho_prime.shape = {}".format(tf_shape(rho_prime))

        rho = tf.stop_gradient(rho, name="rho")
        rho_prime = tf.stop_gradient(rho_prime, name="rho_prime")

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

            # Q^{ret} \leftarrow c_i (Q^{ret} - Q(x_i, a_i)) + V(x_i)
            with tf.name_scope("post_update"):
                with tf.name_scope("c_i"): c_i = c[i:i+1, ...]
                with tf.name_scope("Q_i"): Q_i = Q_tilt_a[i:i+1, ...]
                with tf.name_scope("V_i"): V_i = values[i:i+1, ...]

                Q_ret_i = c_i * (Q_ret_i - Q_i) + V_i
                Q_opc_i = 1.0 * (Q_opc_i - Q_i) + V_i

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

        Q_ret = tf.stop_gradient(Q_ret[:-1, ...], name="Q_ret")
        Q_opc = tf.stop_gradient(Q_opc[:-1, ...], name="Q_opc")

        return Q_ret, Q_opc

    def compute_trust_region_update(self, g, pi_avg, pi, delta=0.5):
        """
        In ACER's original paper, they use delta uniformly sampled from [0.1, 2]
        """
        try:
            # Compute the KL-divergence between the policy distribution of the
            # average policy network and those of this network, i.e. KL(avg || this)
            KL_divergence = tf.contrib.distributions.kl(
                pi_avg.dist, pi.dist, allow_nan=False)

            # Take the partial derivatives w.r.t. phi (i.e. mu and sigma)
            k = tf_concat(-1, tf.gradients(KL_divergence, pi.phi))

            # Compute \frac{k^T g - \delta}{k^T k}, perform reduction only on axis 2
            num   = tf.reduce_sum(k * g, axis=2, keep_dims=True) - delta
            denom = tf.reduce_sum(k * k, axis=2, keep_dims=True)

            # z* is the TRPO regularized gradient
            z_star = g - tf.maximum(0., num / denom) * k
        except:
            z_star = g

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

            self.lstm.prev_state_out = [
                np.zeros((B, s.get_shape().as_list()[1]), np.float32)
                for s in self.lstm.state_in
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

        # Sample in TF
        tensors = [self.a_prime, {
            k: v for k, v in self.pi.iteritems() if isinstance(v, tf.Tensor)
        }]

        feed_dict = self.to_feed_dict(state)

        # a_prime, mu, sigma = self.predict(tensors, feed_dict, sess)
        a_prime, pi_vars = self.predict(tensors, feed_dict, sess)

        a_prime = a_prime[0, ...].T
        pi_vars = {k: v[0, ...].T for k, v in pi_vars.iteritems()}

        return a_prime, pi_vars
        
        """
        # Sample in NumPy
        tensors = {
            k: v for k, v in self.pi.iteritems() if isinstance(v, tf.Tensor)
        }

        feed_dict = self.to_feed_dict(state)

        # a_prime, mu, sigma = self.predict(tensors, feed_dict, sess)
        pi_vars = self.predict(tensors, feed_dict, sess)

        pi_vars = AttrDict({k: v[0, ...].T for k, v in pi_vars.iteritems()})

        return pi_vars
        """

    def get_policy_loss(self, rho, pi, a, Q_opc, value, rho_prime,
                        Q_tilt_a_prime, a_prime):

        with tf.name_scope("ACER"):
            pi_obj = self.compute_ACER_policy_obj(
                rho, pi, a, Q_opc, value, rho_prime, Q_tilt_a_prime, a_prime)

        pi_loss, pi_loss_sur = self.add_TRPO_regularization(pi, pi_obj)

        return pi_loss, pi_loss_sur

    def add_TRPO_regularization(self, pi, pi_obj):

        # ACER gradient is the gradient of policy objective function, which is
        # the negatation of policy loss
        g_acer = tf_concat(-1, tf.gradients(pi_obj, pi.phi))

        with tf.name_scope("TRPO"):
            g_acer_trpo = self.compute_trust_region_update(
                g_acer, self.avg_net.pi, pi)

        phi = tf_concat(-1, pi.phi)

        # Compute the objective function (obj) we try to maximize
        obj = pi_obj
        obj_sur = phi * g_acer_trpo

        # Sum over the intermediate variable phi
        obj     = tf.reduce_sum(obj    , axis=2)
        obj_sur = tf.reduce_sum(obj_sur, axis=2)

        # Take mean over batch axis
        obj     = tf.reduce_mean(obj    , axis=1)
        obj_sur = tf.reduce_mean(obj_sur, axis=1)

        # Sum over time axis
        obj     = tf.reduce_mean(obj    , axis=0)
        obj_sur = tf.reduce_mean(obj_sur, axis=0)

        # loss is the negative of objective function
        return -obj, -obj_sur

    def get_value_loss(self, Q_ret, Q_tilt_a, rho, value):

        # ======================= DEBUG =======================
        cond = tf.reduce_mean(0.5 * tf.square(Q_ret - Q_tilt_a)) > 5000.
        Q_ret = tf_print(Q_ret, "Q_ret = ", cond)
        Q_tilt_a = tf_print(Q_tilt_a, "Q_tilt_a = ", cond)
        # ======================= DEBUG =======================

        Q_diff = Q_ret - Q_tilt_a
        V_diff = tf.minimum(1., rho) * Q_diff

        # ======================= DEBUG =======================
        Q_diff = tf_print(Q_diff, "Q_diff = ", cond)
        # ======================= DEBUG =======================

        # L2 norm as loss function
        Q_l2_loss = 0.5 * tf.square(Q_diff)
        V_l2_loss = 0.5 * tf.square(V_diff)

        # This is the surrogate loss function for V_l2
        V_l2_loss_sur = tf.stop_gradient(V_diff) * value

        # Compute the objective function (obj) we try to maximize
        loss     = Q_l2_loss + V_l2_loss
        loss_sur = Q_l2_loss + V_l2_loss_sur

        # Take mean over batch axis
        loss     = tf.reduce_mean(tf.squeeze(loss,     -1), axis=1)
        loss_sur = tf.reduce_mean(tf.squeeze(loss_sur, -1), axis=1)

        # Sum over time axis
        loss     = tf.reduce_mean(loss,     axis=0)
        loss_sur = tf.reduce_mean(loss_sur, axis=0)

        return loss, loss_sur

    def compute_ACER_policy_obj(self, rho, pi, a, Q_opc, value, rho_prime,
                                 Q_tilt_a_prime, a_prime):

        # compute gradient with importance weight truncation using c = 10
        c = tf.constant(10, tf.float32, name="importance_weight_truncation_thres")

        with tf.name_scope("truncation"):
            with tf.name_scope("truncated_importance_weight"):
                rho_bar = tf.minimum(c, rho)

            with tf.name_scope("d_log_prob_a"):
                log_a = pi.log_prob(a)[..., None]
                log_a = tf_print(log_a, "log_a = ")

            with tf.name_scope("target_1"):
                target_1 = self.Q_opc - self.value

            # Policy gradient should only flow backs from log \pi
            truncation = tf.stop_gradient(rho_bar * target_1) * log_a

        # compute bias correction term
        with tf.name_scope("bias_correction"):
            with tf.name_scope("bracket_plus"):
                plus = tf.nn.relu(1. - c / rho_prime)

            with tf.name_scope("d_log_prob_a_prime"):
                log_ap = pi.log_prob(a_prime)[..., None]
                log_ap = tf_print(log_ap, "log_ap = ")

            with tf.name_scope("target_2"):
                target_2 = Q_tilt_a_prime - value

            # Policy gradient should only flow backs from log \pi
            bias_correction = tf.stop_gradient(plus * target_2) * log_ap

        # g is called "truncation with bias correction" in ACER
        obj = truncation + bias_correction

        return obj

    def SDN_network(self, advantage, value, pi):
        """
        This function wrap advantage, value, policy pi within closure, so that
        the caller doesn't have to pass these as argument anymore
        """

        def Q_tilt(action, name, num_samples=32):
            with tf.name_scope(name):
                # See eq. 13 in ACER
                if len(action.get_shape()) != 4:
                    action = action[None, ...]

                adv = tf.squeeze(advantage(action, name="A_action"), 0)

                # TODO Use symmetric low variance sampling !!
                advs = advantage(pi.sample_n(num_samples), "A_sampled", num_samples)
                mean_adv = tf.reduce_mean(advs, axis=0)

                return value + adv - mean_adv

        return Q_tilt

    def mixture_density_policy(self, input, num_outputs):

        def get_steer_buckets(w):
            n = int(30 / w)
            return np.linspace(-n*w, n*w, 2*n+1).astype(np.float32)

        # mu: [B, 2], sigma: [B, 2], phi is just a syntatic sugar
        with tf.variable_scope("speed_control_policy"):
            mu_vf, sigma_vf = policy_network(input, 1)

        with tf.variable_scope("steer_control_policy"):
            bucket_width = 30
            steer_buckets = get_steer_buckets(bucket_width)
            n_buckets = len(steer_buckets)

            print "steer_buckets      = {} degree".format(steer_buckets)
            steer_buckets = to_radian(steer_buckets)

            min_sigma = to_radian(0.1)
            max_sigma = to_radian(bucket_width / 2)

            logits_steer, sigma_steer = policy_network(input, n_buckets)
            sigma_steer = softclip(sigma_steer, min_sigma, max_sigma)

        print "mu_vf.shape        = {}".format(tf_shape(mu_vf))
        print "sigma_vf.shape     = {}".format(tf_shape(sigma_vf))
        print "logits_steer.shape = {}".format(tf_shape(logits_steer))
        print "sigma_steer.shape  = {}".format(tf_shape(sigma_steer))

        pi = mixture_density(mu_vf, sigma_vf, logits_steer, sigma_steer,
                             self.get_forward_velocity(), steer_buckets,
                             msg="pi")

        return pi

    def gaussian_policy(self, input, num_outputs):

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

    def summarize(self, add_summaries):

        if not add_summaries:
            return tf.no_op()

        with tf.name_scope("summaries"):
            tf.summary.scalar("vf_loss", self.vf_loss)
            tf.summary.scalar("pi_loss", self.pi_loss)
            tf.summary.scalar("loss", self.loss)

        return tf.summary.merge_all()

    @staticmethod
    def create_averge_network():
        if "average_net" not in AcerEstimator.__dict__:
            with tf.variable_scope("average_net"):
                AcerEstimator.average_net = AcerEstimator(add_summaries=False, trainable=False)

# AcerEstimator.create_averge_network = create_averge_network

AcerEstimator.Worker = ac.acer.worker.AcerWorker
