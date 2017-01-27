import numpy as np
import tensorflow as tf
from tensorflow_helnet.layers import DenseLayer, Conv2DLayer, MaxPool2DLayer
from a3c.utils import *
FLAGS = tf.flags.FLAGS
batch_size = None

def get_state_placeholder():
    # Note that placeholder are tf.Tensor not tf.Variable
    front_view = tf.placeholder(tf.float32, [batch_size, 20, 20, 1], "front_view")
    vehicle_state = tf.placeholder(tf.float32, [batch_size, 6], "vehicle_state")
    prev_action = tf.placeholder(tf.float32, [batch_size, 2], "prev_action")
    prev_reward = tf.placeholder(tf.float32, [batch_size, 1], "prev_reward")

    state = {
        "front_view": front_view,
        "vehicle_state": vehicle_state,
        "prev_action": prev_action,
        "prev_reward": prev_reward
    }

    return state

def build_shared_network(state, add_summaries=False):
    """
    Builds a 3-layer network conv -> conv -> fc as described
    in the A3C paper. This network is shared by both the policy and value net.
    Args:
    add_summaries: If true, add layer summaries to Tensorboard.
    Returns:
    Final layer activations.
    """

    front_view = state["front_view"]
    vehicle_state = state["vehicle_state"]
    prev_action = state["prev_action"]
    prev_reward = state["prev_reward"]

    input = front_view

    with tf.name_scope("conv"):
        conv1 = Conv2DLayer(input, 32, 3, dilation=1, pad=1, nonlinearity="relu", name="conv1")
        conv2 = Conv2DLayer(conv1, 32, 3, dilation=2, pad=2, nonlinearity="relu", name="conv2")
        conv3 = Conv2DLayer(conv2, 32, 3, dilation=4, pad=4, nonlinearity="relu", name="conv3")
        conv4 = Conv2DLayer(conv3, 32, 3, dilation=8, pad=8, nonlinearity="relu", name="conv4")
        pool1 = MaxPool2DLayer(conv4, pool_size=3, stride=2, name='pool1')
        pool2 = MaxPool2DLayer(pool1, pool_size=3, stride=2, name='pool2')
        '''
        # Three convolutional layers
        conv1 = tf.contrib.layers.conv2d(
            input, 128, 3, 2, activation_fn=tf.nn.relu, scope="conv1")
        conv2 = tf.contrib.layers.conv2d(
            conv1, 128, 3, 2, activation_fn=tf.nn.relu, scope="conv2")
        conv3 = tf.contrib.layers.conv2d(
            conv2, 128, 3, 2, activation_fn=tf.nn.relu, scope="conv3")
        '''

    with tf.name_scope("dense"):
        # Fully connected layer
        fc1 = DenseLayer(
            input=tf.contrib.layers.flatten(pool2),
            num_outputs=256,
            nonlinearity="relu",
            name="fc1")

        concat1 = tf.concat(1, [fc1, prev_reward, vehicle_state, prev_action])

        fc2 = DenseLayer(
            input=concat1,
            num_outputs=256,
            nonlinearity="relu",
            name="fc2")

        concat2 = tf.concat(1, [fc1, fc2, prev_reward, vehicle_state, prev_action])

        fc3 = DenseLayer(
            input=concat2,
            num_outputs=256,
            nonlinearity="relu",
            name="fc3")

        concat3 = fc3
        # concat3 = tf.concat(1, [fc1, fc2, fc3, prev_reward, vehicle_state, prev_action])

    if add_summaries:
        with tf.name_scope("summaries"):
            conv1_w = [v for v in tf.trainable_variables() if "conv1/weights"][0]
            grid = put_kernels_on_grid(conv1_w)
            tf.summary.image("conv1/weights", grid)

            tf.summary.image("front_view", front_view, max_outputs=100)

            tf.contrib.layers.summarize_activation(conv1)
            tf.contrib.layers.summarize_activation(conv2)
            tf.contrib.layers.summarize_activation(fc1)
            tf.contrib.layers.summarize_activation(fc2)
            tf.contrib.layers.summarize_activation(fc3)
            tf.contrib.layers.summarize_activation(concat1)
            tf.contrib.layers.summarize_activation(concat2)

    return concat3

def policy_network(input, num_outputs):

    input = DenseLayer(
        input=input,
        num_outputs=256,
        nonlinearity="relu",
        name="policy-input-dense")

    # Linear classifiers for mu and sigma
    mu = DenseLayer(
        input=input,
        num_outputs=num_outputs,
        name="policy-mu")

    sigma = DenseLayer(
        input=input,
        num_outputs=num_outputs,
        name="policy-sigma")

    with tf.name_scope("mu_sigma_constraints"):
        min_mu = tf.constant([[FLAGS.min_mu_vf, FLAGS.min_mu_steer]], dtype=tf.float32)
        max_mu = tf.constant([[FLAGS.max_mu_vf, FLAGS.max_mu_steer]], dtype=tf.float32)

        min_sigma = tf.constant([[FLAGS.min_sigma_vf, FLAGS.min_sigma_steer]], dtype=tf.float32)
        max_sigma = tf.constant([[FLAGS.max_sigma_vf, FLAGS.max_sigma_steer]], dtype=tf.float32)

        # Clip mu by min and max, use softplus and capping for sigma
        # mu = clip(mu, min_mu, max_mu)
        # sigma = tf.minimum(tf.nn.softplus(sigma) + min_sigma, max_sigma)
        # sigma = tf.nn.sigmoid(sigma) * 1e-20 + max_sigma

        mu = softclip(mu, min_mu, max_mu)
        sigma = softclip(sigma, min_sigma, max_sigma)

    return mu, sigma

def state_value_network(input, num_outputs=1):
    """
    This is state-only value V
    """
    input = DenseLayer(
        input=input,
        num_outputs=256,
        nonlinearity="relu",
        name="value-input-dense")

    input = DenseLayer(
        input=input,
        num_outputs=256,
        nonlinearity="relu",
        name="value-input-dense-2")

    # This is just linear classifier
    value = DenseLayer(
        input=input,
        num_outputs=num_outputs,
        name="value-dense")

    value = tf.reshape(value, [-1, 1], name="value")

    return value

class AcerEstimator():
    def __init__(self, avg_net=None, add_summaries=False):

        self.avg_net = avg_net
        self.trainable = (avg_net is not None)

        with tf.name_scope("inputs"):
            self.state = get_state_placeholder()

            self.actions         = tf.placeholder(tf.float32, [batch_size, 2], "actions")
            self.sampled_actions = tf.placeholder(tf.float32, [batch_size, 2], "sampled_actions")
            self.Q_ret           = tf.placeholder(tf.float32, [batch_size, 1], "Q_ret")
            self.Q_opc           = tf.placeholder(tf.float32, [batch_size, 1], "Q_opc")
            self.rho             = tf.placeholder(tf.float32, [batch_size, 1], "rho")
            self.rho_prime       = tf.placeholder(tf.float32, [batch_size, 1], "rho_prime")

        with tf.variable_scope("shared"):
            shared = build_shared_network(self.state, add_summaries=add_summaries)

        with tf.variable_scope("pi"):
            self.pi = self.policy_network(shared, 2)

        with tf.variable_scope("V"):
            self.value = state_value_network(shared)

        with tf.variable_scope("A"):
            self.adv = self.advantage_network(shared)
            self.Q_tilt = self.SDN_network(self.adv, self.value, self.pi)

        if not self.trainable:
            return

        with tf.name_scope("ACER"):
            self.acer_g = self.compute_ACER_gradient(
                self.rho, self.pi, self.actions, self.Q_opc, self.value,
                self.rho_prime, self.Q_tilt, self.sampled_actions)

        with tf.name_scope("losses"):
            self.pi_loss = self.get_policy_loss(self.pi, self.acer_g)
            self.vf_loss = self.get_value_loss(
                self.Q_ret, self.Q_tilt, self.actions, self.rho, self.value)

            self.loss = tf.reduce_mean(self.pi_loss + self.vf_loss)

    def compute_trust_region_update(self, g, pi_avg, pi, delta=0.5):
        """
        In ACER's original paper, they use delta uniformly sampled from [0.1, 2]
        """
        # Compute the KL-divergence between the policy distribution of the
        # average policy network and those of this network, i.e. KL(avg || this)
        KL_divergence = tf.contrib.distributions.kl(
            pi_avg.dist, pi.dist, allow_nan=False)

        # Take the partial derivatives w.r.t. phi (i.e. mu and sigma)
        k = tf.concat(1, tf.gradients(KL_divergence, pi.phi))

        # z* is the TRPO regularized gradient
        z_star = g - tf.maximum(0., (k * g - delta) / (g ** 2)) * k

        # By using stop_gradient, we make z_star being treated as a constant
        z_star = tf.stop_gradient(z_star)

        return z_star

    def get_policy_loss(self, pi, acer_g):

        with tf.name_scope("TRPO"):
            self.z_star = self.compute_trust_region_update(
                acer_g, self.avg_net.pi, self.pi)

        phi = tf.concat(1, pi.phi)
        losses = -tf.reduce_sum(phi * self.z_star, axis=1)

        return losses

    def get_value_loss(self, Q_ret, Q_tilt, actions, rho, value):

        with tf.name_scope("Q_target"):
            Q_tilt_a = Q_tilt(actions, name="Q_tilt_a")
            Q_target = tf.stop_gradient(Q_ret - Q_tilt_a)

        losses = - Q_target * (Q_tilt_a + tf.minimum(1., rho) * value)

        return losses

    def compute_ACER_gradient(self, rho, pi, actions, O_opc, value, rho_prime,
                              Q_tilt, sampled_actions):

        def d_log_prob(actions):
            return tf.concat(1, tf.gradients(pi.dist.log_prob(actions), pi.phi))

        # compute gradient with importance weight truncation using c = 10
        c = tf.constant(10, tf.float32, name="importance_weight_truncation_thres")

        with tf.name_scope("truncation"):
            with tf.name_scope("truncated_importance_weight"):
                rho_bar = tf.minimum(c, rho)

            with tf.name_scope("d_log_prob_a"):
                g_a = d_log_prob(self.actions)

            with tf.name_scope("target_1"):
                target_1 = self.Q_opc - self.value

            truncation = (rho_bar * target_1) * g_a

        # compute bias correction term
        with tf.name_scope("bias_correction"):
            with tf.name_scope("bracket_plus"):
                plus = tf.nn.relu(1. - c / rho_prime)

            with tf.name_scope("d_log_prob_a_prime"):
                g_ap = d_log_prob(sampled_actions)

            with tf.name_scope("target_2"):
                target_2 = Q_tilt(sampled_actions, name="Q_tilt_a_prime") - value

            bias_correction = (plus * target_2) * g_ap

        # g is called "truncation with bias correction" in ACER
        g = truncation + bias_correction

        return g

    def SDN_network(self, advantage, value, pi):
        """
        This function wrap advantage, value, policy pi within closure, so that
        the caller doesn't have to pass these as argument anymore
        """

        def Q_tilt(action, name, num_samples=5):
            with tf.name_scope(name):
                # See eq. 13 in ACER
                adv = advantage(action, name="A_action")
                advs = advantage(pi.dist.sample_n(num_samples), "A_sampled", num_samples)
                mean_adv = tf.reduce_mean(advs, axis=0)
                return value + adv - mean_adv

        return Q_tilt

    def policy_network(self, input, num_outputs):

        pi = AttrDict()

        # mu: [B, 2], sigma: [B, 2], phi is just a syntatic sugar
        pi.mu, pi.sigma = policy_network(input, num_outputs)
        pi.phi = [pi.mu, pi.sigma]

        # Reshape & create normal distribution and sample some actions
        pi.dist = tf.contrib.distributions.MultivariateNormalDiag(pi.mu, pi.sigma)

        return pi

    def advantage_network(self, input):

        # Given states
        def advantage(actions, name, num_samples=1):

            with tf.name_scope(name):
                ndims = len(actions.get_shape())
                broadcaster = tf.zeros([num_samples] + [1] * (ndims-1))
                input_ = input + broadcaster

                input_with_a = flatten(tf.concat(ndims - 1, [input_, actions]))

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
                if ndims == 3:
                    output = tf.reshape(output, [num_samples, -1, 1])

            return output

        return tf.make_template('advantage', advantage)

class PolicyValueEstimator():
    """
    Policy Function approximator. 
    """

    def __init__(self, add_summaries=False):

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
            yawrate = self.steer_to_yawrate(steer)
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
            steer = self.yawrate_to_steer(yawrate)
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

    def steer_to_yawrate(self, steer):
        # Use Ackerman formula to compute yawrate from steering angle and
        # forward velocity (4-th element of vehicle_state)
        v = self.state["vehicle_state"][:, 4:5]
        radius = FLAGS.wheelbase / tf.tan(steer)
        omega = v / radius
        return omega

    def yawrate_to_steer(self, omega):
        # Use Ackerman formula to compute steering angle from yawrate and
        # forward velocity (4-th element of vehicle_state)
        v = self.state["vehicle_state"][:, 4:5]
        # v = tf_print(v, "v[:, 4:5] = ")
        # omega = tf_print(omega, "omega = ")
        radius = v / omega
        # radius = tf_print(radius, "radius = ")
        steer = tf.atan(FLAGS.wheelbase / radius)
        # steer = tf_print(steer, "steer = ")
        return steer

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
