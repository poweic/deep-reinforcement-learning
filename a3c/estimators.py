import numpy as np
import tensorflow as tf
from tensorflow_helnet.layers import DenseLayer, Conv2DLayer
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

from math import sqrt

def put_kernels_on_grid (kernel, pad = 1):

    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.
    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)
    Return:
      Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    '''
    # get shape of the grid. NumKernels == grid_Y * grid_X
    def factorization(n):
        for i in range(int(sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1: print('Who would enter a prime number of filters')
                return (i, int(n / i))
    (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
    print 'grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X)

    '''
    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)
    kernel = (kernel - x_min) / (x_max - x_min)
    '''

    # pad X and Y
    x1 = tf.pad(kernel, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel.get_shape()[0] + 2 * pad
    X = kernel.get_shape()[1] + 2 * pad

    channels = kernel.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, channels]))

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, channels]))

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scaling to [0, 255] is not necessary for tensorboard
    return x7

def clip(x, min_v, max_v):
    return tf.maximum(tf.minimum(x, max_v), min_v)

def softclip(x, min_v, max_v):
    return (max_v + min_v) / 2 + tf.nn.tanh(x) * (max_v - min_v) / 2

def tf_print(x, message):
    if not FLAGS.debug:
        return x

    step = tf.contrib.framework.get_global_step()
    cond = tf.equal(tf.mod(step, 1), 0)
    message = "\33[93m" + message + "\33[0m"
    return tf.cond(cond, lambda: tf.Print(x, [x], message=message, summarize=100), lambda: x)

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

        with tf.name_scope("shared"):
            shared = self.build_shared_network(add_summaries=add_summaries)

        with tf.name_scope("policy_network"):
            self.mu, self.sigma = self.policy_network(shared, 2)
            normal_dist = self.get_normal_dist(self.mu, self.sigma)
            self.actions = self.sample_actions(normal_dist)

        with tf.name_scope("value_network"):
            self.logits = self.value_network(shared)

        with tf.name_scope("losses"):
            self.pi_loss = self.get_policy_loss(normal_dist)
            self.entropy = self.get_exploration_loss(normal_dist)
            self.vf_loss = self.get_value_loss()

            self.loss = self.pi_loss + self.vf_loss + FLAGS.entropy_cost_mult * self.entropy

        with tf.name_scope("regularization"):
            self.reg_loss = self.get_reg_loss()
            self.loss += FLAGS.l2_reg * self.reg_loss

        with tf.name_scope("grads_and_optimizer"):
            self.optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            self.grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
            # self.g_mu = tf.gradients(self.pi_loss, [self.mu])[0]

        if add_summaries:
            self.summaries = self.summarize()

    def build_shared_network(self, add_summaries=False):
        """
        Builds a 3-layer network conv -> conv -> fc as described
        in the A3C paper. This network is shared by both the policy and value net.
        Args:
        add_summaries: If true, add layer summaries to Tensorboard.
        Returns:
        Final layer activations.
        """
        state = self.state

        front_view = state["front_view"]
        vehicle_state = state["vehicle_state"]
        prev_action = state["prev_action"]
        prev_reward = state["prev_reward"]

        input = front_view

        with tf.name_scope("conv"):
            # Three convolutional layers
            conv1 = tf.contrib.layers.conv2d(
                input, 32, 7, 2, activation_fn=tf.nn.relu, scope="conv1")
            conv2 = tf.contrib.layers.conv2d(
                conv1, 32, 3, 2, activation_fn=tf.nn.relu, scope="conv2")
            conv3 = tf.contrib.layers.conv2d(
                conv2, 32, 3, 2, activation_fn=tf.nn.relu, scope="conv3")

        with tf.name_scope("dense"):
            # Fully connected layer
            fc1 = DenseLayer(
                input=tf.contrib.layers.flatten(conv3),
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
                num_outputs=512,
                nonlinearity="relu",
                name="fc3")

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

        return fc3

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

        '''
        # Add some initial bias for debugging (to see whether it can recover from it)
        mu_steer    =    mu[:, 1:2] + 15. * np.pi / 180 * np.sign(np.random.rand() - 0.5)
        mu_steer = tf_print(mu_steer, "mu_steer = ")
        mu    = tf.concat(1, [   mu[:, 0:1],    mu_steer])
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

    def policy_network(self, input, num_outputs):

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

    def value_network(self, input, num_outputs=1):
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

        value = tf.reshape(value, [-1, 1])

        return value

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
