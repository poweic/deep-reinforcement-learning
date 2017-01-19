import numpy as np
import tensorflow as tf
from tensorflow_helnet.layers import DenseLayer, Conv2DLayer
FLAGS = tf.flags.FLAGS
batch_size = None

def get_state_placeholder():
    # Note that placeholder are tf.Tensor not tf.Variable
    map_with_vehicle = tf.placeholder(tf.float32, [batch_size, 20, 20, 1], "map_with_vehicle")
    vehicle_state = tf.placeholder(tf.float32, [batch_size, 6], "vehicle_state")
    prev_action = tf.placeholder(tf.float32, [batch_size, 2], "prev_action")
    prev_reward = tf.placeholder(tf.float32, [batch_size, 1], "prev_reward")

    state = {
        "map_with_vehicle": map_with_vehicle,
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
    state: Input state contains rewards, vehicle_state, prev_reward, prev_action
    add_summaries: If true, add layer summaries to Tensorboard.
    Returns:
    Final layer activations.
    """
    X = state["map_with_vehicle"]

    # Three convolutional layers
    conv1 = tf.contrib.layers.conv2d(
        X, 32, 5, 2, activation_fn=tf.nn.relu, scope="conv1")
    conv2 = tf.contrib.layers.conv2d(
        conv1, 32, 3, 2, activation_fn=tf.nn.relu, scope="conv2")
    conv3 = tf.contrib.layers.conv2d(
        conv2, 32, 3, 2, activation_fn=tf.nn.relu, scope="conv3")
    conv4 = tf.contrib.layers.conv2d(
        conv3, 16, 3, 2, activation_fn=tf.nn.relu, scope="conv4")

    # Fully connected layer
    fc1 = DenseLayer(
        input=tf.contrib.layers.flatten(conv4),
        num_outputs=256,
        nonlinearity="relu",
        name="fc1")

    concat1 = tf.concat(1, [fc1, state["prev_reward"], state["vehicle_state"], state["prev_action"]])

    fc2 = DenseLayer(
        input=concat1,
        num_outputs=256,
        nonlinearity="relu",
        name="fc2")

    concat2 = tf.concat(1, [fc1, fc2, state["prev_reward"], state["vehicle_state"], state["prev_action"]])

    fc3 = DenseLayer(
        input=concat2,
        num_outputs=512,
        nonlinearity="relu",
        name="fc3")

    if add_summaries:
        tf.contrib.layers.summarize_activation(conv1)
        tf.contrib.layers.summarize_activation(conv2)
        tf.contrib.layers.summarize_activation(fc1)
        tf.contrib.layers.summarize_activation(fc2)
        tf.contrib.layers.summarize_activation(fc3)
        tf.contrib.layers.summarize_activation(concat1)
        tf.contrib.layers.summarize_activation(concat2)

    return fc3

def tf_print(x, message):
    if not FLAGS.debug:
        return x

    step = tf.contrib.framework.get_global_step()
    cond = tf.equal(tf.mod(step, 10), 0)
    message = "\33[93m" + message + "\33[0m"
    return tf.cond(cond, lambda: tf.Print(x, [x], message=message, summarize=100), lambda: x)

class PolicyValueEstimator():
    """
    Policy Function approximator. 
    """

    def __init__(self, reuse=False):

        self.advantage = tf.placeholder(tf.float32, [batch_size, 1], "advantage")
        self.r = tf.placeholder(tf.float32, [batch_size, 1], "r")

        # Graph shared with Value Net
        self.state = get_state_placeholder()
        with tf.variable_scope("shared", reuse=reuse):
            shared = build_shared_network(self.state, add_summaries=(not reuse))

        '''
        with tf.variable_scope("shared_policy", reuse=reuse):
            shared_policy = build_shared_network(self.state, add_summaries=(not reuse))

        with tf.variable_scope("shared_value", reuse=reuse):
            shared_value = build_shared_network(self.state, add_summaries=(not reuse))
        '''
        shared_policy = shared
        shared_value = shared

        with tf.variable_scope("local"):
            self.mu, self.sigma = self.policy_network(shared_policy, 2)

            '''
            # Make it deterministic for debugging
            self.mu = self.mu * 1e-7 + (max_a + min_a) / 2.
            self.sigma = self.sigma * 1e-7
            '''

            # Reshape, sample, and reshape it back
            self.mu = tf.reshape(self.mu, [-1])
            self.sigma = tf.reshape(self.sigma, [-1])

            # For debugging
            self.mu = tf_print(self.mu, "mu = ")
            self.sigma = tf_print(self.sigma, "sigma = ")

            # Create normal distribution and sample some actions
            normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
            self.actions = self.sample_actions(normal_dist)

            # Compute log probabilities
            self.log_prob = self.compute_log_prob(normal_dist, self.actions)

            # loss of policy (policy gradient)
            self.pi_loss = -tf.reduce_mean(self.log_prob * self.advantage)

            # Add entropy cost to encourage exploration
            self.entropy = tf.reduce_mean(normal_dist.entropy())

            # loss of value function
            self.logits = self.value_network(shared_value)
            self.vf_loss = 0.5 * tf.reduce_mean(tf.square(self.logits - self.r))

            self.loss = self.pi_loss + self.vf_loss + FLAGS.entropy_cost_mult * self.entropy

            self.optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            self.grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]

        self.summarize_policy_estimator()
        self.summarize_value_estimator()

    def sample_actions(self, dist):
        actions = tf.reshape(dist.sample_n(1), [-1, 2])

        # Extract steer from actions (2rd column), turn it to yawrate, and
        # concatenate it back
        vf = actions[:, 0:1]
        steer = actions[:, 1:2]
        yawrate = self.steer_to_yawrate(steer)
        actions = tf.concat(1, [vf, yawrate])

        return actions

    def compute_log_prob(self, dist, actions):
        # Extract yawrate from actions (2rd column), turn it to steer, and
        # concatenate it back
        vf = actions[:, 0:1]
        yawrate = actions[:, 1:2]
        yawrate = tf_print(yawrate, "yawrate = ")
        steer = self.yawrate_to_steer(yawrate)
        steer = tf_print(steer, "steer = ")
        actions = tf.concat(1, [vf, steer])

        reshaped_actions = tf.reshape(actions, [-1])
        log_prob = dist.log_prob(reshaped_actions)
        log_prob = tf_print(log_prob, "log_prob = ")
        log_prob = tf.reshape(log_prob, [-1, 2])

        return log_prob

    def summarize_policy_estimator(self):
        scope = "policy_estimator"
        with tf.variable_scope(scope):
            prefix = tf.get_variable_scope().name
            tf.summary.scalar("{}/loss".format(prefix), self.loss)
            tf.summary.scalar("{}/entropy".format(prefix), self.entropy)
            tf.summary.scalar("{}/sigma_mean".format(prefix), tf.reduce_mean(self.sigma))

        var_scope_name = tf.get_variable_scope().name
        summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summaries = [s for s in summary_ops if var_scope_name in s.name]
        self.summaries = tf.summary.merge(summaries)

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
        radius = v / omega
        steer = tf.atan(FLAGS.wheelbase / radius)
        return steer

    def policy_network(self, input, num_outputs):

        min_mu = tf.constant([[FLAGS.min_mu_vf, FLAGS.min_mu_steer]], dtype=tf.float32)
        max_mu = tf.constant([[FLAGS.max_mu_vf, FLAGS.max_mu_steer]], dtype=tf.float32)

        min_sigma = tf.constant([[FLAGS.min_sigma_vf, FLAGS.min_sigma_steer]], dtype=tf.float32)
        max_sigma = tf.constant([[FLAGS.max_sigma_vf, FLAGS.max_sigma_steer]], dtype=tf.float32)

        def clip(x, min_v, max_v):
            return tf.maximum(tf.minimum(x, max_v), min_v)
        
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

        # Clip mu by min and max, use softplus and capping for sigma
        mu = clip(mu, min_mu, max_mu)
        sigma = tf.minimum(tf.nn.softplus(sigma) + min_sigma, max_mu)

        return mu, sigma

    def summarize_value_estimator(self):
        scope = "value_estimator"
        with tf.variable_scope(scope):
            # Summaries
            prefix = tf.get_variable_scope().name
            tf.summary.scalar("{}/loss".format(prefix), self.loss)

            tf.summary.scalar("{}/max_value".format(prefix), tf.reduce_max(self.logits))
            tf.summary.scalar("{}/min_value".format(prefix), tf.reduce_min(self.logits))
            tf.summary.scalar("{}/mean_value".format(prefix), tf.reduce_mean(self.logits))

            tf.summary.scalar("{}/max_advantage".format(prefix), tf.reduce_max(self.advantage))
            tf.summary.scalar("{}/min_advantage".format(prefix), tf.reduce_min(self.advantage))
            tf.summary.scalar("{}/mean_advantage".format(prefix), tf.reduce_mean(self.advantage))

        var_scope_name = tf.get_variable_scope().name
        summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summaries = [s for s in summary_ops if var_scope_name in s.name and "shared" not in s.name]
        self.summaries = tf.summary.merge(summaries)

    def value_network(self, input, num_outputs=1):
        input = DenseLayer(
            input=input,
            num_outputs=256,
            nonlinearity="relu",
            name="value-input-dense")

        '''
        input = DenseLayer(
            input=input,
            num_outputs=256,
            nonlinearity="relu",
            name="value-input-dense-2")
        '''

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
            tensors += [self.mu, self.mu_steer, self.sigma, self.log_prob]
        return self.predict(state, tensors, sess)

    def predict_actions(self, state, sess=None):
        tensors = [self.actions]
        return self.predict(state, tensors, sess)
