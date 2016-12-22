import numpy as np
import tensorflow as tf
from tensorflow_helnet.layers import DenseLayer, Conv2DLayer

def get_state_placeholder():
    vehicle_state = tf.placeholder(tf.float32, [None, 6], "vehicle_state")
    prev_action = tf.placeholder(tf.float32, [None, 2], "prev_action")
    prev_reward = tf.placeholder(tf.float32, [None, 1], "prev_reward")

    return {
        "vehicle_state": vehicle_state,
        "prev_action": prev_action,
        "prev_reward": prev_reward
    }

def build_shared_network(rewards, state, add_summaries=False):
    """
    Builds a 3-layer network conv -> conv -> fc as described
    in the A3C paper. This network is shared by both the policy and value net.
    Args:
    state: Input state contains rewards, vehicle_state, prev_reward, prev_action
    add_summaries: If true, add layer summaries to Tensorboard.
    Returns:
    Final layer activations.
    """

    X = rewards

    # Three convolutional layers
    conv1 = tf.contrib.layers.conv2d(
        X, 16, 5, 2, activation_fn=tf.nn.relu, scope="conv1")
    conv2 = tf.contrib.layers.conv2d(
        conv1, 16, 3, 2, activation_fn=tf.nn.relu, scope="conv2")
    conv3 = tf.contrib.layers.conv2d(
        conv2, 16, 3, 2, activation_fn=tf.nn.relu, scope="conv3")
    conv4 = tf.contrib.layers.conv2d(
        conv3, 16, 3, 2, activation_fn=tf.nn.relu, scope="conv4")

    # Fully connected layer
    fc1 = DenseLayer(
        input=tf.contrib.layers.flatten(conv4),
        num_outputs=256,
        name="fc1")

    broadcaster = state["prev_reward"] * 0
    fc1 += broadcaster
    concat1 = tf.concat(1, [fc1, state["prev_reward"]])

    fc2 = DenseLayer(
        input=tf.contrib.layers.flatten(concat1),
        num_outputs=256,
        name="fc2")

    concat2 = tf.concat(1, [fc1, fc2, state["vehicle_state"], state["prev_action"]])

    if add_summaries:
        tf.contrib.layers.summarize_activation(conv1)
        tf.contrib.layers.summarize_activation(conv2)
        tf.contrib.layers.summarize_activation(fc1)
        tf.contrib.layers.summarize_activation(fc2)
        tf.contrib.layers.summarize_activation(concat1)
        tf.contrib.layers.summarize_activation(concat2)

    return concat2

def tf_print(x, message):
    step = tf.contrib.framework.get_global_step()
    cond = tf.equal(tf.mod(step, 100), 0)
    return tf.cond(cond, lambda: tf.Print(x, [x], message=message, summarize=100), lambda: x)

class PolicyEstimator():
    """
    Policy Function approximator. 
    """

    def __init__(self, rewards, reuse=False, learning_rate=0.001, scope="policy_estimator"):

        self.rewards = rewards

        with tf.variable_scope(scope):
            self.target = tf.placeholder(tf.float32, [None, 1], "target")

        # Graph shared with Value Net
        with tf.variable_scope("shared", reuse=reuse):
            self.state = get_state_placeholder()
            shared = build_shared_network(self.rewards, self.state, add_summaries=(not reuse))

        with tf.variable_scope(scope):
            self.mu, self.sigma = self.policy_network(shared, 2)

            FLAGS = tf.flags.FLAGS
            max_a = tf.constant([[FLAGS.max_forward_speed, FLAGS.max_yaw_rate]], dtype=tf.float32)
            min_a = tf.constant([[FLAGS.min_forward_speed, FLAGS.min_yaw_rate]], dtype=tf.float32)
            self.mu = (max_a + min_a) / 2 + tf.nn.tanh(self.mu) * (max_a - min_a) / 2
            # self.mu = tf.nn.sigmoid(self.mu) * (max_a - min_a) + min_a

            self.mu = tf.reshape(self.mu, [-1])
            self.sigma = tf.reshape(self.sigma, [-1])

            self.mu = tf_print(self.mu, "mu = ")
            self.sigma = tf_print(self.sigma, "sigma = ")

            normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
            self.action = normal_dist.sample_n(1)
            self.action = tf.reshape(self.action, [-1, 2])

            # clip action if exceed low/high defined in env.action_space
            self.action = tf.maximum(tf.minimum(self.action, max_a), min_a)

            action = self.action
            action = tf_print(action, "action = ")
            reshaped_action = tf.reshape(action, [-1])
            # reshaped_action = tf_print(reshaped_action, "reshaped_action = ")

            # Loss and train op
            log_prob = tf.reshape(normal_dist.log_prob(reshaped_action), [-1, 2])
            # log_prob = tf_print(log_prob, "log_prob = ")
            self.loss = tf.reduce_sum(-log_prob * self.target)

            # Add cross entropy cost to encourage exploration
            self.loss -= tf.reduce_sum(1e-1 * normal_dist.entropy())
            # self.loss = tf.Print(self.loss, [tf.shape(self.loss)], message="loss.shape = ")

            # self.summary = tf.summary.scalar('policy loss', self.loss)
            # import ipdb; ipdb.set_trace()

            # self.optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.5)
            self.optimizer = tf.train.AdamOptimizer(learning_rate)
            self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
            self.grads_and_vars = [
                (tf.clip_by_norm(grad, 100.), var)
                for grad, var in self.grads_and_vars if grad is not None
            ]
            self.train_op = self.optimizer.apply_gradients(
                self.grads_and_vars, global_step=tf.contrib.framework.get_global_step())

    def policy_network(self, input, num_outputs):

        # This is just linear classifier
        mu = DenseLayer(
            input=input,
            num_outputs=num_outputs,
            name="policy-mu-dense")
        mu = tf.reshape(mu, [-1, 2])

        sigma = DenseLayer(
            input=input,
            num_outputs=num_outputs,
            name="policy-sigma-dense")
        sigma = tf.reshape(sigma, [-1, 2])

        # Add 1e-5 exploration to make sure it's stochastic
        sigma = tf.nn.sigmoid(sigma) * 5 + 0.1

        return mu, sigma

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.state[k]: state[k] for k in state.keys() }
        return sess.run(self.action, feed_dict)

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        # state = featurize_state(state)
        feed_dict = { self.state[k]: state[k] for k in state.keys() }
        feed_dict[self.target] = np.array(target).reshape(-1)
        feed_dict[self.action] = action.reshape(-1, 2)

        # merged = tf.summary.merge_all()
        # merged = tf.merge_summary([self.summary])
        # summary_op = tf.merge_all_summaries()
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

class ValueEstimator():
    """
    Value Function approximator. 
    """

    def __init__(self, rewards, reuse=False, learning_rate=0.001, scope="value_estimator"):
        self.rewards = rewards
        with tf.variable_scope(scope):
            self.target = tf.placeholder(tf.float32, [None, 1], "target")

        # Graph shared with Value Net
        with tf.variable_scope("shared", reuse=reuse):
            self.state = get_state_placeholder()
            shared = build_shared_network(self.rewards, self.state, add_summaries=(not reuse))

        with tf.variable_scope(scope):
            self.value_estimate = self.value_network(shared)
            self.loss = tf.reduce_sum(tf.squared_difference(self.value_estimate, self.target))

            # self.optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.5)
            self.optimizer = tf.train.AdamOptimizer(learning_rate)
            self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
            self.grads_and_vars = [
                (tf.clip_by_norm(grad, 100.), var)
                for grad, var in self.grads_and_vars if grad is not None
            ]
            self.train_op = self.optimizer.apply_gradients(
                self.grads_and_vars, global_step=tf.contrib.framework.get_global_step())

    def value_network(self, input, num_outputs=1):
        # This is just linear classifier
        value = DenseLayer(
            input=input,
            num_outputs=num_outputs,
            name="value-dense")

        value = tf.squeeze(value)

        return value

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.state[k]: state[k] for k in state.keys() }
        return sess.run(self.value_estimate, feed_dict)

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.state[k]: state[k] for k in state.keys() }
        feed_dict[self.target] = np.array(target).reshape(-1)
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss
