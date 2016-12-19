#!/usr/bin/python
import sys
import cv2
import scipy.io
import itertools
import numpy as np
import tensorflow as tf
import collections

from gym_offroad_nav.envs import OffRoadNavEnv
from lib import plotting

class VehicleModel():

    def __init__(self, timestep=0.01):
        model = scipy.io.loadmat("../vehicle_modeling/vehicle_model_ABCD.mat")
        self.A = model["A"]
        self.B = model["B"]
        self.C = model["C"]
        self.D = model["D"]
        self.timestep = timestep

        print "A: {}, B: {}, C: {}, D: {}".format(
            self.A.shape, self.B.shape, self.C.shape, self.D.shape)

        # x is the unobservable hidden state, y is the observation
        # u is (v_forward, yaw_rate), y is (vx, vy, w), where
        # vx is v_slide, vy is v_forward, w is yaw rate
        # x' = Ax + Bu (prediction)
        # y' = Cx + Du (measurement)
        self.x = None

    def _predict(self, x, u):
        y = np.dot(self.C, x) + np.dot(self.D, u)
        x = np.dot(self.A, x) + np.dot(self.B, u)
        return y, x

    def predict(self, state, action):
        # TODO
        # y = state[3:6]
        y, self.x = self._predict(self.x, action)
        
        theta = state[2]
        c, s = np.cos(theta)[0], np.sin(theta)[0]
        M = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

        delta = np.dot(M, state[3:6].reshape(3, 1)) * self.timestep
        # print "(x, y, theta): ({}, {}, {}) => ({}, {}, {})".format(state[0], state[1], state[2], delta[0], delta[1], delta[2])
        state[0:3] += delta
        state[3:6] = y[:]

        return state

    def reset(self, state):
        # state: [x, y, theta, x', y', theta']
        # extract the last 3 elements from state
        y0 = state[3:6].reshape(3, 1)
        self.x = np.dot(np.linalg.pinv(self.C), y0)

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

    X = state["rewards"]

    # Three convolutional layers
    conv1 = tf.contrib.layers.conv2d(
        X, 16, 8, 4, activation_fn=tf.nn.relu, scope="conv1")
    conv2 = tf.contrib.layers.conv2d(
        conv1, 32, 4, 2, activation_fn=tf.nn.relu, scope="conv2")

    # Fully connected layer
    fc1 = tf.contrib.layers.fully_connected(
        inputs=tf.contrib.layers.flatten(conv2),
        num_outputs=256,
        scope="fc1")

    concat1 = tf.concat(1, [fc1, state["prev_reward"]])

    fc2 = tf.contrib.layers.fully_connected(
        inputs=tf.contrib.layers.flatten(concat1),
        num_outputs=256,
        scope="fc2")

    concat2 = tf.concat(1, [fc1, fc2, state["vehicle_state"], state["prev_action"]])

    if add_summaries:
        tf.contrib.layers.summarize_activation(conv1)
        tf.contrib.layers.summarize_activation(conv2)
        tf.contrib.layers.summarize_activation(fc1)
        tf.contrib.layers.summarize_activation(fc2)
        tf.contrib.layers.summarize_activation(concat1)
        tf.contrib.layers.summarize_activation(concat2)

    return concat2

def get_state_placeholder():
    # rewards' shape: (B, H, W, C)
    rewards = tf.placeholder(tf.float32, [None, 40, 40, 1], "rewards")
    vehicle_state = tf.placeholder(tf.float32, [None, 6], "vehicle_state")
    prev_action = tf.placeholder(tf.float32, [None, 2], "prev_action")
    prev_reward = tf.placeholder(tf.float32, [None, 1], "prev_reward")

    return {
        "rewards": rewards,
        "vehicle_state": vehicle_state,
        "prev_action": prev_action,
        "prev_reward": prev_reward
    }

class PolicyEstimator():
    """
    Policy Function approximator. 
    """

    def __init__(self, env, reuse=False, learning_rate=0.01, scope="policy_estimator"):
        with tf.variable_scope(scope):
            self.state = get_state_placeholder()
            self.target = tf.placeholder(tf.float32, [None], "target")

        # Graph shared with Value Net
        with tf.variable_scope("shared", reuse=reuse):
            shared = build_shared_network(self.state, add_summaries=(not reuse))

        with tf.variable_scope(scope):
            self.mu, self.sigma = self.policy_network(shared, 2)

            normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
            self.action = tf.reshape(normal_dist.sample_n(1), (-1, 2))

            # clip action if exceed low/high defined in env.action_space
            max_a = tf.constant(env.max_a, dtype=tf.float32)
            min_a = tf.constant(env.min_a, dtype=tf.float32)
            self.action = tf.maximum(tf.minimum(self.action, max_a), min_a)

            # Loss and train op
            self.loss = -normal_dist.log_prob(self.action) * self.target

            # Add cross entropy cost to encourage exploration
            self.loss -= 1e-1 * normal_dist.entropy()

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def policy_network(self, input, num_outputs):

        # This is just linear classifier
        mu = tf.contrib.layers.fully_connected(
            inputs=tf.expand_dims(input, 0),
            num_outputs=num_outputs,
            activation_fn=None,
            weights_initializer=tf.zeros_initializer)
        mu = tf.reshape(mu, (num_outputs, -1))

        sigma = tf.contrib.layers.fully_connected(
            inputs=tf.expand_dims(input, 0),
            num_outputs=num_outputs,
            activation_fn=None,
            weights_initializer=tf.zeros_initializer)
        sigma = tf.reshape(sigma, (num_outputs, -1))

        # Add 1e-5 exploration to make sure it's stochastic
        sigma = tf.nn.softplus(sigma) + 1e-1

        return mu, sigma

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.state[k]: state[k] for k in self.state.keys() }
        return sess.run(self.action, feed_dict).T

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        # state = featurize_state(state)
        feed_dict = { self.state[k]: state[k] for k in self.state.keys() }
        feed_dict[self.target] = np.array(target).reshape(-1)
        feed_dict[self.action] = action.reshape(-1, 2)

        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

class ValueEstimator():
    """
    Value Function approximator. 
    """

    def __init__(self, reuse=False, learning_rate=0.1, scope="value_estimator"):
        with tf.variable_scope(scope):
            self.state = get_state_placeholder()
            self.target = tf.placeholder(tf.float32, [None], "target")

        # Graph shared with Value Net
        with tf.variable_scope("shared", reuse=reuse):
            shared = build_shared_network(self.state, add_summaries=(not reuse))

        with tf.variable_scope(scope):
            self.value_estimate = self.value_network(shared)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())        

    def value_network(self, input, num_outputs=1):
        # This is just linear classifier
        value = tf.contrib.layers.fully_connected(
            inputs=tf.expand_dims(input, 0),
            num_outputs=num_outputs,
            activation_fn=None,
            weights_initializer=tf.zeros_initializer)

        value = tf.squeeze(value)

        return value

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.state[k]: state[k] for k in self.state.keys() }
        return sess.run(self.value_estimate, feed_dict)

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.state[k]: state[k] for k in self.state.keys() }
        feed_dict[self.target] = np.array(target).reshape(-1)
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

def actor_critic(env, estimator_policy, estimator_value, num_episodes, discount_factor=1.0):
    """
    Actor Critic Algorithm. Optimizes the policy 
    function approximator using policy gradient.

    Args:
    env: OpenAI environment.
    estimator_policy: Policy Function to be optimized 
    estimator_value: Value function approximator, used as a baseline
    num_episodes: Number of episodes to run for
    discount_factor: Time-discount factor

    Returns:
    An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    def form_mdp_state(env, state, prev_action, prev_reward):
        mdp_state = {
            "rewards": env.rewards[None, ..., None],
            "vehicle_state": state.T,
            "prev_action": prev_action.T,
            "prev_reward": np.array(prev_reward, dtype=np.float32).reshape((-1, 1))
        }
        return mdp_state

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    counter = 0

    for i in range(num_episodes):
        # Reset the environment and pick the fisrst action

        # set initial state (x, y, theta, x', y', theta')
        state = np.array([+7, 10, 0, 0, 20, 1.5], dtype=np.float32).reshape(6, 1)
        action = np.array([0, 0], dtype=np.float32).reshape(2, 1)
        reward = 0
        env._reset(state)

        episode = []

        # One step in the environment
        for t in itertools.count():

            env.render()

            # Take a step
            mdp_state = form_mdp_state(env, state, action, reward)
            action = estimator_policy.predict(mdp_state)
            '''
            counter += 1
            if counter < 500:
                action[0] = 10
                action[1] = 1.5
            '''
            next_state, reward, done, _ = env.step(action)

            # Keep track of the transition
            episode.append(Transition(
                state=state, action=action, reward=reward, next_state=next_state, done=done))

            # Update statistics (minus 1 reward per step)
            stats.episode_rewards[i] += reward
            stats.episode_lengths[i] = t

            # Calculate TD Target
            next_mdp_state = form_mdp_state(env, next_state, action, reward)
            value_next = estimator_value.predict(next_mdp_state)
            td_target = reward + discount_factor * value_next
            td_error = td_target - estimator_value.predict(mdp_state)

            # Update the value estimator
            estimator_value.update(mdp_state, td_target)

            # Update the policy estimator
            # using the td error as our advantage estimate
            estimator_policy.update(mdp_state, td_error, action)
            action2 = estimator_policy.predict(mdp_state)
            # print "action = ({}, {}), action2 = ({}, {})".format(action[0], action[1], action2[0], action2[1])

            # Print out which step we're on, useful for debugging.
            print("Step {:3d} @ Episode {} {:10.4f} (\33[93m{:10.4f}\33[0m)".format(
                t, i + 1, reward, stats.episode_rewards[i])),
            print "td_target = {:5.2f} + {:5.2f} * {:5.2f} = {:5.2f}".format(reward, discount_factor, value_next, td_target),

            if t > 400 or stats.episode_rewards[i] < -300:
                break

            if done:
                break

            state = next_state

    return stats

def main():
    vehicle_model = VehicleModel()
    # rewards = scipy.io.loadmat("/share/Research/Yamaha/d-irl/td_lambda_example/data.mat")["reward"]
    rewards = scipy.io.loadmat("circle2.mat")["reward"].astype(np.float32) - 100
    env = OffRoadNavEnv(rewards, vehicle_model)

    tf.reset_default_graph()

    global_step = tf.Variable(0, name="global_step", trainable=False)
    # import ipdb; ipdb.set_trace()
    policy_estimator = PolicyEstimator(env, learning_rate=0.001)
    value_estimator = ValueEstimator(reuse=True, learning_rate=0.1)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        # Note, due to randomness in the policy the number of episodes you need to learn a good
        # TODO: Sometimes the algorithm gets stuck, I'm not sure what exactly is happening there.
        stats = actor_critic(env, policy_estimator, value_estimator, 50000, discount_factor=0.95)

if __name__ == '__main__':
    main()
