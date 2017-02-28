#!/usr/bin/python
import gym
import itertools
import numpy as np
import sys
import tensorflow as tf
import collections
from tqdm import tqdm, trange
from collections import namedtuple
from gym import wrappers

import sklearn.pipeline
import sklearn.preprocessing

if "../" not in sys.path:
    sys.path.append("../") 

from sklearn.kernel_approximation import RBFSampler

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

tf.flags.DEFINE_string("dist", "Gaussian", "Choose distributions. Either gaussian, student-t, beta")
tf.flags.DEFINE_string("logfile", None, "logfile")
tf.flags.DEFINE_string("exp", None, "experiment name (folder to save)")

tf.flags.DEFINE_boolean("record-video", None, "Record video for openai gym upload")
tf.flags.DEFINE_integer("render-every", None, "Render environment every n episodes")
tf.flags.DEFINE_integer("max-episodes", "1001", "Maximum steps per episode")
tf.flags.DEFINE_integer("max-steps", "5000", "Maximum steps per episode")

tf.flags.DEFINE_float("learning-rate", "0.01", "Learning rate for policy estimator")
tf.flags.DEFINE_float("df", "5", "Degree of freedom for StudentT distribution")

tf.flags.DEFINE_integer("random-seed", None, "Random seed for gym.env and TensorFlow")

FLAGS = tf.flags.FLAGS

# Open file to log
assert FLAGS.logfile is not None and FLAGS.exp is not None
f = open(FLAGS.logfile, "w")
print >> f, "episode length reward"

# Create Environments
env = gym.envs.make("MountainCarContinuous-v0")
if FLAGS.random_seed is not None:
    env.seed(FLAGS.random_seed)

# Add monitor (None will use default video recorder, False will disable video recording)
env = wrappers.Monitor(env, 'exp/' + FLAGS.exp, video_callable=None if FLAGS.record_video else False)
env.observation_space.sample()

if FLAGS.record_video:
    FLAGS.render_every = None

if FLAGS.record_video or FLAGS.render_every is not None:
    env.render()

# Get Low/High of action space
LOW  = env.action_space.low[0]
HIGH = env.action_space.high[0]

# Feature Preprocessing: Normalize to zero mean and unit variance
# We use a few samples from the observation space to do this
observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

# Used to converte a state to a featurizes represenation.
# We use RBF kernels with different variances to cover different parts of the space
featurizer = sklearn.pipeline.FeatureUnion([
    ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
    ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
    ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
    ("rbf4", RBFSampler(gamma=0.5, n_components=100))
])
featurizer.fit(scaler.transform(observation_examples))

def featurize_state(state):
    """
    Returns the featurized representation for a state.
    """
    scaled = scaler.transform([state])
    featurized = featurizer.transform(scaled)
    return featurized[0]

class PolicyEstimator():
    """
    Policy Function approximator. 
    """

    def __init__(self, learning_rate=FLAGS.learning_rate, scope="policy_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [400], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            # This is just linear classifier
            self.params = params = [
                tf.squeeze(
                    tf.contrib.layers.fully_connected(
                        inputs=tf.expand_dims(self.state, 0),
                        num_outputs=1,
                        activation_fn=None,
                        weights_initializer=tf.zeros_initializer
                    )
                ) for i in range(3)
            ]

            mu    = params[0]
            sigma = tf.nn.softplus(params[1]) + 1e-5

            alpha = tf.nn.sigmoid(params[0]) * 50 + 2
            beta  = tf.nn.sigmoid(params[1]) * 50 + 2

            # alpha = tf.Print(alpha, [alpha, beta], "\33[93malpha, beta = \33[0m")

            ds = tf.contrib.distributions

            # Create Distribution
            if FLAGS.dist == "Gaussian":
                dist = ds.Normal(mu, sigma)
            elif FLAGS.dist == "StudentT":
                dist = ds.StudentT(mu=mu, sigma=sigma, df=FLAGS.df)
            elif FLAGS.dist == "Beta":
                scale = tf.constant(HIGH - LOW, tf.float32)
                shift = tf.constant(LOW, tf.float32)

                dist = ds.TransformedDistribution(
                    distribution = ds.Beta(alpha, beta),
                    bijector=ds.bijector.ScaleAndShift(shift=shift, scale=scale, event_ndims=0),
                )

                dist.entropy = dist.distribution.entropy

            # Sample action & clip
            action = dist.sample_n(1)
            # action = tf.Print(action, [action], "\33[93maction = \33[0m")
            self.action = tf.clip_by_value(action, LOW, HIGH)

            # Loss and train op
            self.loss = -dist.log_prob(self.action) * self.target
            # Add cross entropy cost to encourage exploration
            self.loss -= 1e-1 * dist.entropy()

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        state = featurize_state(state)
        return sess.run(self.action, { self.state: state })

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        state = featurize_state(state)
        feed_dict = { self.state: state, self.target: target, self.action: action  }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

class ValueEstimator():
    """
    Value Function approximator. 
    """

    def __init__(self, learning_rate=0.1, scope="value_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [400], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            # This is just linear classifier
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)

            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())        

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        state = featurize_state(state)
        return sess.run(self.value_estimate, { self.state: state })

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        state = featurize_state(state)
        feed_dict = { self.state: state, self.target: target }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

def actor_critic(env, estimator_policy, estimator_value, num_episodes, discount_factor=1.0):
    """ Actor Critic Algorithm. Optimizes the policy function approximator using policy gradient.
    Args:
    env: OpenAI environment.
    estimator_policy: Policy Function to be optimized
    estimator_value: Value function approximator, used as a baseline
    num_episodes: Number of episodes to run for
    discount_factor: Time-discount factor

    Returns:
    An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    desc_tmpl = "reward = {:+8.2f} "

    # Train until reach maximum number of episodes or solved
    for i_episode in trange(num_episodes, desc="Training Progress ", position=1):

        # Reset the environment and pick the fisrst action
        state = env.reset()

        episode = []

        # One step in the environment
        pbar = trange(FLAGS.max_steps, leave=False, desc=desc_tmpl.format(0), position=2, unit=" step")
        for t in pbar:

            if FLAGS.render_every is not None and (i_episode + 1) % FLAGS.render_every == 0:
                env.render()

            # Take a step
            action = estimator_policy.predict(state)

            next_state, reward, done, _ = env.step(action)

            # Keep track of the transition
            episode.append(Transition(
                state=state, action=action, reward=reward, next_state=next_state, done=done))

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # Calculate TD Target
            value_next = estimator_value.predict(next_state)
            td_target = reward + discount_factor * value_next
            td_error = td_target - estimator_value.predict(state)

            # Update the value estimator
            estimator_value.update(state, td_target)

            # Update the policy estimator
            # using the td error as our advantage estimate
            estimator_policy.update(state, td_error, action)

            # set progress bar's description
            pbar.set_description(desc_tmpl.format(stats.episode_rewards[i_episode]))

            if done:
                pbar.refresh()
                break

            state = next_state

        if f is not None:
            print >> f, "{} {} {:.4f}".format(i_episode, t, stats.episode_rewards[i_episode])
            f.flush()

        # getting average reward of 90.0 over 100 consecutive trials
        if i_episode >= 100 and np.mean(stats.episode_rewards[i_episode-100:i_episode]) > 90:
            print stats.episode_rewards[:i_episode]
            break

    return stats

tf.reset_default_graph()
if FLAGS.random_seed is not None:
    tf.set_random_seed(FLAGS.random_seed)

global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = PolicyEstimator(learning_rate=0.001)
value_estimator = ValueEstimator(learning_rate=0.1)

x = tf.random_normal((1, ))

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    # Note, due to randomness in the policy the number of episodes you need to learn a good
    # TODO: Sometimes the algorithm gets stuck, I'm not sure what exactly is happening there.

    stats = actor_critic(env, policy_estimator, value_estimator, num_episodes=FLAGS.max_episodes, discount_factor=0.95)
