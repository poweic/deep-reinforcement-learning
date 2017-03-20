import traceback
import itertools
import numpy as np
import tensorflow as tf
from ac.utils import *
from collections import deque
FLAGS = tf.flags.FLAGS

class Worker(object):
    """
    An A3C worker thread. Runs episodes locally and updates global shared value and policy nets.

    Args:
    name: A unique name for this worker
    env: The Gym environment used by this worker
    global_net: Instance of the globally shared network
    global_counter: Iterator that holds the global step
    summary_writer: A tf.train.SummaryWriter for Tensorboard summaries
    """
    def __init__(self, name, make_env, global_counter, global_episode_stats, global_net, add_summaries,
                 n_agents=1):

        self.name = name
        self.env = make_env(self)
        self.global_counter = global_counter
        self.global_episode_stats = global_episode_stats
        self.global_net = global_net
        self.add_summaries = add_summaries
        self.n_agents = n_agents

        # Get global variables and flags
        self.global_step = tf.contrib.framework.get_global_step()
        self.discount_factor = FLAGS.discount_factor
        self.max_global_steps = FLAGS.max_global_steps

        # Create local policy/value nets that are not updated asynchronously
        with tf.variable_scope(name):
            self.local_net = self.Estimator(add_summaries)
        self.set_global_net(global_net)

        # Initialize counter, maximum return of this worker, and summary_writer
        self.counter = 0
        self.max_return = 0
        self.summary_writer = None

        self.local_counter = itertools.count()

        self.reset_env()

    def copy_params_from_global(self):
        # Copy Parameters from the global networks
        self.sess.run(self.copy_params_op)

    def reset_env(self):

        self.state = self.env.reset()

        # Reshape to compatiable format
        # self.state = self.state.astype(np.float32).reshape(6, -1)
        self.action = np.zeros((FLAGS.num_actions, self.n_agents), dtype=np.float32)

        self.total_return = np.zeros((1, self.n_agents), dtype=np.float32)
        self.current_reward = np.zeros((1, self.n_agents), dtype=np.float32)

        # Set initial timestamp
        self.global_episode_stats.set_initial_timestamp()

    def run_n_steps(self, n_steps):

        transitions = []

        # Initial state
        seed = self.env.seed()
        self.reset_env()
        self.local_net.reset_lstm_state()
        prev_done = False

        reward = np.zeros((1, self.n_agents), dtype=np.float32)
        for i in range(n_steps):

            state = form_state(self.env, self.state, self.action, reward)

            # Predict an action
            self.action, pi_stats = self.local_net.predict_actions(state, self.sess)

            # Take a step in environment
            next_state, reward, done, _ = self.env.step(self.action.squeeze())
            reward = np.array([reward], np.float32).reshape(1, self.n_agents)
            done = np.array(done).reshape(1, self.n_agents)

            self.current_reward = reward
            self.total_return += reward
            if np.max(self.total_return) > self.max_return:
                self.max_return = np.max(self.total_return)

            # Store transition
            # Down-sample transition to reduce correlation between samples
            transitions.append(AttrDict(
                state=state,
                pi_stats=pi_stats,
                action=self.action.copy(),
                reward=reward.copy(),
                done=done.copy()
            ))

            if prev_done:
                break

            if np.any(done):
                prev_done = True

            self.state = next_state

        rollout = self.process_rollouts(transitions, seed)

        return rollout

    def process_rollouts(self, trans, seed):

        states = AttrDict({
            key: np.stack([t.state[key] for t in trans])
            for key in trans[0].state.keys()
        })

        # len(states) = len(action) + 1
        trans = trans[:-1]

        action = np.stack([t.action.T for t in trans])
        reward = np.stack([t.reward.T for t in trans])
        done = np.stack([t.done.T for t in trans])

        if trans[0].pi_stats is not None:
            pi_stats = {
                k: np.concatenate([t.pi_stats[k] for t in trans])
                for k in trans[0].pi_stats.keys()
            }
        else:
            pi_stats = None

        return AttrDict(
            states = states,
            action = action,
            reward = reward,
            done = done,
            pi_stats = pi_stats,
            seq_length = len(trans),
            batch_size = self.n_agents,
            seed = seed,
        )

    def get_partial_rollout(self, rollout, max_length):

        if rollout.seq_length <= max_length:
            return rollout

        start = np.random.randint(max(0, rollout.seq_length - max_length))
        end = start + max_length
        s  = slice(start, end)
        s1 = slice(start, end + 1)

        return AttrDict(
            states = AttrDict({k: v[s1] for k, v in rollout.states.iteritems()}),
            action = rollout.action[s],
            reward = rollout.reward[s],
            done = rollout.done[s],
            pi_stats = None if rollout.pi_stats is None else {k: v[s] for k, v in rollout.pi_stats.iteritems()},
            seq_length = min(rollout.seq_length, max_length),
            batch_size = rollout.batch_size,
            seed = rollout.seed,
        )

    def summarize_rollout(self, rollout):

        for k, v in rollout.states.iteritems():
            tf.logging.info("states[{}].shape = {}".format(k, v.shape))

        for key in ["action", "reward", "done"]:
            tf.logging.info("{}.shape = {}".format(key, rollout[key].shape))

        for k, v in rollout.pi_stats.iteritems():
            tf.logging.info("pi_stats[{}].shape = {}".format(k, v.shape))

        for key in ["seq_length", "batch_size", "seed"]:
            tf.logging.info("{} = {}".format(key, rollout[key]))

    def collect_statistics(self, rollout):
        avg_total_return = np.mean(self.total_return)
        self.global_episode_stats.append(
            rollout.seq_length, avg_total_return, self.total_return.flatten()
        )

    def store_experience(self, rollout):
        # Some bad simulation can have episode length 0 or 1, and that's outlier
        if rollout.seq_length <= 1:
            return

        self.collect_statistics(rollout)

        # Store rollout in the replay buffer, discard the oldest by popping
        # the 1st element if it exceeds maximum buffer size
        rp = Worker.replay_buffer

        rp.append(rollout)

        if len(rp) % 100 == 0:
            tf.logging.info("len(replay_buffer) = {}".format(len(rp)))

    def run(self, sess, coord):

        self.sess = sess
        self.coord = coord

        with sess.as_default(), sess.graph.as_default():
            try:
                while not coord.should_stop() and not Worker.stop:
                    self._run()
            except tf.errors.CancelledError:
                return
            except:
                print "\33[91m"
                traceback.print_exc()

Worker.stop = False
Worker.replay_buffer = deque(maxlen=FLAGS.max_replay_buffer_size)
