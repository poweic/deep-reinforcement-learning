import traceback
import itertools
import threading
import numpy as np
import tensorflow as tf
from drl.ac.utils import *
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
    def __init__(self, name, env, global_counter, global_episode_stats, global_net, add_summaries,
                 n_agents=1):

        self.name = name
        self.env = env
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

        # Assign each worker (thread) a memory replay buffer
        self.replay_buffer = ReplayBuffer(maxlen=FLAGS.max_replay_buffer_size)

    def copy_params_from_global(self):
        # Copy Parameters from the global networks
        self.sess.run(self.copy_params_op)

    def reset_env(self):

        # Re-seed environment and store the seed
        self.seed = self.env.seed()

        # Get environmental state
        self.env_state = self.env.reset()

        # Get initial hidden states (this can also be viewed as "state")
        self.hidden_states = self.local_net.get_initial_hidden_states(self.n_agents)

        # intialize action to 0
        self.action = np.zeros((FLAGS.num_actions, self.n_agents), dtype=np.float32)

        # intialize reward & return to 0
        self.reward = np.zeros((1, self.n_agents), dtype=np.float32)
        self.total_return = np.zeros((1, self.n_agents), dtype=np.float32)

        # Set initial timestamp
        self.global_episode_stats.set_initial_timestamp()

    def run_n_steps(self, n_steps):

        transitions = []

        for i in range(n_steps):

            # Note: state is "fully observable" state, it contains env.state,
            # lstm.hidden_states, and other things like prev_action and reward
            state = form_state(self.env, self.env_state, self.action, self.reward, self.hidden_states)

            # Predict an action
            self.action, pi_stats, self.hidden_states = \
                self.local_net.predict_actions(state, self.sess)

            # Take a step in environment
            self.env_state, self.reward, done, _ = self.env.step(self.action.squeeze())
            self.reward = np.array([self.reward], np.float32).reshape(1, self.n_agents)
            done = np.array(done).reshape(1, self.n_agents)

            self.total_return += self.reward

            if np.max(self.total_return) > self.max_return:
                self.max_return = np.max(self.total_return)

            # Store transition
            # Down-sample transition to reduce correlation between samples
            transitions.append(AttrDict(
                state=state,
                pi_stats=pi_stats,
                action=self.action.copy(),
                reward=self.reward.copy(),
                done=done.copy()
            ))

            if np.any(done):
                break

        transitions.append(AttrDict(state=form_state(
            self.env, self.env_state, self.action, self.reward, self.hidden_states
        )))

        rollout = self.process_rollouts(transitions)
        rollout.r = self.total_return
        rollout.seed = self.seed

        # reset environment if it's terminated
        if np.any(done):
            self.collect_statistics(rollout)
            self.reset_env()

        return rollout

    def process_rollouts(self, trans):

        states = AttrDict({
            key: np.stack([t.state[key] for t in trans])
            for key in trans[0].state.keys()
        })

        # len(states) = len(action) + 1
        trans = trans[:-1]

        action = np.stack([t.action.T for t in trans])
        reward = np.stack([t.reward.T for t in trans])
        done = np.stack([t.done.T for t in trans])

        pi_stats = None
        if trans[0].pi_stats is not None:
            pi_stats = {
                k: np.concatenate([t.pi_stats[k] for t in trans])
                for k in trans[0].pi_stats.keys()
            }

        return AttrDict(
            states = states,
            action = action,
            reward = reward,
            done = done,
            pi_stats = pi_stats,
            seq_length = len(trans),
            batch_size = self.n_agents,
        )

    def get_partial_rollout(self, rollout, length=None, start=None):
        """
        Returns a random slice of rollout consisting of
        min(length, FLAGS.max_seq_length) steps
        """

        # if length is not specified, then use the full sequence
        if length is None:
            length = rollout.seq_length

        # the length can't be longer than max_seq_length, which is used to
        # protect GPU OOM (out of memory) error
        length = min(length, FLAGS.max_seq_length)

        # Decide a start index, either from 0 if rollout is not long enough, or
        # randomly choose one between 0 to max(0, rollout.seq_length - length)
        if rollout.seq_length <= length:
            start = 0

        if start is None:
            start = np.random.randint(max(0, rollout.seq_length - length))

        # We use slice(start, end) for most of the attributes in rollout, but
        # slice(start, end + 1) for states, since we need to bootstrap value
        # from the last state
        end = start + length
        s  = slice(start, end)
        s1 = slice(start, end + 1)

        lstm_state_keys = self.local_net.lstm.inputs.keys()

        return AttrDict(
            states = AttrDict({
                k: v[s1] if k not in lstm_state_keys else v[start]
                for k, v in rollout.states.iteritems()
            }),
            action = rollout.action[s],
            reward = rollout.reward[s],
            done = rollout.done[s],
            pi_stats = None if rollout.pi_stats is None else {k: v[s] for k, v in rollout.pi_stats.iteritems()},
            seq_length = min(rollout.seq_length, length),
            batch_size = rollout.batch_size,
            seed = rollout.seed,
        )

    def batch_rollouts(self, rollouts):

        for rollout in rollouts:
            assert rollout.seq_length == rollouts[0].seq_length

        def concat(key):
            return np.concatenate([r[key] for r in rollouts], axis=-2)

        return AttrDict(
            states = {
                k: np.concatenate(
                    [r.states[k] for r in rollouts],
                    axis=(-2 if k is not "front_view" else 1)
                )
                for k in rollouts[0].states.keys()
            },
            action = concat('action'),
            reward = concat('reward'),
            done = concat('done'),
            pi_stats = {
                k: np.concatenate([r.pi_stats[k] for r in rollouts], axis=-2)
                for k in rollouts[0].pi_stats.keys()
            },
            seq_length = rollouts[0].seq_length,
            batch_size = self.n_agents * len(rollouts),
            seed = [r.seed for r in rollouts],
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
        avg_total_return = np.mean(rollout.r)
        self.global_episode_stats.append(
            rollout.seq_length, avg_total_return, rollout.r.flatten()
        )

    def store_experience(self, rollout):
        # Some bad simulation can have episode length 0 or 1, and that's outlier
        if rollout.seq_length <= 1:
            return

        # Store rollout in the replay buffer, discard the oldest by popping
        # the 1st element if it exceeds maximum buffer size
        rp = self.replay_buffer

        rp.append(rollout)

        if len(rp) % 100 == 0 and len(rp) < FLAGS.max_replay_buffer_size:
            tf.logging.info("len(replay_buffer) = {}".format(len(rp)))

    def should_stop(self):

        # Condition 1: maximum step reached
        # max_step_reached = self.gstep > FLAGS.max_global_steps
        max_step_reached = self.gstep > FLAGS.max_global_steps

        # Condition 2: problem solved by achieving a high average reward over
        # last consecutive N episodes
        stats = self.global_episode_stats
        mean, std = stats.last_n_stats()

        """
        solved = stats.num_episodes() > FLAGS.min_episodes \
            and mean > FLAGS.score_to_win
        """

        # return (max_step_reached or solved)
        return max_step_reached

    def run(self, sess, coord):

        self.sess = sess
        self.coord = coord

        self.reset_env()

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
Worker.lock = threading.Lock()
# Worker.pause = False
