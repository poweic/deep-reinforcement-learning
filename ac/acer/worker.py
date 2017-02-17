# -*- coding: utf-8 -*-
import gc
from collections import OrderedDict, deque
import tensorflow as tf
import ac.acer.estimators
from ac.worker import Worker
from ac.utils import *
import time

class AcerWorker(Worker):
    """
    An A3C worker thread. Runs episodes locally and updates global shared value and policy nets.

    Args:
    name: A unique name for this worker
    env: The Gym environment used by this worker
    global_net: Instance of the globally shared network
    global_counter: Iterator that holds the global step
    summary_writer: A tf.train.SummaryWriter for Tensorboard summaries
    """
    def __init__(self, **kwargs):
        self.Estimator = ac.acer.estimators.AcerEstimator
        super(AcerWorker, self).__init__(**kwargs)

    def set_global_net(self, global_net):
        # Get global, local, and the average net var_list
        avg_vars = self.Estimator.average_net.var_list
        global_vars = global_net.var_list
        local_vars = self.local_net.var_list

        # Operation to copy params from global net to local net
        self.copy_params_op = make_copy_params_op(global_vars, local_vars)

        self.global_net = global_net
        self.gstep = 0
        self.timer = 0
        self.timer_counter = 0

        self.prev_debug = None
        self.prev_mdp_states = None

        train_and_update_avgnet_op = ac.acer.estimators.create_avgnet_init_op(
            self.global_step, avg_vars, global_net, self.local_net
        )

        with tf.control_dependencies([train_and_update_avgnet_op]):
            self.inc_global_step = tf.assign_add(self.global_step, 1)

        net = self.local_net
        self.step_op = [
            {
                'pi': net.pi_loss,
                'vf': net.vf_loss,
                'total': net.loss,
                'global_norm': net.global_norm,
            },
            net.summaries,
            train_and_update_avgnet_op,
            tf.no_op(),
            self.inc_global_step,
        ]

    def reset_env(self):

        self.state = self.env.reset()

        # Reshape to compatiable format
        # self.state = self.state.astype(np.float32).reshape(6, -1)
        self.action = np.zeros((FLAGS.num_actions, self.n_agents), dtype=np.float32)

        self.total_return = np.zeros((1, self.n_agents), dtype=np.float32)
        self.current_reward = np.zeros((1, self.n_agents), dtype=np.float32)

        # Set initial timestamp
        self.global_episode_stats.set_initial_timestamp()

    def _run(self):

        # Show learning rate every FLAGS.decay_steps
        if self.gstep % FLAGS.decay_steps == 0:
            tf.logging.info("learning rate = {}".format(self.sess.run(self.local_net.lr)))

        # show_mem_usage()

        # Run on-policy ACER
        self._run_on_policy()

        # Run off-policy ACER N times
        self._run_off_policy_n_times()

        if self.should_stop():
            Worker.stop = True
            self.coord.request_stop()

    def copy_params_from_global(self):
        # Copy Parameters from the global networks
        self.sess.run(self.copy_params_op)

    def store_experience(self, transitions):
        # Some bad simulation can have episode length 0 or 1, and that's outlier
        if len(transitions) <= 1:
            return

        self._collect_statistics(transitions)

        # Store transitions in the replay buffer, discard the oldest by popping
        # the 1st element if it exceeds maximum buffer size
        rp = AcerWorker.replay_buffer

        rp.append(transitions)

        if len(rp) % 100 == 0:
            tf.logging.info("len(replay_buffer) = {}".format(len(rp)))

    def should_stop(self):

        # Condition 1: maximum step reached
        max_step_reached = self.gstep > FLAGS.max_global_steps

        # Condition 2: problem solved by achieving a high average reward over
        # last consecutive N episodes
        stats = self.global_episode_stats
        mean, std, msg = stats.last_n_stats()
        tf.logging.info("\33[93m" + msg + "\33[0m")

        """
        solved = stats.num_episodes() > FLAGS.min_episodes and mean > FLAGS.score_to_win
        """

        # if max_step_reached or solved:
        if max_step_reached:
            """ tf.logging.info("Optimization done. @ step {} because {}".format(
            self.gstep, "problem solved." if solved else "maximum steps reached"
            )) """
            tf.logging.info("Optimization done. @ step {}".format(self.gstep))
            tf.logging.info(stats.summary())
            save_model(self.sess)
            return True

        return False

    def _run_on_policy(self):
        self.copy_params_from_global()

        # Collect transitions {(s_0, a_0, r_0, mu_0), (s_1, ...), ... }
        n_steps = FLAGS.max_steps
        transitions = self.run_n_steps(n_steps)
        # tf.logging.info("Average time to predict actions: {}".format(self.timer / self.timer_counter))

        # Compute gradient and Perform update
        self.update(transitions)

        # Store experience and collect statistics
        self.store_experience(transitions)

        if len(transitions) > 1:
            self._collect_statistics(transitions)

    def _collect_statistics(self, transitions):
        avg_total_return = np.mean(self.total_return)
        self.global_episode_stats.append(
            len(transitions), avg_total_return, self.total_return.flatten()
        )

        self.gstep = self.sess.run(self.global_step)
        if self.gstep % 10 == 0:
            # also print experiment configuration in MATLAB parseable JSON
            cfg = "'" + repr(FLAGS.exp_config)[1:-1].replace("'", '"') + "'\n"
            print >> open(FLAGS.stats_file, 'w'), cfg, self.global_episode_stats

    def _run_off_policy_n_times(self):
        N = np.random.poisson(FLAGS.replay_ratio)

        """
        for i in range(N):
            self._run_off_policy()
        """
        self._run_off_policy(N)

    def _run_off_policy(self, N):
        rp = AcerWorker.replay_buffer

        if len(rp) <= N:
            return

        # Random select on episode from past experiences
        # idx = np.random.randint(len(rp))
        # lengths = np.array([len(t) for t in rp], dtype=np.float32)
        lengths = np.array([1 for t in rp], dtype=np.float32)
        prob = lengths / np.sum(lengths)
        # tf.logging.info("lengths = {}, prob = {}, len(prob) = {}, len(rp) = {}".format(lengths, prob, len(prob), len(rp)))
        indices = np.random.choice(len(prob), N, p=prob, replace=False)
        # tf.logging.info("len(indices) = {}".format(len(indices)))

        for idx in indices:
            self.copy_params_from_global()

            # Compute gradient and Perform update
            self.update(rp[idx], on_policy=False)

    def run_n_steps(self, n_steps):

        transitions = []

        # Initial state
        self.reset_env()
        self.local_net.reset_lstm_state()

        reward = np.zeros((1, self.n_agents), dtype=np.float32)
        for i in range(n_steps):

            state = form_state(self.state, self.action, reward)

            # Predict an action
            self.action, pi_stats = self.local_net.predict_actions(state, self.sess)

            next_state, reward, done, _ = self.env.step(self.action.squeeze())
            reward = np.array([reward], np.float32).reshape(1, self.n_agents)

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
                next_state=next_state.copy(),
                reward=reward.copy(),
                done=done
            ))

            if done:
                break

            self.state = next_state

        return transitions

    def update(self, trans, on_policy=True):

        if len(trans) == 0:
            return

        states = AttrDict({
            key: np.concatenate([
                t.state[key][None, ...] for t in trans
            ], axis=0)
            for key in trans[0].state.keys()
        })

        S, B = len(trans), self.n_agents

        action = np.concatenate([t.action.T[None, ...] for t in trans], axis=0)
        reward = np.concatenate([t.reward.T[None, ...] for t in trans], axis=0)

        # Start to put things in placeholders in graph
        net = self.local_net
        avg_net = self.Estimator.average_net

        feed_dict = {
            net.r: reward,
            net.a: action,
        }

        feed_dict.update({net.state[k]:     v for k, v in states.iteritems()})
        feed_dict.update({avg_net.state[k]: v for k, v in states.iteritems()})

        for k in trans[0].pi_stats.keys():
            feed_dict.update({net.pi_behavior.stats[k]: np.concatenate([t.pi_stats[k] for t in trans], axis=0)})

        net.reset_lstm_state()

        loss, summaries, _, debug, self.gstep = net.update(self.step_op, feed_dict, self.sess)
        loss = AttrDict(loss)

        if self.gstep % 100 == 0:
            tf.logging.info((
                "#{:6d}: pi_loss = {:+12.3f}, vf_loss = {:+12.3f}, "
                "loss = {:+12.3f} {}\33[0m S = {:3d}, B = {} [{}] global_norm = {:.2f}"
            ).format(
                self.gstep, loss.pi, loss.vf, loss.total,
                "\33[92m[on  policy]" if on_policy else "\33[93m[off policy]",
                S, B, self.name, loss.global_norm
            ))

        """
        if "grad_norms" in loss:
            grad_norms = OrderedDict(sorted(loss.grad_norms.items()))
            max_len = max(map(len, grad_norms.keys()))
            for k, v in grad_norms.iteritems():
                tf.logging.info("{} grad norm: {}{:12.6e}\33[0m".format(
                    k.ljust(max_len), "\33[94m" if v > 0 else "\33[2m", v))
        """

AcerWorker.replay_buffer = deque(maxlen=FLAGS.max_replay_buffer_size)
