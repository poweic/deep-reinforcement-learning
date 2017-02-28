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
            self.inc_global_step, # TODO is it the correct way to use?
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

    """
    def reset_env(self):

        self.state = np.array([+1, 1, -10 * np.pi / 180, 0, 0, 0])
        self.action = np.array([0, 0])

        # Reshape to compatiable format
        self.state = self.state.astype(np.float32).reshape(6, -1)
        self.action = np.zeros((2, self.n_agents), dtype=np.float32)
        self.total_return = np.zeros((1, self.n_agents), dtype=np.float32)
        self.current_reward = np.zeros((1, self.n_agents), dtype=np.float32)

        # Add some noise to have diverse start points
        noise = np.random.randn(6, self.n_agents).astype(np.float32) * 0.5
        noise[2, :] /= 2

        self.state = self.state + noise

        self.env._reset(self.state)
    """

    def _run(self):

        # if resume from some unfinished training, we first re-generate
        # lots of experiecnes before further training/updating
        if FLAGS.regenerate_exp_after_resume:
            self.regenerate_experiences()
            FLAGS.regenerate_exp_after_resume = False

        # Show learning rate every FLAGS.decay_steps
        if self.gstep % FLAGS.decay_steps == 0:
            tf.logging.info("learning rate = {}".format(self.sess.run(self.local_net.lr)))

        show_mem_usage()

        # Run on-policy ACER
        self._run_on_policy()

        # Run off-policy ACER N times
        self._run_off_policy_n_times()

        if self.should_stop():
            Worker.stop = True
            self.coord.request_stop()

    def regenerate_experiences(self):
        N = FLAGS.regenerate_size
        tf.logging.info("Re-generating {} experiences ...".format(N))

        self.copy_params_from_global()
        while len(AcerWorker.replay_buffer) < N:
            self.store_experience(self.run_n_steps(FLAGS.max_steps))

        tf.logging.info("Regeneration done. len(replay_buffer) = {}.".format(
            len(AcerWorker.replay_buffer)))

    def copy_params_from_global(self):
        # Copy Parameters from the global networks
        self.sess.run(self.copy_params_op)

    def store_experience(self, rollout):
        # Some bad simulation can have episode length 0 or 1, and that's outlier
        if rollout.seq_length <= 1:
            return

        self._collect_statistics(rollout)

        # Store rollout in the replay buffer, discard the oldest by popping
        # the 1st element if it exceeds maximum buffer size
        rp = AcerWorker.replay_buffer

        rp.append(rollout)

        if len(rp) % 100 == 0:
            tf.logging.info("len(replay_buffer) = {}".format(len(rp)))

    def should_stop(self):

        # Condition 1: maximum step reached
        # max_step_reached = self.gstep > FLAGS.max_global_steps
        """ FIXME from gym-offroad-nav:master
        global_timestep = self.sess.run(FLAGS.global_timestep)
        t, lr = self.sess.run([FLAGS.global_timestep, self.local_net.lr])
        tf.logging.info("global_timestep = {}, learning rate = {}".format(t, lr))
        max_step_reached = global_timestep > FLAGS.max_global_steps
        """
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
            self._write_statistics()
            return True

        return False

    def _run_on_policy(self):
        self.copy_params_from_global()

        # Collect rollout {(s_0, a_0, r_0, mu_0), (s_1, ...), ... }
        # FLAGS.max_steps = int(np.ceil(FLAGS.t_max * FLAGS.command_freq))
        rollout = self.run_n_steps(FLAGS.max_steps)

        # Compute gradient and Perform update
        self.update(rollout)

        # Store experience and collect statistics
        self.store_experience(rollout)

        if rollout.seq_length > 1:
            self._collect_statistics(rollout)

    def _collect_statistics(self, rollout):
        avg_total_return = np.mean(self.total_return)
        self.global_episode_stats.append(
            rollout.seq_length, avg_total_return, self.total_return.flatten()
        )

        self.gstep = self.sess.run(self.global_step)
        if self.gstep % 1000 == 0:
            self._write_statistics()

    def _write_statistics(self):
        # also print experiment configuration in MATLAB parseable JSON
        cfg = "'" + repr(FLAGS.exp_config)[1:-1].replace("'", '"') + "'\n"
        print >> open(FLAGS.stats_file, 'w'), cfg, self.global_episode_stats

    def _run_off_policy_n_times(self):
        N = np.random.poisson(FLAGS.replay_ratio)
        self._run_off_policy(N)

    def _run_off_policy(self, N):
        rp = AcerWorker.replay_buffer

        if len(rp) <= N:
            return

        # Random select on episode from past experiences
        # idx = np.random.randint(len(rp))
        # lengths = np.array([len(t) for t in rp], dtype=np.float32)
        if FLAGS.prioritize_replay:
            lengths = np.array([len(t) for t in rp], dtype=np.float32)
            prob = lengths / np.sum(lengths)
            indices = np.random.choice(len(prob), N, p=prob, replace=False)
        else:
            indices = np.random.randint(len(rp), size=N)

        for i, idx in enumerate(indices):
            self.copy_params_from_global()

            # Compute gradient and Perform update
            self.update(rp[idx], on_policy=False, display=(i == 0))

    def run_n_steps(self, n_steps):

        transitions = []

        # Initial state
        self.reset_env()
        self.local_net.reset_lstm_state()

        reward = np.zeros((1, self.n_agents), dtype=np.float32)
        for i in range(n_steps):

            # state = form_mdp_state(self.env, self.state, self.action, reward)
            state = form_state(self.state, self.action, reward)

            # Predict an action
            self.action, pi_stats = self.local_net.predict_actions(state, self.sess)

            """ FIXME move this section to environment
            # Take several sub-steps in environment (the smaller the timestep,
            # the smaller each sub-step, the more accurate the simulation
            n_sub_steps = int(1. / FLAGS.command_freq / FLAGS.timestep)
            # state___ = self.env.state.copy()
            for j in range(n_sub_steps):
                next_state, reward, done, _ = self.env.step(self.action)
            """

            """
            # self.env.state = state___.copy()
            next_state, reward, done, _ = self.env._step_2(self.action, n_sub_steps)

            print np.linalg.norm(next_state - next_state_2)
            print np.linalg.norm(reward     - reward_2    )
            print np.linalg.norm(done       - done_2      )
            """

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
                done=done # done.copy()
            ))

            if done: # if np.any(done):
                break

            self.state = next_state

        rollout = self.process_rollouts(transitions)

        return rollout

    """
    # FIXME turn this function to another process_rollouts, and figure out
    # how to make this configuration (this shouldn't be task-dependent!!)
    def update(self, trans, on_policy=True):

        if len(trans) == 0:
            return

        mdp_states = AttrDict({
            key: np.concatenate([
                t.mdp_state[key][None, ...] for t in trans
            ], axis=0)
            for key in trans[0].mdp_state.keys()
        })

        S, B = mdp_states.front_view.shape[:2]

        action = np.concatenate([t.action.T[None, ...] for t in trans], axis=0)
        reward = np.concatenate([t.reward.T[None, ...] for t in trans], axis=0)

        net = self.local_net
        avg_net = self.Estimator.average_net

        feed_dict = {
            net.r: reward,
            net.a: action,
        }

        feed_dict.update({net.state[k]:     v for k, v in mdp_states.iteritems()})
        feed_dict.update({avg_net.state[k]: v for k, v in mdp_states.iteritems()})

        for k, v in trans[0].pi_stats.iteritems():
            for i in range(len(v)):
                tensor = net.pi_behavior.stats[k][i]
                feed_dict[tensor] = np.concatenate([
                    t.pi_stats[k][i] for t in trans
                ], axis=0)

        # ======================= DEBUG =================================
        if FLAGS.dump_crash_report:
            debug_keys = [
                'a', 'pi_a', 'mu_a', 'a_prime', 'pi_a_prime', 'mu_a_prime',
                'rho', 'g_phi', 'g_acer',
                'Q_ret', 'Q_opc', 'Q_tilt_a', 'Q_tilt_a_prime',
                'rho_bar', 'log_a' , 'target_1', 'value',
                'plus'   , 'log_ap', 'target_2',
                # 'pi_mu', 'mu_mu', 'pi_sigma', 'mu_sigma'
            ]

            ops[-1] = { k: getattr(net, k) for k in debug_keys }
        # ======================= DEBUG =================================

        net.reset_lstm_state()
        avg_net.reset_lstm_state()

        loss, summaries, _, debug = net.predict(self.step_op, feed_dict, self.sess)
        loss = AttrDict(loss)

        self.gstep = self.sess.run(self.inc_global_step)
        tf.logging.info((
            "#{:6d}: pi_loss = {:+12.3f}, vf_loss = {:+12.3f}, "
            "loss = {:+12.3f} {}\33[0m S = {:3d}, B = {} [{}] global_norm = {:.2f}"
        ).format(
            self.gstep, loss.pi, loss.vf, loss.total,
            "\33[92m[on  policy]" if on_policy else "\33[93m[off policy]",
            S, B, self.name, loss.global_norm
        ))
    """

    def process_rollouts(self, trans):

        states = AttrDict({
            key: np.concatenate([
                t.state[key][None, ...] for t in trans
            ], axis=0)
            for key in trans[0].state.keys()
        })

        S, B = len(trans), self.n_agents

        action = np.concatenate([t.action.T[None, ...] for t in trans], axis=0)
        reward = np.concatenate([t.reward.T[None, ...] for t in trans], axis=0)
        pi_stats = {
            k: np.concatenate([t.pi_stats[k] for t in trans], axis=0)
            for k in trans[0].pi_stats.keys()
        }

        return AttrDict(
            states = states,
            action = action,
            reward = reward,
            pi_stats = pi_stats,
            seq_length = S,
            batch_size = B,
        )

    def update(self, rollout, on_policy=True, display=True):

        if rollout.seq_length == 0:
            return

        # Start to put things in placeholders in graph
        net = self.local_net
        avg_net = self.Estimator.average_net

        # To feeddict
        feed_dict = {
            net.r: rollout.reward,
            net.a: rollout.action,
        }

        feed_dict.update({net.state[k]:     v for k, v in rollout.states.iteritems()})
        feed_dict.update({avg_net.state[k]: v for k, v in rollout.states.iteritems()})
        feed_dict.update({net.pi_behavior.stats[k]: v for k, v in rollout.pi_stats.iteritems()})

        net.reset_lstm_state()

        loss, summaries, _, debug, self.gstep = net.update(self.step_op, feed_dict, self.sess)
        loss = AttrDict(loss)

        if display:
            tf.logging.info((
                "#{:6d}: pi_loss = {:+12.3f}, vf_loss = {:+12.3f}, "
                "loss = {:+12.3f} {}\33[0m S = {:3d}, B = {} [{}] global_norm = {:.2f}"
            ).format(
                self.gstep, loss.pi, loss.vf, loss.total,
                "\33[92m[on  policy]" if on_policy else "\33[93m[off policy]",
                rollout.seq_length, rollout.batch_size, self.name, loss.global_norm
            ))

        """
        if "grad_norms" in loss:
            grad_norms = OrderedDict(sorted(loss.grad_norms.items()))
            max_len = max(map(len, grad_norms.keys()))
            for k, v in grad_norms.iteritems():
                tf.logging.info("{} grad norm: {}{:12.6e}\33[0m".format(
                    k.ljust(max_len), "\33[94m" if v > 0 else "\33[2m", v))
        """

        """
        # ======================= DEBUG =================================
        if FLAGS.dump_crash_report and np.isnan(loss.total):
            np.set_printoptions(precision=4, linewidth=500, suppress=True, formatter={
                'float_kind': lambda x: ("\33[2m" if x == 0 else "") + (("{:+12.5e}" if abs(x) < 1e-9 or abs(x) > 10 else "{:+12.9f}").format(x)) + "\33[0m"
            })
            
            for k in debug_keys:
                tf.logging.info("\33[93m {} [{}] = \33[0m\n{}".format(k, debug[k].shape, debug[k]))

            import scipy.io
            scipy.io.savemat("debug.mat", debug)
            scipy.io.savemat("prev_debug.mat", self.prev_debug)

            scipy.io.savemat("mdp_states.mat", mdp_states)
            scipy.io.savemat("prev_mdp_states.mat", self.prev_mdp_states)

            import ipdb; ipdb.set_trace()

            np.set_printoptions()

            self.prev_debug = debug
            self.prev_mdp_states = mdp_states
        # ======================= DEBUG =================================

        self.sess.run(FLAGS.set_time_op, feed_dict={
            FLAGS.global_timestep_placeholder: int(time.time())
        })

        # Write summaries
        if self.summary_writer is not None:
            self.summary_writer.add_summary(summaries, self.gstep)
            self.summary_writer.flush()
        """

AcerWorker.replay_buffer = deque(maxlen=FLAGS.max_replay_buffer_size)
