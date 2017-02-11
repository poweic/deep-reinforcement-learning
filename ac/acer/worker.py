# -*- coding: utf-8 -*-
import gc
from collections import OrderedDict
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

        def copy_global_to_avg():
            msg = "\33[94mInitialize average net when global_step = \33[0m"
            disp_op = tf.Print(self.global_step, [self.global_step], msg)
            copy_op = make_copy_params_op(global_vars, avg_vars)
            return tf.group(*[copy_op, disp_op])

        init_avg_net = tf.cond(
            tf.equal(self.global_step, 0),
            copy_global_to_avg,
            lambda: tf.no_op()
        )

        with tf.control_dependencies([init_avg_net]):
            train_op = make_train_op(self.local_net, global_net)

            with tf.control_dependencies([train_op]):

                train_and_update_avgnet_op = make_copy_params_op(
                    global_vars, avg_vars, alpha=FLAGS.avg_net_momentum
                )

                net = self.local_net

                self.step_op = [
                    {
                        'pi': net.pi_loss,
                        'vf': net.vf_loss,
                        'total': net.loss,
                        'global_norm': net.global_norm,
                        # 'grad_norms': net.grad_norms
                    },
                    net.summaries,
                    train_and_update_avgnet_op,
                    tf.no_op()
                ]

        self.inc_global_step = tf.assign_add(self.global_step, 1)

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

        # Set initial timestamp
        self.global_episode_stats.set_initial_timestamp()

    def _run(self):
        
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
        if len(rp) > FLAGS.max_replay_buffer_size:
            rp.pop(0)
            gc.collect()

        if len(rp) % 20 == 0:
            tf.logging.info("len(replay_buffer) = {}".format(len(rp)))

    def should_stop(self):
        # Condition 1: maximum step reached
        max_step_reached = self.gstep > FLAGS.max_global_steps

        # Condition 2: problem solved by achieving a high average reward over
        # last consecutive N episodes
        stats = self.global_episode_stats
        mean, std, msg = stats.last_n_stats()
        tf.logging.info(msg)

        solved = stats.num_episodes() > FLAGS.min_episodes and mean > FLAGS.score_to_win

        if max_step_reached or solved:
            tf.logging.info("Optimization done. @ step {} because {}".format(
                self.gstep, "problem solved." if solved else "maximum steps reached"
            ))
            tf.logging.info(stats.summary())
            save_model(self.sess)
            return True

        return False

    def _run_on_policy(self):
        self.copy_params_from_global()

        # Collect transitions {(s_0, a_0, r_0, mu_0), (s_1, ...), ... }
        n_steps = int(np.ceil(FLAGS.t_max * FLAGS.command_freq))
        transitions = self.run_n_steps(n_steps)
        # tf.logging.info("Average time to predict actions: {}".format(self.timer / self.timer_counter))

        # Compute gradient and Perform update
        self.update(transitions)

        # Store experience and collect statistics
        self.store_experience(transitions)

    def _collect_statistics(self, transitions):
        avg_total_return = np.mean(self.total_return)
        self.global_episode_stats.append(
            len(transitions), avg_total_return, self.total_return.flatten()
        )

        self.gstep = self.sess.run(self.inc_global_step)
        if self.gstep % 10 == 0:
            print >> open(FLAGS.stats_file, 'w'), self.global_episode_stats

    def _run_off_policy_n_times(self):
        N = np.random.poisson(FLAGS.replay_ratio)

        for i in range(N):
            self._run_off_policy()

    def _run_off_policy(self):
        rp = AcerWorker.replay_buffer

        if len(rp) <= 0:
            return

        self.copy_params_from_global()

        # Random select on episode from past experiences
        idx = np.random.randint(len(rp))

        # Compute gradient and Perform update
        self.update(rp[idx], on_policy=False)

    def run_n_steps(self, n_steps):

        transitions = []

        # Initial state
        self.reset_env()
        self.local_net.reset_lstm_state()

        reward = np.zeros((1, self.n_agents), dtype=np.float32)
        for i in range(n_steps):

            mdp_state = form_mdp_state(self.env, self.state, self.action, reward)

            # Predict an action
            self.timer -= time.time()
            self.action, pi_stats = self.local_net.predict_actions(mdp_state, self.sess)

            self.timer += time.time()
            self.timer_counter += 1

            assert not np.any(np.isnan(self.action)), "i = {}, self.action = {}, mdp_state = {}".format(i, self.action, mdp_state)

            # Take several sub-steps in environment (the smaller the timestep,
            # the smaller each sub-step, the more accurate the simulation
            n_sub_steps = int(1. / FLAGS.command_freq / FLAGS.timestep)
            for j in range(n_sub_steps):
                next_state, reward, done, _ = self.env.step(self.action)

            self.current_reward = reward
            self.total_return += reward
            if np.max(self.total_return) > self.max_return:
                self.max_return = np.max(self.total_return)

            # Store transition
            # Down-sample transition to reduce correlation between samples
            transitions.append(AttrDict(
                mdp_state=mdp_state,
                pi_stats=pi_stats,
                action=self.action.copy(),
                next_state=next_state.copy(),
                reward=reward.copy(),
                done=done.copy()
            ))

            if np.any(done):
                break
            else:
                self.state = next_state

        return transitions

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

        if "grad_norms" in loss:
            grad_norms = OrderedDict(sorted(loss.grad_norms.items()))
            max_len = max(map(len, grad_norms.keys()))
            for k, v in grad_norms.iteritems():
                tf.logging.info("{} grad norm: {}{:12.6e}\33[0m".format(
                    k.ljust(max_len), "\33[94m" if v > 0 else "\33[2m", v))

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

        # Write summaries
        if self.summary_writer is not None:
            self.summary_writer.add_summary(summaries, self.gstep)
            self.summary_writer.flush()

AcerWorker.replay_buffer = []
