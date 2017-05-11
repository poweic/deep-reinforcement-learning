# -*- coding: utf-8 -*-
import gc
from collections import OrderedDict
import tensorflow as tf
import drl.ac.acer.estimators
from drl.ac.worker import Worker
from drl.ac.utils import *
from drl.ac.estimators import create_avgnet_init_op
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
        self.Estimator = drl.ac.acer.estimators.AcerEstimator
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

        train_and_update_avgnet_op = create_avgnet_init_op(
            self.global_step, avg_vars, global_net, self.local_net
        )

        with tf.control_dependencies([train_and_update_avgnet_op]):
            self.inc_global_step = tf.assign_add(self.global_step, 1)

        net = self.local_net
        self.step_op = [
            {
                'pi': net.pi_loss,
                'vf': net.vf_loss,
                'entropy': net.entropy_loss,
                'total': net.loss,
                'global_norm': net.global_norm,
            },
            net.summaries,
            train_and_update_avgnet_op,
            {
                'Q_ret': net.Q_ret,
                'Q_tilt_a': net.Q_tilt_a,
                'value': net.value_all,
                'done': net.done
            },
            self.inc_global_step, # TODO is it the correct way to use?
        ]

    def _run(self):

        # if resume from some unfinished training, we first re-generate
        # lots of experiecnes before further training/updating
        if FLAGS.regenerate_exp_after_resume:
            self.regenerate_experiences()
            FLAGS.regenerate_exp_after_resume = False

        if FLAGS.show_memory_usage:
            show_mem_usage()

        # Run on-policy ACER
        self._run_on_policy()

        # Run off-policy ACER N times
        self._run_off_policy_n_times()

        if self.should_stop():
            tf.logging.info("Optimization done. @ step {}".format(self.gstep))
            Worker.stop = True
            self.coord.request_stop()

    def regenerate_experiences(self):
        N = FLAGS.regenerate_size
        tf.logging.info("Re-generating {} experiences ...".format(N))

        self.copy_params_from_global()
        while len(self.replay_buffer) < N:
            self.store_experience(self.run_n_steps())

        tf.logging.info("Regeneration done. len(replay_buffer) = {}.".format(
            len(self.replay_buffer)))

    def _run_on_policy(self):
        self.copy_params_from_global()

        # Collect rollout {(s_0, a_0, r_0, mu_0), (s_1, ...), ... }
        rollout = self.run_n_steps()

        # Compute gradient and Perform update
        self.update(self.get_partial_rollout(rollout))

        # Store experience and collect statistics
        self.store_experience(rollout)

        """
        # If this rollout is 3*std better than last 1000 average, then we run
        # this policy more times than usual (without copy_params_from_global)
        mean, std = self.global_episode_stats.last_n_stats(1000)
        if rollout.r - mean > 2 * std:
            n_more_times = int((rollout.r - mean) / std * 5)
            tf.logging.info("\33[96mRe-run {} more times\33[0m".format(
                n_more_times))

            for i in range(n_more_times):
                rollout = self.run_n_steps()
                self.update(rollout)
                self.store_experience(rollout)
        """

    def _run_off_policy_n_times(self):
        N = np.random.poisson(FLAGS.replay_ratio) * FLAGS.off_policy_batch_size
        self._run_off_policy(N)

    def _run_off_policy(self, N):
        rp = self.replay_buffer

        if len(rp) <= N:
            return

        # Random select on episode from past experiences
        # idx = np.random.randint(len(rp))
        # lengths = np.array([len(t) for t in rp], dtype=np.float32)
        if FLAGS.prioritize_replay:
            lengths = np.array([rp[i].seq_length for i in range(len(rp))], dtype=np.float32)
            prob = lengths / np.sum(lengths)
            indices = np.random.choice(len(prob), N, p=prob, replace=False)
        else:
            indices = np.random.randint(len(rp), size=N)

        for i, chucked_indices in enumerate(chunks(indices, FLAGS.off_policy_batch_size)):
            self.copy_params_from_global()

            """
            # Compute gradient and perform update
            length = min([rp[idx].seq_length for idx in chucked_indices])
            length = min(length, 64)
            rollouts = [
                self.get_partial_rollout(rp[idx], length=length)
                for idx in chucked_indices
            ]

            batched_rollouts = self.batch_rollouts(rollouts)
            """

            rollout = rp[chucked_indices[0]]
            batched_rollouts = self.get_partial_rollout(rollout)

            self.update(batched_rollouts, on_policy=False, display=(i == 0))

    def update(self, rollout, on_policy=True, display=True):

        if rollout.seq_length == 0:
            return

        # self.summarize_rollout(rollout)

        # Start to put things in placeholders in graph
        net = self.local_net
        avg_net = self.Estimator.average_net

        # To feeddict
        feed_dict = {
            net.r: rollout.reward,
            net.a: rollout.action,
            net.done: rollout.done[-1],
            net.seq_length: rollout.seq_length,
            avg_net.seq_length: rollout.seq_length,
        }

        feed_dict.update({net.state[k]:     v for k, v in rollout.states.iteritems()})
        feed_dict.update({avg_net.state[k]: v for k, v in rollout.states.iteritems()})
        feed_dict.update({net.pi_behavior.stats[k]: v for k, v in rollout.pi_stats.iteritems()})

        loss, summaries, _, debug, self.gstep = net.update(self.step_op, feed_dict, self.sess)
        loss = AttrDict(loss)

        if display and self.name == "worker_0":
            tf.logging.info((
                "#{:6d}: pi_loss = {:+8.3f}, vf_loss = {:+8.3f}, ent = {:5.3f},"
                "loss = {:+10.3f} {}\33[0m S = {:3d}, B = {} norm = {:7.2e}"
            ).format(
                self.gstep, loss.pi, loss.vf, loss.entropy, loss.total,
                "\33[92m[on ]" if on_policy else "\33[93m[off]",
                rollout.seq_length, rollout.batch_size, loss.global_norm
            ))

        # Write summaries
        if self.summary_writer is not None:
            self.summary_writer.add_summary(summaries, self.gstep)
            self.summary_writer.flush()

        # Show learning rate every FLAGS.decay_steps
        if self.gstep % (FLAGS.decay_steps / 10) == 0:
            tf.logging.info("learning rate = {}".format(
                self.sess.run(self.local_net.lr)))

        if FLAGS.debug_dump and self.gstep > 1000 and self.gstep % 100 == 0:
            import scipy.io
            scipy.io.savemat("{}/Q_{}.mat".format(FLAGS.debug_dir, self.gstep), debug)
