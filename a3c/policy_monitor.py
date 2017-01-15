import gym
import sys
import os
import itertools
import collections
import numpy as np
import tensorflow as tf
import time

from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
    sys.path.append(import_path)

from lib.atari.state_processor import StateProcessor
from lib.atari import helpers as atari_helpers
from estimators import ValueEstimator, PolicyEstimator
from worker import make_copy_params_op, form_mdp_state

class PolicyMonitor(object):
    """
    Helps evaluating a policy by running an episode in an environment,
    saving a video, and plotting summaries to Tensorboard.

    Args:
    env: environment to run in
    policy_net: A policy estimator
    summary_writer: a tf.train.SummaryWriter used to write Tensorboard summaries
    """
    def __init__(self, env, policy_net, summary_writer, saver=None):
        self.env = env
        self.global_policy_net = policy_net
        self.summary_writer = summary_writer
        self.saver = saver
        self.sp = StateProcessor()

        self.video_dir = os.path.join(summary_writer.get_logdir(), "../videos")
        self.video_dir = os.path.abspath(self.video_dir)
        self.checkpoint_path = os.path.abspath(os.path.join(summary_writer.get_logdir(), "../checkpoints/model"))

        try:
            os.makedirs(self.video_dir)
        except:
            pass

        # Local policy net
        with tf.variable_scope("policy_eval"):
            self.policy_net = PolicyEstimator()

        # Op to copy params from global policy/value net parameters
        self.copy_params_op = make_copy_params_op(
            tf.contrib.slim.get_variables(scope="global", collection=tf.GraphKeys.TRAINABLE_VARIABLES),
            tf.contrib.slim.get_variables(scope="policy_eval", collection=tf.GraphKeys.TRAINABLE_VARIABLES))

        # Enable monitoring
        # self.env.monitor.start(directory=self.video_dir, video_callable=lambda x: True, resume=True)

    def reset_env(self):
        # self.state = np.array([+7, 10, 0, 0, 20, 1.5], dtype=np.float32).reshape(6, 1)
        self.state = np.array([+9, 1, 0, 0, 20, 0], dtype=np.float32).reshape(6, 1)
        self.action = np.array([0, 0], dtype=np.float32).reshape(2, 1)
        self.env._reset(self.state)

    def eval_once(self, sess):
        with sess.as_default(), sess.graph.as_default():
            # Copy params to local model
            global_step, _ = sess.run([tf.contrib.framework.get_global_step(), self.copy_params_op])

        # Run an episode
        done = False
        self.reset_env()
        total_reward = 0.0
        episode_length = 0
        reward = 0

        while not done:
            mdp_state = form_mdp_state(self.env, self.state, self.action, reward)
            action = self.policy_net.predict(mdp_state, sess).reshape(2, -1)
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            episode_length += 1
            state = next_state

            if episode_length % 1000 == 0:
                print "episode_length = {}, total_reward = {}".format(episode_length, total_reward)

            if episode_length > 100000:
                break

        # Add summaries
        episode_summary = tf.Summary()
        episode_summary.value.add(simple_value=total_reward, tag="eval/total_reward")
        episode_summary.value.add(simple_value=episode_length, tag="eval/episode_length")
        self.summary_writer.add_summary(episode_summary, global_step)
        self.summary_writer.flush()

        if self.saver is not None:
            self.saver.save(sess, self.checkpoint_path)

        tf.logging.info("Eval results at step {}: total_reward {}, episode_length {}".format(global_step, total_reward, episode_length))

        return total_reward, episode_length

    def continuous_eval(self, eval_every, sess, coord):
        """
        Continuously evaluates the policy every [eval_every] seconds.
        """
        try:
            while not coord.should_stop():
                self.eval_once(sess)
                # Sleep until next evaluation cycle
                time.sleep(eval_every)
        except tf.errors.CancelledError:
            return
