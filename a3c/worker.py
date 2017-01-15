import gym
import sys
import os
import cv2
import itertools
import collections
import numpy as np
import tensorflow as tf
import scipy.io
import traceback

from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
    sys.path.append(import_path)

from estimators import ValueEstimator, PolicyEstimator

Transition = collections.namedtuple("Transition", ["mdp_state", "state", "action", "reward", "next_state", "done"])
def get_map_with_vehicle(env, state):
    # Rotate the map according the vehicle orientation in degree (not radian)
    # print "\33[91mstate = {}, state.shape = {}\33[0m".format(state, state.shape)
    img = env.get_front_view(state)
    return img.reshape(1, 20, 20, 1)

def form_mdp_state(env, state, prev_action, prev_reward):

    state = state.copy()
    prev_action = prev_action.copy()

    map_with_vehicle = get_map_with_vehicle(env, state)

    return {
        "map_with_vehicle": map_with_vehicle,
        "vehicle_state": state.T,
        "prev_action": prev_action.T,
        "prev_reward": np.array(prev_reward, dtype=np.float32).reshape((-1, 1))
    }

def make_copy_params_op(v1_list, v2_list):
    """
    Creates an operation that copies parameters from variable in v1_list to variables in v2_list.
    The ordering of the variables in the lists must be identical.
    """
    v1_list = list(sorted(v1_list, key=lambda v: v.name))
    v2_list = list(sorted(v2_list, key=lambda v: v.name))

    update_ops = []
    for v1, v2 in zip(v1_list, v2_list):
        op = v2.assign(v1)
        update_ops.append(op)

    return update_ops

def make_train_op(local_estimator, global_estimator):
    """
    Creates an op that applies local estimator gradients
    to the global estimator.
    """
    local_grads, _ = zip(*local_estimator.grads_and_vars)

    # Clip gradients
    max_grad = tf.flags.FLAGS.max_gradient
    local_grads, _ = tf.clip_by_global_norm(local_grads, max_grad)
    _, global_vars = zip(*global_estimator.grads_and_vars)
    local_global_grads_and_vars = list(zip(local_grads, global_vars))
    return global_estimator.optimizer.apply_gradients(
        local_global_grads_and_vars, global_step=tf.contrib.framework.get_global_step())

class Worker(object):
    """
    An A3C worker thread. Runs episodes locally and updates global shared value and policy nets.

    Args:
    name: A unique name for this worker
    env: The Gym environment used by this worker
    policy_net: Instance of the globally shared policy net
    value_net: Instance of the globally shared value net
    global_counter: Iterator that holds the global step
    discount_factor: Reward discount factor
    summary_writer: A tf.train.SummaryWriter for Tensorboard summaries
    max_global_steps: If set, stop coordinator when global_counter > max_global_steps
    """
    def __init__(self, name, env, policy_net, value_net, global_counter, discount_factor=0.99, summary_writer=None, max_global_steps=None):
        print "In {}'s __init__ method".format(name)
        self.name = name
        self.discount_factor = discount_factor
        self.max_global_steps = max_global_steps
        self.global_step = tf.contrib.framework.get_global_step()
        self.global_policy_net = policy_net
        self.global_value_net = value_net
        self.global_counter = global_counter
        self.local_counter = itertools.count()
        self.summary_writer = summary_writer
        self.env = env
        self.counter = 0

        # Create local policy/value nets that are not updated asynchronously
        with tf.variable_scope(name):
            self.policy_net = PolicyEstimator()
            self.value_net = ValueEstimator(self.policy_net.state, reuse=True)

        # Op to copy params from global policy/valuenets
        self.copy_params_op = make_copy_params_op(
            tf.contrib.slim.get_variables(scope="global", collection=tf.GraphKeys.TRAINABLE_VARIABLES),
            tf.contrib.slim.get_variables(scope=self.name, collection=tf.GraphKeys.TRAINABLE_VARIABLES))

        self.vnet_train_op = make_train_op(self.value_net, self.global_value_net)
        self.pnet_train_op = make_train_op(self.policy_net, self.global_policy_net)

        self.state = None
        self.action = None
        self.max_return = 0
        self.total_return = 0
        self.current_reward = 0

    def run(self, sess, coord, t_max):

        print "In {}'s run method".format(self.name)

        with sess.as_default(), sess.graph.as_default():

            try:
                while not coord.should_stop():
                    # Copy Parameters from the global networks
                    sess.run(self.copy_params_op)

                    # Collect some experience
                    transitions, local_t, global_t = self.run_n_steps(t_max, sess)

                    if self.max_global_steps is not None and global_t >= self.max_global_steps:
                        tf.logging.info("Reached global step {}. Stopping.".format(global_t))
                        coord.request_stop()
                        return

                    # Update the global networks
                    self.update(transitions, sess)

                print "\33[91mEnd of for loop\33[0m"
            
            except tf.errors.CancelledError:
                return
            except:
                traceback.print_exc()

    def reset_env(self):
        # self.state = np.array([+7, 10, 0, 0, 2.0, 1.0])
        # self.state = np.array([+7.1, 7.1, 0, 0, 2.0, np.pi / 20])
        # self.state = np.array([+7 + np.random.rand(), 10 + np.random.rand(), 0, 0, 2, 0])
        # self.state = np.array([+6, 0, 0, 0, 0, 0])
        theta = np.random.rand() * np.pi * 2
        phi = np.random.rand() * np.pi * 2
        self.state = np.array([6 * np.cos(theta), 10 + 6 * np.sin(theta), phi, 0, 0, 1.0])
        # self.state = np.array([+9, 1, 0, 0, 2.0, 0])
        self.action = np.array([0, 0])

        # Reshape to compatiable format
        self.state = self.state.astype(np.float32).reshape(6, 1)
        self.action = self.action.astype(np.float32).reshape(2, 1)
        self.total_return = 0

        self.env._reset(self.state)

    def run_n_steps(self, n, sess):

        # Initial state
        self.reset_env()

        transitions = []
        reward = 0
        for i in range(n):

            mdp_state = form_mdp_state(self.env, self.state, self.action, reward)
            self.action = self.policy_net.predict(mdp_state, sess).reshape(2, -1)
            assert not np.any(np.isnan(self.action)), "self.action = {}, mdp_state = ".format(self.action, mdp_state)
            '''
            self.action[0, 0] = 2
            self.action[1, 0] = np.pi / 11.2
            '''

            # Take a step
            next_state, reward, done, _ = self.env.step(self.action)
            '''
            if reward < 0:
                done = True
                reward = -1000
            '''
            self.current_reward = reward
            self.total_return += reward
            if self.total_return > self.max_return:
                self.max_return = self.total_return
                '''
                print "{}'s max_return = {} (step #{:05d}, action = {})".format(
                    self.name, self.max_return, i, self.action.flatten())
                '''

            # Store transition
            transitions.append(Transition(
                mdp_state=mdp_state,
                state=self.state.copy(),
                action=self.action.copy(),
                next_state=next_state.copy(),
                reward=reward,
                done=done
            ))

            # Increase local and global counters
            local_t = next(self.local_counter)
            global_t = next(self.global_counter)

            if local_t % 100 == 0:
                tf.logging.info("{}: local Step {}, global step {}".format(self.name, local_t, global_t))

            if done:
                break
            else:
                self.state = next_state

        return transitions, local_t, global_t

    def update(self, transitions, sess):
        """
        Updates global policy and value networks based on collected experience

        Args:
          transitions: A list of experience transitions
          sess: A Tensorflow session
        """

        # If we episode was not done we bootstrap the value from the last state
        reward = 0.0
        last = transitions[-1]
        if not last.done:
            mdp_state = form_mdp_state(self.env, last.next_state, last.action, last.reward)
            reward = self.value_net.predict(mdp_state, sess)

        # Accumulate minibatch exmaples
        T = len(transitions)

        mdp_states = {
            key: np.squeeze(np.array([transitions[t].mdp_state[key] for t in range(T)]), axis=1)
            for key in transitions[0].mdp_state.keys()
        }

        value_targets = np.zeros((T, 1), dtype=np.float32)
        policy_targets = np.zeros((T, 1), dtype=np.float32)
        actions = np.zeros((T, 2), dtype=np.float32)
        values = self.value_net.predict(mdp_states, sess).squeeze()

        for t in range(T)[::-1]:
            # discounted reward
            reward = transitions[t].reward + self.discount_factor * reward

            # Get the advantage (td_error)
            value = values[t] # self.value_net.predict(mdp_state, sess)
            td_error = reward - value

            value_targets[t] = reward
            policy_targets[t] = td_error
            actions[t, :] = transitions[t].action.T

            # print "step #{:04d}: reward = {}, value = {}, td_error = {}".format(t, reward, value, td_error)
            # print "{} {} {}".format(reward, value, td_error)

        # Add TD Target, TD Error, and action to feed_dict
        feed_dict = {
            self.value_net.target: value_targets,
            self.policy_net.target: policy_targets,
            self.policy_net.action: actions,
        }

        for k in mdp_states.keys():
            feed_dict[self.policy_net.state[k]] = mdp_states[k]

        # Train the global estimators using local gradients
        global_step, pnet_loss, vnet_loss, _, _, pnet_summaries, vnet_summaries = sess.run([
            self.global_step,
            self.policy_net.loss,
            self.value_net.loss,
            self.pnet_train_op,
            self.vnet_train_op,
            self.policy_net.summaries,
            self.value_net.summaries
        ], feed_dict)
        print "Updates from {}, reward = {}, # of steps = {}, pnet_loss = {}, vnet_loss = {}, total_loss = {}".format(
            self.name, reward, T, pnet_loss, vnet_loss, pnet_loss + vnet_loss)

        # sys.exit(0)

        # Write summaries
        if self.summary_writer is not None:
            self.summary_writer.add_summary(pnet_summaries, global_step)
            self.summary_writer.add_summary(vnet_summaries, global_step)
            self.summary_writer.flush()

        assert pnet_loss == pnet_loss, "pnet_loss = {}, vnet_loss = {}".format(pnet_loss, vnet_loss)

        return pnet_loss, vnet_loss, pnet_summaries, vnet_summaries
