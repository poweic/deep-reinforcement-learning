import gym
import sys
import os
import cv2
import itertools
import collections
import numpy as np
import tensorflow as tf
import scipy.io

max_return = 0

from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
    sys.path.append(import_path)

from estimators import ValueEstimator, PolicyEstimator

Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
def get_map_with_vehicle(env, state):
    # Rotate the map according the vehicle orientation in degree (not radian)
    img = env.get_front_view(state)
    return img.reshape(1, 20, 20, 1)

def form_mdp_state(env, state, prev_action, prev_reward):

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
    # local_grads, _ = tf.clip_by_global_norm(local_grads, 5.0)
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
    def __init__(self, name, rewards, env, policy_net, value_net, global_counter, discount_factor=0.99, summary_writer=None, max_global_steps=None):
        self.name = name
        self.discount_factor = discount_factor
        self.max_global_steps = max_global_steps
        self.rewards = rewards
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

    def run(self, sess, coord, t_max):

        with sess.as_default(), sess.graph.as_default():

            try:
                while not coord.should_stop():
                    # Copy Parameters from the global networks
                    sess.run(self.copy_params_op)

                    # Initial state
                    self.reset_env()

                    # Collect some experience
                    transitions, local_t, global_t = self.run_n_steps(t_max, sess)

                    if self.max_global_steps is not None and global_t >= self.max_global_steps:
                        tf.logging.info("Reached global step {}. Stopping.".format(global_t))
                        coord.request_stop()
                        return

                    # Update the global networks
                    self.update(transitions, sess)
            
            except tf.errors.CancelledError:
                return

    def reset_env(self):
        # FLAGS = tf.flags.FLAGS
        # self.state = np.array([+7, 10, 0, 0, 2.0, 1.0])
        self.state = np.array([+7.5, 7.1, 0, 0, 2.0, 1.0])
        # theta = np.random.rand() * np.pi * 2
        # self.state = np.array([6 * np.cos(theta), 10 + 6 * np.sin(theta), theta, 0, 0, 1.0])
        # self.state = np.array([+9, 1, 0, 0, 2.0, 0])
        self.action = np.array([0, 0])

        # Reshape to compatiable format
        self.state = self.state.astype(np.float32).reshape(6, 1)
        self.action = self.action.astype(np.float32).reshape(2, 1)

        self.env._reset(self.state)

    def run_n_steps(self, n, sess):

        global max_return

        # print "{} started a new episode".format(self.name)
        transitions = []
        reward = 0
        total_return = 0
        for i in range(n):

            mdp_state = form_mdp_state(self.env, self.state, self.action, reward)
            self.action = self.policy_net.predict(mdp_state, sess).reshape(2, -1)
            self.action[0, 0] = 2
            self.action[1, 0] = np.pi / 11.2

            # Take a step
            next_state, reward, done, _ = self.env.step(self.action)
            total_return += reward
            if total_return > max_return:
                max_return = total_return
                print "max_return = \33[93m{}\33[0m (step #{:05d}, action = {})".format(max_return, i, self.action.flatten())

            # Store transition
            transitions.append(Transition(
                state=self.state, action=self.action, reward=reward, next_state=next_state, done=done))

            # Increase local and global counters
            local_t = next(self.local_counter)
            global_t = next(self.global_counter)

            if local_t % 100 == 0:
                tf.logging.info("{}: local Step {}, global step {}".format(self.name, local_t, global_t))

            if done:
                break
            else:
                self.state = next_state

        # print "{} steps in episode".format(i)

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
            # print "mdp_state['prev_action'].shape = {}".format(mdp_state["prev_action"].shape)
            reward = self.value_net.predict(mdp_state, sess)

        # Accumulate minibatch exmaples
        policy_targets = []
        value_targets = []
        actions = []

        mdp_states = []

        T = len(transitions)
        for t in range(T)[::-1]:
            transition = transitions[t]

            # Get previous action and reward: a_{t-1}, r_{t-1}
            a_tm1 = transitions[t-1].action if t > 0 else np.zeros((2, 1))
            r_tm1 = transitions[t-1].reward if t > 0 else 0
            mdp_state = form_mdp_state(self.env, transition.state, a_tm1, r_tm1)

            # Get td_target (reward) and td_error
            reward = transition.reward + self.discount_factor * reward
            value = self.value_net.predict(mdp_state, sess)
            td_error = reward - value
            # print "step #{:04d}: reward = {}, value = {}, td_error = {}".format(t, reward, value, td_error)
            # print "{} {} {}".format(reward, value, td_error)

            # Accumulate updates
            mdp_states.append(mdp_state)

            actions.append(transition.action)
            policy_targets.append(td_error)
            value_targets.append(reward)

        '''
        scipy.io.savemat("feed_dict/{}.mat".format(self.counter), dict(
            mdp_states=mdp_states,
            actions=actions,
            policy_targets=policy_targets,
            value_targets=value_targets
        ))
        self.counter += 1
        '''

        # Turn list of dictionaries to dictionary of lists (numpy array)
        mdp_states = {
            key: np.squeeze(np.array([s[key] for s in mdp_states]), axis=1)
            for key in mdp_state.keys()
        }

        # Add TD Target, TD Error, and action to feed_dict
        feed_dict = {
            self.policy_net.target: np.array(policy_targets).reshape((-1, 1)),
            self.policy_net.action: np.array(actions).reshape((-1, 2)),
            self.value_net.target: np.array(value_targets).reshape((-1, 1)),
        }

        for k in mdp_states.keys():
            feed_dict[self.policy_net.state[k]] = mdp_states[k]

        '''
        for k, v in feed_dict.iteritems():
            print k.name, v.shape
        '''

        # Train the global estimators using local gradients
        print "Updates from {}, reward = {}, # of steps = {}".format(self.name, reward, T)
        global_step, pnet_loss, vnet_loss, _, _, pnet_summaries, vnet_summaries = sess.run([
            self.global_step,
            self.policy_net.loss,
            self.value_net.loss,
            self.pnet_train_op,
            self.vnet_train_op,
            self.policy_net.summaries,
            self.value_net.summaries
        ], feed_dict)
        print "pnet_loss = {:.7e}, vnet_loss = {:.7e}".format(pnet_loss, vnet_loss)

        # Write summaries
        if self.summary_writer is not None:
            self.summary_writer.add_summary(pnet_summaries, global_step)
            self.summary_writer.add_summary(vnet_summaries, global_step)
            self.summary_writer.flush()

        assert pnet_loss == pnet_loss, "pnet_loss = {}, vnet_loss = {}".format(pnet_loss, vnet_loss)

        return pnet_loss, vnet_loss, pnet_summaries, vnet_summaries
