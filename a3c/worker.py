import gym
import sys
import os
import cv2
import itertools
import collections
import numpy as np
import tensorflow as tf
import scipy.io
import scipy.signal
import traceback
from time import time

from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
    sys.path.append(import_path)

from estimators import PolicyValueEstimator

FLAGS = tf.flags.FLAGS

def discount(x, gamma):
    if x.ndim == 1:
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
    else:
        return scipy.signal.lfilter([1], [1, -gamma], x[:, ::-1], axis=1)[:, ::-1]

def timer_decorate(func):
    def func_wrapper(*args, **kwargs):
        timer = time()
        result = func(*args, **kwargs)
        print "\33[2m{} took {:.4f} seconds\33[0m".format(func.__name__, time() - timer)
        return result
    return func_wrapper

Transition = collections.namedtuple("Transition", ["mdp_state", "state", "action", "reward", "next_state", "done"])
def form_mdp_state(env, state, prev_action, prev_reward):
    return {
        "map_with_vehicle": env.get_front_view(state),
        "vehicle_state": state.copy().T,
        "prev_action": prev_action.copy().T,
        "prev_reward": prev_reward.copy().T
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
    Creates an op that applies local gradients to the global variables.
    """
    # Get local gradients and global variables
    local_grads, _ = zip(*local_estimator.grads_and_vars)
    _, global_vars = zip(*global_estimator.grads_and_vars)

    # Clip gradients
    max_grad = FLAGS.max_gradient
    # local_grads, _ = tf.clip_by_global_norm(local_grads, max_grad)

    # Zip clipped local grads with global variables
    local_grads_global_vars = list(zip(local_grads, global_vars))
    global_step = tf.contrib.framework.get_global_step()

    return global_estimator.optimizer.apply_gradients(
        local_grads_global_vars, global_step=global_step)

class Worker(object):
    """
    An A3C worker thread. Runs episodes locally and updates global shared value and policy nets.

    Args:
    name: A unique name for this worker
    env: The Gym environment used by this worker
    global_net: Instance of the globally shared network
    global_counter: Iterator that holds the global step
    discount_factor: Reward discount factor
    summary_writer: A tf.train.SummaryWriter for Tensorboard summaries
    max_global_steps: If set, stop coordinator when global_counter > max_global_steps
    """
    def __init__(self, name, env, global_net, global_counter, n_agents=1, discount_factor=0.99, summary_writer=None, max_global_steps=None):
        self.name = name
        self.discount_factor = discount_factor
        self.max_global_steps = max_global_steps
        self.global_step = tf.contrib.framework.get_global_step()
        self.global_net = global_net
        self.global_counter = global_counter
        self.local_counter = itertools.count()
        self.summary_writer = summary_writer
        self.n_agents = n_agents
        self.env = env
        self.counter = 0

        # Create local policy/value nets that are not updated asynchronously
        with tf.variable_scope(name):
            self.local_net = PolicyValueEstimator()

        # Operation to copy params from global net to local net
        global_vars = tf.contrib.slim.get_variables(scope="global", collection=tf.GraphKeys.TRAINABLE_VARIABLES)
        local_vars = tf.contrib.slim.get_variables(scope=self.name, collection=tf.GraphKeys.TRAINABLE_VARIABLES)
        self.copy_params_op = make_copy_params_op(global_vars, local_vars)

        # Operation to 
        self.net_train_op = make_train_op(self.local_net, self.global_net)

        self.state = None
        self.action = None
        self.max_return = 0

    def run(self, sess, coord, t_max):

        self.sess = sess

        with sess.as_default(), sess.graph.as_default():

            try:
                while not coord.should_stop():
                    # Copy Parameters from the global networks
                    sess.run(self.copy_params_op)

                    # Collect 10 episodes of experience
                    transitions = []
                    # for i in range(FLAGS.n_episodes):
                    local_t, global_t = self.run_n_steps(t_max, transitions)

                    if self.max_global_steps is not None and global_t >= self.max_global_steps:
                        tf.logging.info("Reached global step {}. Stopping.".format(global_t))
                        coord.request_stop()
                        return

                    # Update the global networks
                    self.update(transitions)

                print "\33[91mEnd of for loop\33[0m"
            
            except tf.errors.CancelledError:
                return
            except:
                traceback.print_exc()

    def reset_env(self):
        # self.state = np.array([0, 0, 0, 0, 0, 0])
        self.state = np.array([+6.1, 9.15, 0, 0, 0, 0])
        '''
        theta = np.random.rand() * np.pi * 2
        phi = np.random.rand() * np.pi * 2
        self.state = np.array([6 * np.cos(theta), 10 + 6 * np.sin(theta), phi, 0, 0, 0])
        '''
        # self.state = np.array([+9, 1, 0, 0, 2.0, 0])
        self.action = np.array([0, 0])

        # Reshape to compatiable format
        self.state = self.state.astype(np.float32).reshape(6, 1)
        self.action = np.zeros((2, self.n_agents), dtype=np.float32)
        self.total_return = np.zeros((1, self.n_agents))
        self.current_reward = np.zeros((1, self.n_agents))

        # Add some noise to have diverse start points
        noise = np.random.rand(6, self.n_agents).astype(np.float32) * 0.01
        self.state = self.state + noise

        self.env._reset(self.state)

    # @timer_decorate
    def run_n_steps(self, n, transitions):

        timer = [0, 0, 0]

        # Initial state
        self.reset_env()

        reward = np.zeros((1, self.n_agents), dtype=np.float32)
        for i in range(n):

            timer[0] -= time()
            mdp_state = form_mdp_state(self.env, self.state, self.action, reward)
            timer[0] += time()

            timer[1] -= time()
            self.action = self.local_net.predict_actions(mdp_state, self.sess).T
            timer[1] += time()
            assert not np.any(np.isnan(self.action)), "i = {}, self.action = {}, mdp_state = {}".format(i, self.action, mdp_state)
            '''
            if self.counter < 1000:
                self.action[0, 0] = 2
                self.action[1, 0] = -np.pi / 11.2
            '''

            timer[2] -= time()
            # Take a step
            next_state, reward, done, _ = self.env.step(self.action)
            self.current_reward = reward
            self.total_return += reward
            if np.max(self.total_return) > self.max_return:
                self.max_return = np.max(self.total_return)
                '''
                print "{}'s max_return = {} (step #{:05d}, action = {})".format(
                    self.name, self.max_return, i, self.action.flatten())
                '''
            timer[2] += time()

            # Store transition
            # Down-sample transition to reduce correlation between samples
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

            if np.any(done):
                break
            else:
                self.state = next_state

        print "Took {}, {}, {}, {}, # of steps = {}".format(timer[0], timer[1], timer[2], timer[0] + timer[1] + timer[2], i)

        return local_t, global_t

    # @timer_decorate
    def update(self, transitions):
        """
        Updates global policy and value networks based on collected experience

        Args:
          transitions: A list of experience transitions
          sess: A Tensorflow session
        """

        # If we episode was not done we bootstrap the value from the last state
        last = transitions[-1]
        mdp_state = form_mdp_state(self.env, last.next_state, last.action, last.reward)
        reward = self.local_net.predict_values(mdp_state, self.sess)
        reward[last.done] = -100.

        rewards = np.concatenate([trans.reward.T for trans in transitions] + [reward], axis=1)
        returns = discount(rewards, self.discount_factor)[:, :-1]
        rewards = rewards[:, :-1]
        # print "returns.shape = ", returns.shape

        # Accumulate minibatch exmaples
        T = len(transitions)

        def flatten(x):
            # flatten first 2 axes
            return x.reshape((-1,) + x.shape[2:])

        mdp_states = {
            key: flatten(np.concatenate([
                transitions[t].mdp_state[key][:, None, :] for t in range(T)
            ], axis=1))
            for key in transitions[0].mdp_state.keys()
        }

        '''
        for key in mdp_states:
            print "mdp_states[{}].shape = {}".format(key, mdp_states[key].shape)
        '''

        lambda_ = 1.0

        values = self.local_net.predict_values(mdp_states, self.sess)
        values = np.concatenate([values.reshape(-1, T), reward], axis=1)

        delta_t = rewards + self.discount_factor * values[:, 1:] - values[:, :-1]
        # print "delta_t.shape = ", delta_t.shape

        advantages = discount(delta_t, self.discount_factor * lambda_)
        # print "advantages.shape = ", advantages.shape

        values = values[:, :-1]
        # print "values.shape = ", values.shape

        actions = flatten(np.concatenate([
            trans.action.T[:, None, :] for trans in transitions
        ], axis=1))
        # print "actions.shape = ", actions.shape

        '''
        values, mu, sigma, log_prob = self.local_net.predict_values(mdp_states, self.sess, debug=True)
        mu = mu.reshape((-1, 2))
        sigma = sigma.reshape((-1, 2))
        '''

        '''
        for t in range(T)[::-1]:
            # discounted reward
            reward = transitions[t].reward.T + self.discount_factor * reward

            # Get the advantage (td_error)
            value = values[:, t]
            td_error = reward - value

            value_targets[t] = reward
            policy_targets[t] = td_error
            actions[t, :] = transitions[t].action.T

            # print "step #{:04d}: reward = {}, value = {}, td_error = {}".format(t, reward, value, td_error)
            # print "{} {} {}".format(reward, value, td_error)
        '''

        '''
        worker_id = int(self.name[-1])
        if worker_id == 0 and self.counter % 10 == 0:
            fn = "debug/{:04d}.mat".format(self.counter)
            scipy.io.savemat(fn, dict(
                frontviews = mdp_states['map_with_vehicle'],
                vehicle_state = mdp_states['vehicle_state'],
                prev_action = mdp_states['prev_action'],
                prev_reward = mdp_states['prev_reward'],
                rewards = value_targets,
                values = values,
                actions = actions,
                adv = policy_targets,
                mu = mu,
                sigma = sigma,
                log_prob = log_prob
            ))
            print "{} saved.".format(fn)
        '''

        self.counter += 1

        # Add TD Target, TD Error, and actions to feed_dict
        feed_dict = {
            self.local_net.r: rewards.reshape(-1, 1),
            self.local_net.advantage: advantages.reshape(-1, 1),
            self.local_net.actions: actions,
        }

        for k in mdp_states.keys():
            feed_dict[self.local_net.state[k]] = mdp_states[k]

        for k in feed_dict.keys():
            feed_dict[k] = feed_dict[k][::FLAGS.downsample, ...]
            # print "feed_dict[{}].shape = {}".format(k, feed_dict[k].shape)

        # Train the global estimators using local gradients
        global_step, loss, pi_loss, vf_loss, summaries, _ = self.sess.run([
            self.global_step,
            self.local_net.loss,
            self.local_net.pi_loss,
            self.local_net.vf_loss,
            self.local_net.summaries,
            self.net_train_op,
        ], feed_dict)

        print "update from {}, returns = {:.2f}, batch_size = {:5d}, ".format(
            self.name, np.mean(returns[:, 0]), T * self.n_agents),
        print "pi_loss = {:+.5f}, vf_loss = {:+.5f}, total_loss = {:+.5f}".format(
            pi_loss, vf_loss, loss)

        # Write summaries
        if self.summary_writer is not None:
            self.summary_writer.add_summary(summaries, global_step)
            self.summary_writer.flush()

        assert pi_loss == pi_loss, "pi_loss = {}, vf_loss = {}".format(pi_loss, vf_loss)

        # return pnet_loss, vf_loss, summaries
