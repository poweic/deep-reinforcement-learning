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
# from time import time
import time

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

def flatten(x): # flatten the first 2 axes
    return x.reshape((-1,) + x.shape[2:])

def deflatten(x, n): # de-flatten the first axes
    return x.reshape((n, -1,) + x.shape[1:])

Transition = collections.namedtuple("Transition", ["mdp_state", "state", "action", "reward", "next_state", "done"])
def form_mdp_state(env, state, prev_action, prev_reward):
    return {
        "front_view": env.get_front_view(state).copy(),
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
    local_grads, _ = tf.clip_by_global_norm(local_grads, max_grad)

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
    def __init__(self, name, env, global_counter, add_summaries, n_agents=1, discount_factor=0.99, max_global_steps=None):
        self.name = name
        self.discount_factor = discount_factor
        self.max_global_steps = max_global_steps
        self.global_step = tf.contrib.framework.get_global_step()
        self.global_counter = global_counter
        self.local_counter = itertools.count()
        self.n_agents = n_agents
        self.env = env
        self.counter = 0

        self.summary_writer = None

        # Create local policy/value nets that are not updated asynchronously
        with tf.variable_scope(name):
            self.local_net = PolicyValueEstimator(add_summaries)

        self.state = None
        self.action = None
        self.max_return = 0

    def set_global_net(self, global_net):
        # Operation to copy params from global net to local net
        global_vars = tf.contrib.slim.get_variables(scope="global", collection=tf.GraphKeys.TRAINABLE_VARIABLES)
        local_vars = tf.contrib.slim.get_variables(scope=self.name, collection=tf.GraphKeys.TRAINABLE_VARIABLES)
        self.copy_params_op = make_copy_params_op(global_vars, local_vars)

        self.global_net = global_net
        self.net_train_op = make_train_op(self.local_net, self.global_net)

    def run(self, sess, coord, t_max):

        self.sess = sess

        with sess.as_default(), sess.graph.as_default():

            try:
                while not coord.should_stop():
                    # Copy Parameters from the global networks
                    sess.run(self.copy_params_op)

                    # Collect 1 episodes of experience (multiple agents)
                    transitions, local_t, global_t = self.run_n_steps(t_max)

                    if self.max_global_steps is not None and global_t >= self.max_global_steps:
                        tf.logging.info("Reached global step {}. Stopping.".format(global_t))
                        coord.request_stop()
                        return

                    # Update the global networks
                    self.update(transitions)
            
            except tf.errors.CancelledError:
                return
            except:
                traceback.print_exc()

    def reset_env(self):
        self.state = np.array([0.5, 3, 0, 0, 0.001, 0])

        '''
        # self.state = np.array([+7.4, 8.15, 0, 0, 0, 0])
        theta = np.random.rand(self.n_agents) * np.pi * 2
        phi = np.random.rand(self.n_agents) * np.pi * 2
        self.state = np.zeros((6, self.n_agents))
        self.state[0] += 6 * np.cos(theta)
        self.state[1] += 10 + 6 * np.sin(theta)
        self.state[2] += phi
        '''

        self.action = np.array([0, 0])

        # Reshape to compatiable format
        self.state = self.state.astype(np.float32).reshape(6, -1)
        self.action = np.zeros((2, self.n_agents), dtype=np.float32)
        self.total_return = np.zeros((1, self.n_agents))
        self.current_reward = np.zeros((1, self.n_agents))

        # Add some noise to have diverse start points
        noise = np.random.rand(6, self.n_agents).astype(np.float32) * 0.05
        self.state = self.state + noise

        self.env._reset(self.state)

    def run_n_steps(self, n):

        transitions = []

        # Initial state
        self.reset_env()

        reward = np.zeros((1, self.n_agents), dtype=np.float32)
        for i in range(n):

            mdp_state = form_mdp_state(self.env, self.state, self.action, reward)

            # Predict an action
            self.action = self.local_net.predict_actions(mdp_state, self.sess).T
            '''
            if i == 1:
                mean_yaw = np.mean(self.action[1, :])
                steer = np.arctan(2.0 * self.action[1, :] / self.state[4, :]) / np.pi * 180
                print "mean(steer) = \33[33m{}\33[0m deg".format(np.mean(steer))
            '''
            assert not np.any(np.isnan(self.action)), "i = {}, self.action = {}, mdp_state = {}".format(i, self.action, mdp_state)

            # Take several steps in environment
            n_steps = int(1. / FLAGS.command_freq / FLAGS.timestep)
            for j in range(n_steps):
                next_state, reward, done, _ = self.env.step(self.action)

            self.current_reward = reward
            self.total_return += reward
            if np.max(self.total_return) > self.max_return:
                self.max_return = np.max(self.total_return)

            # Store transition
            # Down-sample transition to reduce correlation between samples
            transitions.append(Transition(
                mdp_state=mdp_state,
                state=self.state.copy(),
                action=self.action.copy(),
                next_state=next_state.copy(),
                reward=reward.copy(),
                done=done.copy()
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

        return transitions, local_t, global_t

    def get_rewards_and_returns(self, transitions):
        # If an episode ends, the return is 0. If not, we estimate return
        # by bootstrapping the value from the last state (using value net)
        last = transitions[-1]
        mdp_state = form_mdp_state(self.env, last.next_state, last.action, last.reward)
        v = self.local_net.predict_values(mdp_state, self.sess)
        v[last.done] = 0

        # Collect rewards from transitions, append v to rewards, and compute
        # the total discounted returns
        rewards = [t.reward.T for t in transitions]
        rewards_plus_v = np.concatenate(rewards + [v], axis=1)
        rewards = rewards_plus_v[:, :-1]
        returns = discount(rewards_plus_v, self.discount_factor)[:, :-1]

        '''
        print "v.shape = {}, rewards.shape = {}, returns.shape = {}".format(
            v.shape, rewards.shape, returns.shape)
        '''

        return v, rewards, returns

    def get_values_and_td_targets(self, mdp_states, v, rewards):
        T = rewards.shape[1]

        values = self.local_net.predict_values(mdp_states, self.sess)
        values = np.concatenate([values.reshape(-1, T), v], axis=1)

        delta_t = rewards + self.discount_factor * values[:, 1:] - values[:, :-1]

        values = values[:, :-1]

        '''
        print "values.shape = {}, delta_t.shape = {}".format(
            values.shape, delta_t.shape)
        '''

        return values, delta_t

    def get_mdp_states(self, transitions):
        return {
            key: flatten(np.concatenate([
                trans.mdp_state[key][:, None, :] for trans in transitions
            ], axis=1))
            for key in transitions[0].mdp_state.keys()
        }

    # @timer_decorate
    def update(self, transitions):
        """
        Updates global policy and value networks based on collected experience

        Args:
          transitions: A list of experience transitions
          sess: A Tensorflow session
        """

        # T is the number of transitions
        T = len(transitions)

        mdp_states = self.get_mdp_states(transitions)

        v, rewards, returns = self.get_rewards_and_returns(transitions)

        values, delta_t = self.get_values_and_td_targets(mdp_states, v, rewards)

        # advantages = rewards
        # advantages = delta_t
        # prev_rewards = mdp_states["prev_reward"].reshape(self.n_agents, -1)
        # advantages = rewards - prev_rewards
        advantages = discount(delta_t, self.discount_factor * FLAGS.lambda_)
        # print "advantages.shape = ", advantages.shape

        '''
        print "np.mean(advantages, axis=1) = {}".format(np.mean(advantages, axis=1))
        adv = data['advantages'][:, 0]
        print "advantages[:, 0] = {}".format(adv)
        print "min(adv) = {}, max(adv) = {}, mean(adv) = {}".format(np.min(adv), np.max(adv), np.mean(adv))
        '''

        actions = np.concatenate([
            trans.action.T[:, None, :] for trans in transitions
        ], axis=1)
        # print "actions.shape = ", actions.shape

        # Add TD Target, TD Error, and actions to data
        data = dict(
            returns = returns[..., None],
            advantages = advantages[..., None],
            actions = actions,
        )
        data.update({k: deflatten(v, self.n_agents) for k, v in mdp_states.viewitems()})

        # Downsample experiences
        start_idx = np.random.randint(FLAGS.downsample)
        s = slice(start_idx, None, FLAGS.downsample)
        for k in data.keys():
            data[k] = flatten(data[k][:, s, ...])
            # print "data[{}].shape = {}".format(k, data[k].shape)

        # The 1st dimension is batch_size, get batch_size after down-sampling
        batch_size = len(data[k])
        for k in data.keys():
            assert batch_size == len(data[k])

        feed_dict = {
            self.local_net.returns: data['returns'],
            self.local_net.advantages: data['advantages'],
            self.local_net.actions_ext: data['actions'],
            self.local_net.state["front_view"]: data['front_view'],
            self.local_net.state["vehicle_state"]: data['vehicle_state'],
            self.local_net.state["prev_action"]: data['prev_action'],
            self.local_net.state["prev_reward"]: data['prev_reward']
        }

        ops = [
            self.global_step,
            self.local_net.loss,
            self.local_net.pi_loss,
            self.local_net.vf_loss,
            self.net_train_op
        ]

        # Train the global estimators using local gradients
        if self.summary_writer is None:
            global_step, loss, pi_loss, vf_loss, _ = self.sess.run(ops, feed_dict)
        else:
            ops += [self.local_net.summaries]
            global_step, loss, pi_loss, vf_loss, _, summaries = self.sess.run(ops, feed_dict)

        print "\33[33m#{:04d}\33[0m update (from {}), returns = {:.2f}, batch_size = {},".format(
            global_step, self.name, np.mean(returns[:, 0]), batch_size),

        print "pi_loss = {:+.5f}, vf_loss = {:+.5f}, total_loss = {:+.5f}".format(
            pi_loss, vf_loss, loss)

        # =================== DEBUG ===================
        '''
        worker_id = int(self.name[-1])
        if worker_id == 0: # and self.counter % 10 == 0:
            fn = "debug/{:04d}.mat".format(self.counter)

            for k in data.keys():
                data[k] = deflatten(data[k], self.n_agents)

            data.update(dict(
                mu = mu.reshape(-1, T, 2)[:, s, ...],
                sigma = sigma.reshape(-1, T, 2)[:, s, ...],
                values = values[:, s],
                # log_prob = deflatten(log_prob, self.n_agents),
                # g_mu = deflatten(g_mu, self.n_agents)
            ))

            # for k in data.keys():
            #     print "data[{}].shape = {}".format(k, data[k].shape)

            scipy.io.savemat(fn, data)
            print "{} saved.".format(fn)
            cv2.imwrite("disp_img.png", self.env.to_disp)
        '''
        # =================== DEBUG ===================

        # Write summaries
        if self.summary_writer is not None:
            self.summary_writer.add_summary(summaries, global_step)
            self.summary_writer.flush()

        assert pi_loss == pi_loss, "pi_loss = {}, vf_loss = {}".format(pi_loss, vf_loss)

        self.counter += 1
