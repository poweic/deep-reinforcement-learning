import cv2
import numpy as np
import tensorflow as tf
import scipy.io
import traceback
import time
# from .estimators import *
import ac.a3c.estimators
# from estimators import A3CEstimator
# from monitor import server
from ac.worker import Worker
from ac.utils import *

FLAGS = tf.flags.FLAGS

class A3CWorker(Worker):
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
        self.Estimator = ac.a3c.estimators.A3CEstimator
        super(A3CWorker, self).__init__(**kwargs)

    def set_global_net(self, global_net):
        # Operation to copy params from global net to local net
        # FIXME change it to grads_and_vars[1]
        global_vars = tf.contrib.slim.get_variables(scope="global", collection=tf.GraphKeys.TRAINABLE_VARIABLES)
        local_vars = tf.contrib.slim.get_variables(scope=self.name, collection=tf.GraphKeys.TRAINABLE_VARIABLES)
        self.copy_params_op = make_copy_params_op(global_vars, local_vars)

        self.global_net = global_net
        self.train_op = make_train_op(self.local_net, self.global_net)
        self.inc_global_step = tf.assign_add(self.global_step, 1)

    def _run(self):
        # Copy Parameters from the global networks
        self.sess.run(self.copy_params_op)

        # Collect 1 episodes of experience (multiple agents)
        n = int(np.ceil(FLAGS.t_max * FLAGS.command_freq))
        transitions, local_t, global_t = self.run_n_steps(n)

        if self.max_global_steps is not None and global_t >= self.max_global_steps:
            tf.logging.info("Reached global step {}. Stopping.".format(global_t))
            self.coord.request_stop()
            return

        # Update the global networks
        self.update(transitions)

    def reset_env(self):
        # maze2
        self.state = np.array([0, 2, 0, 0, 0, 0])

        '''
        self.state = self.state.reshape((6, 1)) + np.zeros((6, self.n_agents))
        prob = (self.env.rewards - np.min(self.env.rewards)) * 20 + 1e-10
        N = int(self.n_agents * 0.3)
        x, y = inverse_transform_sampling_2d(prob, N)
        self.state[0, :N] = np.clip((x - 20) * 0.5, -5, 10)
        self.state[1, :N] = np.clip(20 - y * 0.5, 0, 20)
        self.state[2, :N] = np.random.rand(len(x)) * np.pi * 2

        near_top = self.state[1, :N] > 16
        self.state[2, near_top] = np.random.rand(np.sum(near_top)) * np.pi + np.pi / 2

        near_bottom = self.state[1, :N] < 4
        self.state[2, near_bottom] = np.random.rand(np.sum(near_bottom)) * (+np.pi) - np.pi / 2

        near_right = self.state[0, :N] > 6
        self.state[2, near_right] = np.random.rand(np.sum(near_right)) * (+np.pi)
        '''

        '''
        # line
        self.state = np.array([0.5, 2, 0, 0, 0.001, 0])

        # circle
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
        noise = np.random.randn(6, self.n_agents).astype(np.float32) * 0.5
        noise[2, :] /= 5

        self.state = self.state + noise

        self.env._reset(self.state)

    def run_n_steps(self, n_steps):

        transitions = []

        # Initial state
        self.reset_env()

        reward = np.zeros((1, self.n_agents), dtype=np.float32)
        for i in range(n_steps):

            mdp_state = form_mdp_state(self.env, self.state, self.action, reward)

            # Predict an action
            self.action = self.local_net.predict_actions(mdp_state, self.sess).T
            '''
            rand_mask = np.random.rand(self.n_agents) > 0.5
            rand_steer = ((np.random.rand(len(rand_mask.nonzero()[0])) * 60) - 30) * np.pi / 180
            self.action[1, rand_mask] = self.state[4:5, rand_mask] * np.tan(rand_steer).reshape(1, -1) / FLAGS.wheelbase
            '''
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
                # state=self.state.copy(),
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
        # print last.done
        # v[last.done] = 0
        v *= 0

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

    def parse_transitions(self, transitions):
        # T is the number of transitions
        T = len(transitions)

        mdp_states = self.get_mdp_states(transitions)

        v, rewards, returns = self.get_rewards_and_returns(transitions)

        values, delta_t = self.get_values_and_td_targets(mdp_states, v, rewards)

        actions = np.concatenate([
            trans.action.T[:, None, :] for trans in transitions
        ], axis=1)

        # advantages = rewards
        # advantages = returns
        advantages = delta_t
        advantages = discount(advantages, self.discount_factor * FLAGS.lambda_)

        return T, mdp_states, v, rewards, returns, values, delta_t, actions, advantages

    def downsample_data(self, data, T):
        start_idx = np.random.randint(min(FLAGS.downsample, T))
        s = slice(start_idx, None, FLAGS.downsample)
        for k in data.keys():
            # server.set_data(k, data[k])
            data[k] = flatten(data[k][:, s, ...])
            print "data[{}].shape = {}".format(k, data[k].shape)

        return data, s

    # @timer_decorate
    def update(self, transitions):
        """
        Updates global policy and value networks based on collected experience

        Args:
          transitions: A list of experience transitions
          sess: A Tensorflow session
        """

        # Parse transitions (rollouts) to get all MDP states, returns ..., etc.
        T, mdp_states, v, rewards, returns, values, delta_t, actions, advantages = self.parse_transitions(transitions)

        # Once we parse the data, we reshape
        data = self.prepare_data(mdp_states, returns, advantages, actions)

        # Downsample experiences
        data, s = self.downsample_data(data, T)

        # Get batch_size after down-sampling
        batch_size = self.get_batch_size(data)

        # Feed data to the network and perform updates
        global_step, results = self._update_(data, batch_size, FLAGS.debug)

        if FLAGS.debug:
            self._debug_(advantages, s, results)

        print "\33[33m#{:04d}\33[0m update (from {}), returns = {:+6.2f}, batch_size = {},".format(
            global_step, self.name, np.mean(returns[:, 0]), batch_size),

        print "pi_loss = {:+9.4f}, vf_loss = {:+9.4f}, total_loss = {:+9.4f}".format(
            results['pi_loss'], results['vf_loss'], results['loss'])

        # Make sure it's not NaN
        assert results['loss'] == results['loss'], "pi_loss = {}, vf_loss = {}".format(
            results['pi_loss'], results['vf_loss'])

        # Write summaries
        if self.summary_writer is not None:
            self.summary_writer.add_summary(results['summaries'][0], global_step)
            self.summary_writer.flush()

        self.counter += 1

    def prepare_data(self, mdp_states, returns, advantages, actions):

        # Add TD Target, TD Error, and actions to data
        data = dict(
            returns = returns[..., None],
            advantages = advantages[..., None],
            actions = actions,
        )

        data.update({k: deflatten(v, self.n_agents) for k, v in mdp_states.viewitems()})

        return data

    def get_batch_size(self, data):

        # The 1st dimension is batch_size
        batch_size = len(data[data.keys()[0]])

        # It cannot be zero, but let's just make sure it doesn't
        assert batch_size is not 0

        # Make sure all data has the same batch_size
        for k in data.keys():
            assert batch_size == len(data[k])

        print "batch_size = ", batch_size

        return batch_size

    def _update_(self, data, batch_size, debug=False):

        # Maximum size we can feed into a GPU (a conservative estimate) without
        # crashing (triggering stupid segfault bug in CuDNN)
        MAX_MINI_BATCH_SIZE = 128

        num_mini_batches = int(np.ceil(float(batch_size) / MAX_MINI_BATCH_SIZE))

        results = []
        for i in range(0, batch_size, MAX_MINI_BATCH_SIZE):
            s = slice(i, i + MAX_MINI_BATCH_SIZE)

            mini_batch_size, result = self._update_mini_batch(data, s, debug)

            for key in ['loss', 'pi_loss', 'vf_loss']:
                result[key] *= mini_batch_size

            results.append(result)

        def _reduce(x):
            try:
                np.concatenate(x, axis=0)
            except:
                return x

        results = {
            key: _reduce([result[key] for result in results])
            for key in results[0].keys()
        }

        for key in ['loss', 'pi_loss', 'vf_loss']:
            results[key] = np.sum(results[key]) / batch_size

        global_step = self.sess.run(self.inc_global_step)

        return global_step, results

    def _update_mini_batch(self, data, s, debug=False):
        net = self.local_net

        feed_dict = {
            net.returns: data['returns'][s, ...],
            net.advantages: data['advantages'][s, ...],
            net.actions_ext: data['actions'][s, ...],
            net.state["front_view"]: data['front_view'][s, ...],
            net.state["vehicle_state"]: data['vehicle_state'][s, ...],
            net.state["prev_action"]: data['prev_action'][s, ...],
            net.state["prev_reward"]: data['prev_reward'][s, ...]
        }

        ops = {
            'loss': net.loss,
            'pi_loss': net.pi_loss,
            'vf_loss': net.vf_loss,
        }

        if debug:
            ops.update({ 'mu': net.mu, 'sigma': net.sigma, 'g_mu': net.g_mu })

        if self.summary_writer is not None:
            ops.update({ 'summaries': net.summaries })

        results = self.sess.run([ops, self.train_op], feed_dict)[0]

        mini_batch_size = len(feed_dict[net.returns])

        return mini_batch_size, results

    def _debug_(self, advantages, s, results):

        # Get mu, sigma, and the gradient of mu from the results
        mu = results['mu']
        sigma = results['sigma']
        g_mu = results['g_mu']

        # Reshape s.t. dimension matches
        mu = mu.reshape(self.n_agents, -1, 2)[:, :, 1]
        sigma = sigma.reshape(self.n_agents, -1, 2)[:, :, 1]
        g_loss_wrt_mu_from_tf = g_mu[:, -1].reshape(self.n_agents, -1)
        g_gain_wrt_mu_from_tf = -g_loss_wrt_mu_from_tf
        adv = advantages[:, s]

        line_no = np.arange(0, self.n_agents).reshape(self.n_agents, 1)
        def add_line_no(x):
            return np.concatenate([line_no, x], axis=1)

        mu_and_adv = np.concatenate([mu, adv], axis=1)

        np.set_printoptions(precision=4, linewidth=500, suppress=True, formatter={
            'float_kind': lambda x: "#{:02d}".format(int(x)) if x.is_integer() else ("\33[92m" if x > 0 else "\33[91m") + ("{:9.4f}".format(x)) + "\33[0m"
        })

        print (" " * 32) + "mu" + (" " * 31) + "|" + (" " * 32) + "adv"
        print ("-" * 32) + "--" + ("-" * 31) + "." + ("-" * 66)
        print "{}\33[0m".format(add_line_no(mu_and_adv))
        print "==========================================================="
        print " {} (  mean   of mu_and_adv)".format(np.mean(mu_and_adv[:, 1:], axis=0))
        print " {} (variance of mu_and_adv)".format( np.std(mu_and_adv[:, 1:], axis=0))

        vf = mdp_states['vehicle_state'].reshape(self.n_agents, -1, 6)[:, s, 4]
        steer = np.arctan(FLAGS.wheelbase * actions[..., s, 1] / vf).astype(np.float)

        # print "steer.shape = {} [{}, {}], mu_and_adv.shape = {} [{}, {}]".format(steer.shape, steer.dtype, type(steer), mu_and_adv.shape, mu_and_adv.dtype, type(mu_and_adv))
        print "steer = \n{}".format(add_line_no(steer))

        g_gain_wrt_mu = ((steer - mu) * adv / (sigma ** 2)) / batch_size
        print "gradients of expected return w.r.t g_gain_wrt_mu = \n{}".format(add_line_no(g_gain_wrt_mu))
        print "==========================================================="
        print " {} (  mean   of g_gain_wrt_mu)".format(np.mean(g_gain_wrt_mu[:, 1:], axis=0))
        print " {} (variance of g_gain_wrt_mu)".format( np.std(g_gain_wrt_mu[:, 1:], axis=0))

