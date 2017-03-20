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

        net = self.local_net
        self.step_op = [
            {
                'total': net.loss,
                'pi_loss': net.pi_loss,
                'vf_loss': net.vf_loss,
            },
            net.summaries,
            self.train_op,
            self.inc_global_step
        ]

    def _run(self):
        self.copy_params_from_global()

        # FLAGS.max_steps = int(np.ceil(FLAGS.t_max * FLAGS.command_freq))
        rollout = self.run_n_steps(FLAGS.max_steps)

        # Even though A3C can't use experience replay, we still need store
        # experiences for playback visualization and statistics
        self.store_experience(rollout)

        # Update the global networks
        self.update(rollout)

    def get_rewards_and_returns(self, transitions):
        # If an episode ends, the return is 0. If not, we estimate return
        # by bootstrapping the value from the last state (using value net)
        last = transitions[-1]
        mdp_state = form_mdp_state(self.env, last.next_state, last.action, last.reward)
        v = self.local_net.predict_values(mdp_state, self.sess)
        v[last.done] = 0

        # Collect rewards from transitions, append v to rewards, and compute the total discounted returns
        rewards = [t.reward.T[None, ...] for t in transitions]
        rewards_plus_v = np.concatenate(rewards + [v], axis=0)
        rewards = rewards_plus_v[:-1, ...]
        returns = discount(rewards_plus_v, self.discount_factor)[:-1, :]
        return v, rewards, returns

    def get_values_and_td_targets(self, mdp_states, v, rewards):
        values = self.local_net.predict_values(mdp_states, self.sess)
        values = np.concatenate([values, v], axis=0)

        delta_t = rewards + self.discount_factor * values[1:, ...] - values[:-1, ...]
        values = values[:-1, ...]
        return values, delta_t

    def parse_transitions(self, transitions):
        mdp_states = AttrDict({
            key: np.concatenate([
                t.mdp_state[key][None, ...] for t in transitions
            ], axis=0)
            for key in transitions[0].mdp_state.keys()
        })

        v, rewards, returns = self.get_rewards_and_returns(transitions)
        values, delta_t = self.get_values_and_td_targets(mdp_states, v, rewards)

        actions = np.concatenate([
            trans.action.T[None, ...] for trans in transitions
        ], axis=0)

        advantages = discount(delta_t, self.discount_factor * FLAGS.lambda_)

        return len(transitions), mdp_states, v, rewards, returns, values, delta_t, actions, advantages

    # @timer_decorate
    def update(self, rollout):

        rollout = self.get_partial_rollout(rollout, FLAGS.max_seq_length)

        """
        print "rollout.keys = {}".format(rollout.keys())
        for key in rollout.states:
            print "rollout.states[{}].shape = {}".format(key, rollout.states[key].shape)
        """

        # Compute values and also bootstrap from last state
        net = self.local_net
        values = net.predict_values(rollout.states, self.sess)

        # Compute discounted total returns
        rewards = np.concatenate([rollout.reward, values[-1:]])
        rewards[-1, rollout.done[-1]] = 0
        # print "rewards.shape = {}".format(rewards.shape)

        returns = discount(rewards, self.discount_factor)[:-1, :]
        # print "returns.shape = {}".format(returns.shape)

        # Compute TD target
        delta_t = rewards[:-1] + self.discount_factor * values[1:] - values[:-1]
        values = values[:-1]

        # print "delta_t.shape = {}".format(delta_t.shape)
        # print "values.shape = {}".format(values.shape)

        # Use discounted TD target as advantages (GAE)
        advantages = discount(delta_t, self.discount_factor * FLAGS.lambda_)
        # print "advantages.shape = {}".format(advantages.shape)

        feed_dict = {
            net.advantages: advantages,
            net.returns: returns,
            net.actions_ext: rollout.action,
        }
        feed_dict.update({net.state[k]: v[:-1] for k, v in rollout.states.iteritems()})

        loss, summaries, _, self.g_step = net.predict(self.step_op, feed_dict, self.sess)
        loss = AttrDict(loss)

        print "#{:04d}: loss = {}".format(self.g_step, loss)
