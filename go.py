#!/usr/bin/python
import sys
import cv2
import scipy.io
import itertools
import numpy as np
import tensorflow as tf
import collections

from gym_offroad_nav.envs import OffRoadNavEnv
from gym_offroad_nav.vehicle_model import VehicleModel
from lib import plotting

from a3c.estimators import PolicyEstimator, ValueEstimator

def actor_critic(env, estimator_policy, estimator_value, num_episodes, discount_factor=1.0):
    """
    Actor Critic Algorithm. Optimizes the policy 
    function approximator using policy gradient.

    Args:
    env: OpenAI environment.
    estimator_policy: Policy Function to be optimized 
    estimator_value: Value function approximator, used as a baseline
    num_episodes: Number of episodes to run for
    discount_factor: Time-discount factor

    Returns:
    An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    def form_mdp_state(env, state, prev_action, prev_reward):
        mdp_state = {
            "rewards": env.rewards[None, ..., None],
            "vehicle_state": state.T,
            "prev_action": prev_action.T,
            "prev_reward": np.array(prev_reward, dtype=np.float32).reshape((-1, 1))
        }
        return mdp_state

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    max_return = 0

    for i in range(num_episodes):
        # Reset the environment and pick the fisrst action

        # set initial state (x, y, theta, x', y', theta')
        state = np.array([+7, 10, 0, 0, 20, 1.5], dtype=np.float32).reshape(6, 1)
        action = np.array([0, 0], dtype=np.float32).reshape(2, 1)
        reward = 0
        env._reset(state)

        episode = []
        total_return = 0

        # One step in the environment
        for t in itertools.count():
            print "{}-#{:03d} ".format(t, i+1),

            env._render({
                "max_return": max_return,
                "total_return": total_return
            })

            # Take a step
            mdp_state = form_mdp_state(env, state, action, reward)
            action = estimator_policy.predict(mdp_state)
            # action[0] = 10
            next_state, reward, done, _ = env.step(action)

            # Keep track of the transition
            episode.append(Transition(
                state=state, action=action, reward=reward, next_state=next_state, done=done))

            # Update statistics (minus 1 reward per step)
            total_return += reward

            if total_return > max_return:
                max_return = total_return

            # Calculate TD Target
            next_mdp_state = form_mdp_state(env, next_state, action, reward)
            value = estimator_value.predict(mdp_state)
            value_next = estimator_value.predict(next_mdp_state)
            td_target = reward + discount_factor * value_next
            td_error = td_target - value

            # Update the value estimator
            estimator_value.update(mdp_state, td_target)

            # Update the policy estimator (use td-error as advantage estimate)
            estimator_policy.update(mdp_state, td_error, action)

            # Print out which step we're on, useful for debugging.
            print "action = [{:.2f}, {:.2f}]".format(action[0,0], action[1,0]),
            print "{:9.3f} (\33[93m{:9.3f}\33[0m)".format(reward, total_return),
            print "td_target (value) = {:5.2f} + {:5.2f} * {:5.2f} = {:5.2f}, value = {:5.2f}, td_error (policy) = {:5.2f}".format(
                reward, discount_factor, value_next, td_target, value, td_error)

            if done or total_return < 0:
                break

            state = next_state
        
        stats.episode_rewards[i] = total_return
        stats.episode_lengths[i] = t
        # writer.add_summary(summary, i)

    return stats

def main():
    vehicle_model = VehicleModel()
    # rewards = scipy.io.loadmat("/share/Research/Yamaha/d-irl/td_lambda_example/data.mat")["reward"]
    rewards = scipy.io.loadmat("data/circle2.mat")["reward"].astype(np.float32) - 100
    env = OffRoadNavEnv(rewards, vehicle_model)

    tf.reset_default_graph()

    global_step = tf.Variable(0, name="global_step", trainable=False)
    # import ipdb; ipdb.set_trace()
    policy_estimator = PolicyEstimator(env, learning_rate=0.001)
    value_estimator = ValueEstimator(reuse=False, learning_rate=0.1)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        # global writer
        # writer = tf.train.SummaryWriter("/Data3/tensorboard_log_dir", graph=sess.graph)
        # Note, due to randomness in the policy the number of episodes you need to learn a good
        # TODO: Sometimes the algorithm gets stuck, I'm not sure what exactly is happening there.
        stats = actor_critic(env, policy_estimator, value_estimator, 50000, discount_factor=0.95)

if __name__ == '__main__':
    main()
