import numpy as np
import tensorflow as tf
from drl.ac.distributions import create_distribution
from drl.ac.models import policy_network
from drl.ac.utils import *
FLAGS = tf.flags.FLAGS

def build_policy(input, dist_type):
    if dist_type == "Gaussian":
        return gaussian_policy(input)
    elif dist_type == "Beta":
        return beta_policy(input)
    else:
        raise ValueError('dist_type must be either "Gaussian" or "Beta"')

def get_param_placeholder(name):
    return tf.placeholder(FLAGS.dtype, [
        FLAGS.seq_length, FLAGS.batch_size, FLAGS.num_actions
    ], name=name)

def softplus(x):
    # tf.nn.softplus doesn't support 2nd derivatives ...
    return tf.log(tf.exp(x) + 1)

def beta_policy(input):

    alpha, beta = policy_network(input, FLAGS.num_actions, clip_mu=False)

    alpha = softplus(alpha) + 2
    beta  = softplus(beta)  + 2

    pi = create_distribution(dist_type="beta", param1=alpha, param2=beta)

    pi_behavior = create_distribution(
        dist_type="beta", 
        param1 = get_param_placeholder("param1"),
        param2 = get_param_placeholder("param2"),
    )

    return pi, pi_behavior

def gaussian_policy(input):

    mu, sigma = policy_network(input, FLAGS.num_actions, clip_mu=False)

    AS = FLAGS.action_space
    broadcaster = mu[..., 0:1] * 0
    low  = tf_const(AS.low )[None, None, :] + broadcaster
    high = tf_const(AS.high)[None, None, :] + broadcaster
    mu    = softclip(mu, low , high)

    # sigma = tf.nn.softplus(sigma) + 1e-4
    sigma = softclip(sigma, 1e-4, (high - low) / 2.)

    pi = create_distribution(dist_type="normal", param1=mu, param2=sigma)

    pi_behavior = create_distribution(
        dist_type="normal", 
        param1 = get_param_placeholder("param1"),
        param2 = get_param_placeholder("param2"),
    )

    return pi, pi_behavior
