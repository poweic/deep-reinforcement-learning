import numpy as np
import tensorflow as tf
from drl.ac.distributions import create_distribution
from drl.ac.models import policy_network
FLAGS = tf.flags.FLAGS

def build_policy(input, dist_type):
    if dist_type == "Gaussian":
        return gaussian_policy(input)
    elif dist_type == "Beta":
        return beta_policy(input)
    else:
        raise ValueError('dist_type must be either "Gaussian" or "Beta"')

def get_param_placeholder(name):
    return tf.placeholder(tf.float32, [
        FLAGS.seq_length, FLAGS.batch_size, FLAGS.num_actions
    ], name=name)

def beta_policy(input):

    alpha, beta = policy_network(input, FLAGS.num_actions, clip_mu=False)

    alpha = tf.nn.softplus(alpha) + 2
    beta  = tf.nn.softplus(beta)  + 2

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
    low  = tf.constant(AS.low , tf.float32)[None, None, :] + broadcaster
    high = tf.constant(AS.high, tf.float32)[None, None, :] + broadcaster
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

"""
def mixture_density_policy(self, input, use_naive_policy):

    def get_steer_buckets(w):
        n = int(30 / w)
        return np.linspace(-n*w, n*w, 2*n+1).astype(np.float32)

    # mu: [B, 2], sigma: [B, 2], phi is just a syntatic sugar
    with tf.variable_scope("speed_control_policy"):
        mu_vf, sigma_vf = policy_network(input, 1)

    with tf.variable_scope("steer_control_policy"):
        steer_buckets = get_steer_buckets(FLAGS.bucket_width)
        n_buckets = len(steer_buckets)

        tf.logging.info("steer_buckets      = {} degree".format(steer_buckets))
        steer_buckets = to_radian(steer_buckets)

        min_sigma = to_radian(0.1)
        max_sigma = to_radian(FLAGS.bucket_width / 2)

        logits_steer, sigma_steer = policy_network(input, n_buckets)

        if use_naive_policy:
            naive_mu = naive_mean_steer_policy(self.state.front_view)
            diff = naive_mu[..., None] - tf.constant(steer_buckets[None, None, :], tf.float32)
            logits_steer = -tf.square(diff * 7) + logits_steer * 0

        sigma_steer = softclip(sigma_steer, min_sigma, max_sigma)

    tf.logging.info("mu_vf.shape        = {}".format(tf_shape(mu_vf)))
    tf.logging.info("sigma_vf.shape     = {}".format(tf_shape(sigma_vf)))
    tf.logging.info("logits_steer.shape = {}".format(tf_shape(logits_steer)))
    tf.logging.info("sigma_steer.shape  = {}".format(tf_shape(sigma_steer)))

    pi = mixture_density(mu_vf, sigma_vf, logits_steer, sigma_steer,
                         self.get_forward_velocity(), steer_buckets,
                         msg="pi")

    # Placeholder for behavior policy
    pi_behavior = mixture_density(
        tf.placeholder(tf.float32, [seq_length, batch_size, 1], "mu_vf"),
        tf.placeholder(tf.float32, [seq_length, batch_size, 1], "sigma_vf"),
        tf.placeholder(tf.float32, [seq_length, batch_size, n_buckets], "logits_steer"),
        tf.placeholder(tf.float32, [seq_length, batch_size, n_buckets], "sigma_steer"),
        tf.placeholder(tf.float32, [seq_length, batch_size, 1], "vf"),
        steer_buckets,
        msg="pi_behavior"
    )

    self.pi_prob_steer = pi.prob_steer
    self.mu_prob_steer = pi_behavior.prob_steer

    return pi, pi_behavior

def beta_policy(self, input, use_naive_policy):

    AS = FLAGS.action_space
    alpha, beta = policy_network(input, AS.n_actions, clip_mu=False)

    for i in range(len(alpha)):
        alpha[i] = tf.nn.softplus(alpha[i]) + 2
        beta[i]  = tf.nn.softplus(beta[i])  + 2

        alpha[i] = tf_check_numerics(alpha[i])
        beta[i]  = tf_check_numerics(beta[i])

        alpha[i] = tf_print(alpha[i])
        beta[i]  = tf_print(beta[i])

    # Add naive policy as baseline bias
    if use_naive_policy:
        # TODO haven't figured out yet
        pass

    # Turn steering angle to yawrate basedon forward velocity
    vf = self.get_forward_velocity()[..., 0][None, ...]
    def forward_fn(x):
        yawrate = steer_to_yawrate(x, vf)
        yawrate = tf_print(yawrate)
        return yawrate

    def inverse_fn(x):
        x = tf_print(x)
        steer = yawrate_to_steer(x, tf_print(vf))
        steer = tf_print(steer)
        return steer

    bijectors = [
        None,
        AttrDict(forward_fn = forward_fn, inverse_fn = inverse_fn)
    ]

    pi = create_distribution(
        dist_type="beta", bijectors=bijectors, alpha=alpha, beta=beta
    )

    pi_behavior = create_distribution(
        dist_type="beta", 
        bijectors=bijectors,
        alpha = [
            tf.placeholder(tf.float32, [seq_length, batch_size], "alpha_vf"),
            tf.placeholder(tf.float32, [seq_length, batch_size], "alpha_steer"),
        ],
        beta  = [
            tf.placeholder(tf.float32, [seq_length, batch_size], "beta_vf"),
            tf.placeholder(tf.float32, [seq_length, batch_size], "beta_steer"),
        ]
    )

    return pi, pi_behavior

def gaussian_policy(self, input, use_naive_policy):

    mu, sigma = policy_network(input, FLAGS.action_space.n_actions, clip_mu=False)

    # Add naive policy as baseline bias
    if use_naive_policy:
        mu[1] = mu[1] + naive_mean_steer_policy(self.state.front_view)

    # Turn steering angle to yawrate basedon forward velocity
    mu[1] = steer_to_yawrate(mu[1], self.get_forward_velocity()[..., 0])

    # Confine action space
    AS = FLAGS.action_space
    for i in range(len(mu)):
        mu[i]    = softclip(mu[i]   , AS.low[i]      , AS.high[i]      )
        sigma[i] = softclip(sigma[i], AS.sigma_low[i], AS.sigma_high[i])

        # For debugging
        mu[i]    = tf_check_numerics(mu[i])
        sigma[i] = tf_check_numerics(sigma[i])

    pi = create_distribution(dist_type="normal", mu=mu, sigma=sigma)

    pi_behavior = create_distribution(
        dist_type="normal", 
        mu    = [
            tf.placeholder(tf.float32, [seq_length, batch_size], "mu_vf"),
            tf.placeholder(tf.float32, [seq_length, batch_size], "mu_steer"),
        ],
        sigma = [
            tf.placeholder(tf.float32, [seq_length, batch_size], "sigma_vf"),
            tf.placeholder(tf.float32, [seq_length, batch_size], "sigma_steer"),
        ]
    )

    return pi, pi_behavior
"""
