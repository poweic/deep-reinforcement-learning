import tensorflow as tf
from drl.ac.utils import *
FLAGS = tf.flags.FLAGS

def create_distribution(dist_type, bijectors=None, **stats):

    num_actions = FLAGS.num_actions
    AS = FLAGS.action_space
    low, high = AS.low, AS.high

    # Need broadcaster to make every as the shape [seq_length, batch_size, ...]
    broadcaster = stats['param1'][..., 0] * 0

    DIST = tf.contrib.distributions.Normal if dist_type == "normal" else tf.contrib.distributions.Beta

    param1 = tf_check_numerics(stats['param1'])
    param2 = tf_check_numerics(stats['param2'])

    param1 = tf_print(param1)
    param2 = tf_print(param2)

    dist = DIST(param1, param2, allow_nan_stats=False)
    pi = to_transformed_distribution(dist, dist_type)

    pi.phi = [param1, param2]
    pi.stats = AttrDict(stats)

    return pi

def to_transformed_distribution(dist, dist_type):

    low  = tf_const(FLAGS.action_space.low )[None, None, None, ...]
    high = tf_const(FLAGS.action_space.high)[None, None, None, ...]

    def log_prob(x, msg=None):
        x = x[None, ...]
        if dist_type == "beta":
            x = clip((x - low) / (high - low), 0., 1.)
        return tf.reduce_sum(dist.log_prob(x)[0, ...], axis=-1)

    def prob(x, msg=None):
        return tf.exp(log_prob(x))

    def entropy():
        return tf.reduce_sum(dist.entropy(), axis=-1)

    def sample_n(n, msg=None):

        samples = dist.sample_n(n)

        if dist_type == "normal":
            samples = clip(samples, low, high)
        else:
            samples = samples * (high - low) + low

        return samples

    return AttrDict(
        prob = prob,
        log_prob = log_prob,
        sample_n = sample_n,
        entropy = entropy,
        dist = dist
    )
