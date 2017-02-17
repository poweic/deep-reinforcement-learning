import tensorflow as tf
from ac.utils import *
FLAGS = tf.flags.FLAGS

def create_distribution(dist_type, bijectors=None, **stats):

    num_actions = FLAGS.num_actions
    AS = FLAGS.action_space
    low, high = AS.low, AS.high

    # Need broadcaster to make every as the shape [seq_length, batch_size, ...]
    broadcaster = stats['param1'][..., 0] * 0

    DIST = tf.contrib.distributions.Normal if dist_type == "normal" else tf.contrib.distributions.Beta

    stats['param1'] = tf_check_numerics(stats['param1'])
    stats['param2'] = tf_check_numerics(stats['param2'])

    dists = [
        DIST(
            stats['param1'][..., i], stats['param2'][..., i], allow_nan_stats=False
        ) for i in range(num_actions)
    ]

    dists = add_eps_exploration(dists, broadcaster)

    pi = to_joint_distribution(dists, bijectors, dist_type)

    pi.phi = [stats['param1'], stats['param2']]
    # pi.phi = reduce(lambda x,y:x+y, stats.itervalues())
    pi.stats = stats

    return pi

def add_eps_exploration(dists, broadcaster):

    if FLAGS.eps_init == 0:
        return dists

    a = tf.constant(FLAGS.action_space.low,  tf.float32)[:, None, None] + broadcaster[None, ...]
    b = tf.constant(FLAGS.action_space.high, tf.float32)[:, None, None] + broadcaster[None, ...]

    # eps-greedy-like policy with epsilon decays over time.
    # NOTE I use (t / N + 1) rather than t so that it looks like:
    # 1.00, 1.01, 1.02, ... instead of 1, 2, 3, 4, which decays too fast
    global_step = tf.contrib.framework.get_global_step()
    eff_timestep = (tf.to_float(global_step) / FLAGS.effective_timescale + 1.)
    eps = FLAGS.eps_init / eff_timestep

    # With eps probability, we sample our action from random uniform
    prob = tf.pack([1. - eps, eps], axis=-1)[None, None, ...] + broadcaster[..., None]
    cat = tf.contrib.distributions.Categorical(p=prob, allow_nan_stats=False)
    tf.logging.info(cat.sample_n(1))

    for i in range(FLAGS.num_actions):

        # Create uniform dist
        uniform = tf.contrib.distributions.Uniform(a=a[i], b=b[i], allow_nan_stats=False)
        tf.logging.info(uniform.sample_n(1))

        # Eps-Normal as policy
        dists[i] = tf.contrib.distributions.Mixture(
            cat=cat, components=[dists[i], uniform], allow_nan_stats=False
        )

    return dists

def to_joint_distribution(dists, bijectors, dist_type):

    if bijectors is None:
        bijectors = [None] * len(dists)

    low  = tf.constant(FLAGS.action_space.low , tf.float32)[None, None, None, ...]
    high = tf.constant(FLAGS.action_space.high, tf.float32)[None, None, None, ...]

    # identity = AttrDict(forward_fn=lambda x:x, inverse_fn=lambda x:x)
    # bijectors = [b if b is not None else identity for b in bijectors]

    def log_prob(x, msg=None):
        x = x[None, ...]

        """
        log_p = reduce(lambda a,b: a+b, [
            tf_print(dists[i].log_prob(
                bijectors[i].inverse_fn(
                    tf_print(x[..., i], message="x[..., {}] = ".format(i))
                )
            )[0, ...], message="log_prob[..., {}] = ".format(i)) for i in range(len(dists))
        ])
        # log_p = tf_print(log_p, message="in to_joint_distribution: log_p = ")
        """
        if dist_type == "beta":
            x = clip((x - low) / (high - low), 0., 1.)

        log_p = reduce(lambda a,b: a+b, [
            dists[i].log_prob(x[..., i])[0, ...] for i in range(len(dists))
        ])

        return log_p

    def prob(x, msg=None):
        return tf.exp(log_prob(x))

    def entropy():
        return sum([dists[i].entropy() for i in range(len(dists))])

    def sample_n(n, msg=None):
        samples = tf.pack([dists[i].sample_n(n) for i in range(len(dists))], axis=-1)

        if dist_type == "normal":
            samples = clip(samples, low, high)
        else:
            samples = samples * (high - low) + low

        return samples

    dist = AttrDict(
        prob = prob,
        log_prob = log_prob,
        sample_n = sample_n,
        entropy = entropy,
        dists = dists
    )

    return dist
