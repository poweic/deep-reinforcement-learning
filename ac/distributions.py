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

    param1 = tf_check_numerics(stats['param1'])
    param2 = tf_check_numerics(stats['param2'])

    """
    dists = [
        DIST(
            param1[..., i], param2[..., i], allow_nan_stats=False
        ) for i in range(num_actions)
    ]

    dists = add_eps_exploration(dists, broadcaster)
    pi = to_joint_distribution(dists, bijectors, dist_type)
    """

    dist = DIST(param1, param2, allow_nan_stats=False)
    pi = to_transformed_distribution(dist, dist_type)

    pi.phi = [param1, param2]
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

def to_transformed_distribution(dist, dist_type):

    low  = tf.constant(FLAGS.action_space.low , tf.float32)[None, None, None, ...]
    high = tf.constant(FLAGS.action_space.high, tf.float32)[None, None, None, ...]

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

""" FIXME Old codes from gym-offroad-nav:master
def to_joint_distribution(dists, bijectors):
    if bijectors is None:
        bijectors = [None] * len(dists)

    identity = AttrDict(forward_fn=lambda x:x, inverse_fn=lambda x:x)
    bijectors = [b if b is not None else identity for b in bijectors]

    def log_prob(x, msg=None):
        x = x[None, ...]
        log_p = reduce(lambda a,b: a+b, [
            tf_print(dists[i].log_prob(
                bijectors[i].inverse_fn(
                    tf_print(x[..., i], message="x[..., {}] = ".format(i))
                )
            )[0, ...], message="log_prob[..., {}] = ".format(i)) for i in range(len(dists))
        ])
        # log_p = tf_print(log_p, message="in to_joint_distribution: log_p = ")

        return log_p

    def prob(x, msg=None):
        return tf.exp(log_prob(x))

    low  = FLAGS.action_space.low
    high = FLAGS.action_space.high

    def sample_n(n, msg=None):
        samples = [
            bijectors[i].forward_fn(
                clip(dists[i].sample_n(n), low[i], high[i])
            ) for i in range(len(dists))
        ]

        return tf.pack(samples, axis=-1)

    dist = AttrDict(
        prob = prob,
        log_prob = log_prob,
        sample_n = sample_n
    )

    return dist
"""


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

"""
def mixture_density(mu_vf, sigma_vf, logits_steer, sigma_steer, vf, steer_buckets, msg):

    # mu: [B, 2], sigma: [B, 2], phi is just a syntatic sugar
    with tf.variable_scope("speed_control_policy"):
        mu_vf = tf_print(mu_vf, msg + ", mu_vf = ")
        sigma_vf = tf_print(sigma_vf, msg + ", sigma_vf = ")
        dist_vf = tf.contrib.distributions.MultivariateNormalDiag(
            mu_vf, sigma_vf, allow_nan_stats=False
        )

    with tf.variable_scope("steer_control_policy"):
        # Create broadcaster to repeat over time, batch axes
        broadcaster = logits_steer[..., 0:1] * 0

        components = [
            tf.contrib.distributions.MultivariateNormalDiag(
                mu = steer_to_yawrate(s + broadcaster, vf),
                diag_stdev = sigma_steer[..., i:i+1],
                allow_nan_stats = False
            ) for i, s in enumerate(steer_buckets)
        ]

        cat = tf.contrib.distributions.Categorical(
            logits=logits_steer, allow_nan_stats=False
        )

        # Gaussian Mixture Model (GMM)
        dist_steer = gmm = tf.contrib.distributions.Mixture(
            cat, components, allow_nan_stats=False
        )

    pi = AttrDict(
        phi = [mu_vf, sigma_vf, logits_steer, sigma_steer, vf],
        mu_vf         = mu_vf,
        sigma_vf      = sigma_vf,
        logits_steer  = logits_steer,
        prob_steer    = tf.nn.softmax(logits_steer),
        sigma_steer   = sigma_steer,
        vf            = vf,
        steer_buckets = steer_buckets,
        n_buckets     = len(steer_buckets)
    )

    def prob(x, msg):
        x = x[None, ...]

        x1 = tf_print(x[..., 0:1], msg + ", x1 = ")
        x2 = tf_print(x[..., 1:2], msg + ", x2 = ")

        p1 = dist_vf.prob(x1)[0, ...]
        p2 = dist_steer.prob(x2)[0, ...]

        p1 = tf_print(p1, msg + ", p1 = ")
        p2 = tf_print(p2, msg + ", p2 = ")
        return p1 * p2

    def log_prob(x):
        x = x[None, ...]
        lp1 = dist_vf.log_prob(x[..., 0:1])[0, ...]
        lp2 = dist_steer.log_prob(x[..., 1:2])[0, ...]
        lp1 = tf_print(lp1)
        lp2 = tf_print(lp2)
        return lp1 + lp2

    def sample_n(n):
        s1 = dist_vf.sample_n(n)
        s1 = tf.maximum(s1, 0.)

        s2 = dist_steer.sample_n(n)
        s = tf_concat(-1, [s1, s2])
        return s

    pi.prob     = prob
    pi.log_prob = log_prob
    pi.sample_n = sample_n

    return pi
"""
