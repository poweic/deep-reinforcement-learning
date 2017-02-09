import tensorflow as tf
from ac.utils import *
FLAGS = tf.flags.FLAGS

def create_distribution(dist_type, bijectors=None, **stats):

    AS = FLAGS.action_space
    n_actions = AS.n_actions
    low, high = AS.low, AS.high

    if dist_type == "normal":
        # Need broadcaster to make every as the shape [seq_length, batch_size, ...]
        broadcaster = stats['mu'][0] * 0

        dists = [
            tf.contrib.distributions.Normal(
                stats['mu'][i], stats['sigma'][i], allow_nan_stats=False
            ) for i in range(n_actions)
        ]
    elif dist_type == "beta":
        # Need broadcaster to make every as the shape [seq_length, batch_size, ...]
        broadcaster = stats['beta'][0] * 0

        beta_dists = [
            tf.contrib.distributions.Beta(
                stats['alpha'][i], stats['beta'][i], allow_nan_stats=False
            ) for i in range(n_actions)
        ]

        beta_bijectors = [
            tf.contrib.distributions.bijector.ScaleAndShift(
                shift = tf.constant(low[i], tf.float32),
                scale = tf.constant(high[i] - low[i], tf.float32),
            ) for i in range(n_actions)
        ]

        dists = [
            tf.contrib.distributions.TransformedDistribution(
                distribution = dist,
                bijector = bijector
            ) for dist, bijector in zip(beta_dists, beta_bijectors)
        ]

    dists = add_eps_exploration(dists, broadcaster)

    pi = to_joint_distribution(dists, bijectors)

    pi.phi = reduce(lambda x,y:x+y, stats.itervalues())
    pi.stats = stats

    return pi

def add_eps_exploration(dists, broadcaster):

    a = [
        tf.constant([[FLAGS.min_mu_vf]],    tf.float32) + broadcaster,
        tf.constant([[FLAGS.min_mu_steer]], tf.float32) + broadcaster
    ]

    b = [
        tf.constant([[FLAGS.max_mu_vf]],    tf.float32) + broadcaster,
        tf.constant([[FLAGS.max_mu_steer]], tf.float32) + broadcaster
    ]

    # 2 means R^2
    action_space = 2
    for i in range(action_space):

        # eps-greedy-like policy for steering angle
        eps = 0.05
        prob = tf.constant([[[1. - eps, eps]]], tf.float32) + broadcaster[..., None]
        cat = tf.contrib.distributions.Categorical(p=prob, allow_nan_stats=False)
        print cat.sample_n(1)

        # Create uniform dist with compatible size using broadcaster
        uniform = tf.contrib.distributions.Uniform(a=a[i], b=b[i], allow_nan_stats=False)
        print uniform.sample_n(1)

        # Eps-Normal as policy
        dists[i] = tf.contrib.distributions.Mixture(
            cat=cat, components=[dists[i], uniform], allow_nan_stats=False
        )

    return dists

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

        log_p = tf_print(log_p, message="in to_joint_distribution: log_p = ")

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

        s = tf.pack(samples, axis=-1)
        print "\33[94ms = \33[0m", s

        return s

    dist = AttrDict(
        prob = prob,
        log_prob = log_prob,
        sample_n = sample_n
    )

    return dist

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

