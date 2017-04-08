import tensorflow as tf
from a3c.estimators import A3CEstimator
from acer.estimators import AcerEstimator
from qprop.estimators import QPropEstimator
from drl.ac.utils import tf_const

def get_estimator(type):
    type = type.upper()

    print "Using {} as estimator".format(type)

    if type == "A3C":
        return A3CEstimator
    elif type == "ACER":
        AcerEstimator.create_averge_network()
        return AcerEstimator
    elif type == "QPROP":
        QPropEstimator.create_averge_network()
        return QPropEstimator
    else:
        raise TypeError("Unknown type " + type)

def add_fast_TRPO_regularization(pi, avg_net_pi, obj):

    # ACER gradient is the gradient of policy objective function, which is
    # the negatation of policy loss
    g = tf.concat(tf.gradients(obj, pi.phi), -1,)

    g_trpo, mean_KL = compute_trust_region_update(g, avg_net_pi, pi)

    # surrogate objective function
    phi = tf.concat(pi.phi, -1)
    obj_sur = phi * g_trpo

    return obj_sur, mean_KL

def compute_trust_region_update(g, pi_avg, pi, delta=0.5):
    """
    In ACER's original paper, they use delta uniformly sampled from [0.1, 2]
    """
    try:
        # Compute the KL-divergence between the policy distribution of the
        # average policy network and those of this network, i.e. KL(avg || this)
        KL_divergence = tf.reduce_sum(tf.contrib.distributions.kl(
            pi_avg.dist, pi.dist, allow_nan=False), axis=2)

        mean_KL = tf.reduce_mean(KL_divergence)

        # Take the partial derivatives w.r.t. phi (i.e. mu and sigma)
        # k = tf.pack(tf.gradients(KL_divergence, pi.phi), axis=-1)
        k = tf.concat(tf.gradients(KL_divergence, pi.phi), -1)

        # Compute \frac{k^T g - \delta}{k^T k}, perform reduction only on axis 2
        num   = tf.reduce_sum(k * g, axis=2, keep_dims=True) - delta
        denom = tf.reduce_sum(k * k, axis=2, keep_dims=True)

        # Hold gradient back a little bit if KL divergence is too large
        correction = tf.maximum(tf_const(0.), num / denom) * k

        # z* is the TRPO regularized gradient
        z_star = g - correction

    except NotImplementedError:
        tf.logging.warn("\33[33mFailed to create TRPO update. Fall back to normal update\33[0m")
        z_star = g

    # By using stop_gradient, we make z_star being treated as a constant
    z_star = tf.stop_gradient(z_star)

    return z_star, mean_KL
