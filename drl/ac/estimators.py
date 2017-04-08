import tensorflow as tf
from utils import *
FLAGS = tf.flags.FLAGS

def get_estimator(type):
    # defer import to avoid cyclic import
    from a3c.estimators import A3CEstimator
    from acer.estimators import AcerEstimator
    from qprop.estimators import QPropEstimator

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

def create_avgnet_init_op(global_step, avg_vars, global_net, local_net):

    global_vars = global_net.var_list

    def copy_global_to_avg():
        msg = "\33[94mInitialize average net when global_step = \33[0m"
        disp_op = tf.Print(global_step, [global_step], msg)
        copy_op = make_copy_params_op(global_vars, avg_vars)
        return tf.group(*[copy_op, disp_op])

    init_avg_net = tf.cond(
        tf.equal(global_step, 0),
        copy_global_to_avg,
        lambda: tf.no_op()
    )

    with tf.control_dependencies([init_avg_net]):
        train_op = make_train_op(local_net, global_net)

        with tf.control_dependencies([train_op]):

            train_and_update_avgnet_op = make_copy_params_op(
                global_vars, avg_vars, alpha=FLAGS.avg_net_momentum
            )

    return train_and_update_avgnet_op

def compute_rho(a, pi, pi_behavior):
    pi_a = pi.prob(a)[..., None]
    mu_a = pi_behavior.prob(a)[..., None]

    rho = pi_a / mu_a
    rho = tf_print(rho)
    rho = tf.stop_gradient(rho)

    return rho
