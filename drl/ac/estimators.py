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

def compute_Q_ret_Q_opc(values, value_last, c, r, Q_tilt_a):
    """
    Use tf.while_loop to compute Q_ret, Q_opc
    """
    batch_size = FLAGS.batch_size

    tf.logging.info("Compute Q_ret & Q_opc recursively ...")
    gamma = tf_const(FLAGS.discount_factor)
    lambda_ = tf_const(FLAGS.lambda_)
    # gamma = tf_print(gamma, "gamma = ")

    with tf.name_scope("initial_value"):
        # Use "done" to determine whether x_k is terminal state. If yes,
        # set initial Q_ret to 0. Otherwise, bootstrap initial Q_ret from V.
        Q_ret_0 = value_last * int(FLAGS.bootstrap)
        Q_opc_0 = Q_ret_0

        Q_ret = Q_ret_0
        Q_opc = Q_opc_0

        k = tf.shape(values)[0] # if seq_length is None else seq_length
        i_0 = k - 1

    def cond(i, Q_ret_i, Q_opc_i, Q_ret, Q_opc):
        return i >= 0

    def body(i, Q_ret_i, Q_opc_i, Q_ret, Q_opc):

        # Q^{ret} \leftarrow r_i + \gamma Q^{ret}
        with tf.name_scope("r_i"):
            r_i = r[i:i+1, ...]

        with tf.name_scope("pre_update"):
            Q_ret_i = r_i + gamma * Q_ret_i
            Q_opc_i = r_i + gamma * Q_opc_i

        # TF equivalent of .prepend()
        with tf.name_scope("prepend"):
            Q_ret = tf.concat([Q_ret_i, Q_ret], 0)
            Q_opc = tf.concat([Q_opc_i, Q_opc], 0)

        # Q^{ret} \leftarrow c_i (Q^{ret} - Q(x_i, a_i)) + V(x_i)
        with tf.name_scope("post_update"):
            with tf.name_scope("c_i"): c_i = c[i:i+1, ...]
            with tf.name_scope("Q_i"): Q_i = Q_tilt_a[i:i+1, ...]
            with tf.name_scope("V_i"): V_i = values[i:i+1, ...]

            # ACER with Generalized Advantage Estimation (GAE):
            # For lambda = 1: this is original ACER with k-step TD error
            # For lambda = 0: 1-step TD error (low variance, high bias)
            Q_ret_i = lambda_ * c_i * (Q_ret_i - Q_i) + V_i
            Q_opc_i = lambda_       * (Q_opc_i - Q_i) + V_i

        return i-1, Q_ret_i, Q_opc_i, Q_ret, Q_opc

    i, Q_ret_i, Q_opc_i, Q_ret, Q_opc = tf.while_loop(
        cond, body,
        loop_vars=[
            i_0, Q_ret_0, Q_opc_0, Q_ret, Q_opc
        ],
        shape_invariants=[
            i_0.get_shape(),
            Q_ret_0.get_shape(),
            Q_opc_0.get_shape(),
            tf.TensorShape([None, batch_size, 1]),
            tf.TensorShape([None, batch_size, 1])
        ]
    )

    Q_ret = tf.stop_gradient(Q_ret[:-1, ...], name="Q_ret")
    Q_opc = tf.stop_gradient(Q_opc[:-1, ...], name="Q_opc")

    return Q_ret, Q_opc
