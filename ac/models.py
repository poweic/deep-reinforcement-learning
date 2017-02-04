import tensorflow as tf
from tensorflow_helnet.layers import DenseLayer, Conv2DLayer, MaxPool2DLayer
# from keras.layers import Dense, Flatten, Input, merge, Lambda, Convolution2D
# from keras.models import Sequential, Model
from ac.utils import *
FLAGS = tf.flags.FLAGS
batch_size = FLAGS.batch_size
seq_length = FLAGS.seq_length

def get_state_placeholder():
    # Note that placeholder are tf.Tensor not tf.Variable
    FOV = FLAGS.field_of_view
    front_view    = tf.placeholder(tf.float32, [seq_length, batch_size, FOV, FOV, 1], "front_view")
    vehicle_state = tf.placeholder(tf.float32, [seq_length, batch_size, 6], "vehicle_state")
    prev_action   = tf.placeholder(tf.float32, [seq_length, batch_size, 2], "prev_action")
    prev_reward   = tf.placeholder(tf.float32, [seq_length, batch_size, 1], "prev_reward")

    return AttrDict(
        front_view    = front_view,
        vehicle_state = vehicle_state,
        prev_action   = prev_action,
        prev_reward   = prev_reward
    )

def naive_mean_steer_policy(front_view):
    H, W = front_view.get_shape().as_list()[2:4]

    yv, xv = np.meshgrid(range(H), range(W))
    xv = xv.astype(np.float32)
    yv = yv.astype(np.float32)

    theta = np.arctan((yv - W / 2 + 0.5) / (H - xv - 0.5))

    # Make sure it's symmetric with negative sign
    assert np.all(theta[:, :W/2] == -theta[:, -1:W/2-1:-1])

    theta = tf.constant(theta)[None, None, ..., None]

    # Substract min from front_view to make sure positivity
    r = front_view - tf.reduce_min(front_view, keep_dims=True,
                                   reduction_indices=[2,3,4])

    # Compute average steering angle (weighted sum by reward), add epsilon
    # to avoid divide by zero error
    num   = tf.reduce_sum(r * theta, reduction_indices=[2,3,4])
    denom = tf.reduce_sum(r        , reduction_indices=[2,3,4]) + 1e-10

    return num / denom

def LSTM(input, num_outputs, scope=None):
    lstm = tf.nn.rnn_cell.BasicLSTMCell(
        num_outputs, state_is_tuple=True, activation=tf.nn.relu
    )

    with tf.variable_scope(scope):
        state_in = [
            tf.placeholder(tf.float32, [batch_size, num_outputs], name="c"),
            tf.placeholder(tf.float32, [batch_size, num_outputs], name="h")
        ]

    lstm_outputs, state_out = tf.nn.dynamic_rnn(
        lstm, input, dtype=tf.float32, time_major=True, scope=scope,
        initial_state=tf.nn.rnn_cell.LSTMStateTuple(*state_in)
    )

    return AttrDict(
        output    = lstm_outputs,
        state_in  = state_in,
        state_out = state_out
    )

def build_shared_network(state, add_summaries=False):
    """
    Builds a 3-layer network conv -> conv -> fc as described
    in the A3C paper. This network is shared by both the policy and value net.
    Args:
    add_summaries: If true, add layer summaries to Tensorboard.
    Returns:
    Final layer activations.
    """

    front_view = state.front_view
    vehicle_state = state.vehicle_state
    prev_action = state.prev_action
    prev_reward = state.prev_reward

    rank = get_rank(state.front_view)

    if rank == 5:
        S, B = get_seq_length_batch_size(front_view)
        front_view = flatten(front_view)

    input = front_view

    with tf.name_scope("conv"):

        conv1 = Conv2DLayer(input, 32, 5, pad=2, nonlinearity="relu", name="conv1")
        conv2 = Conv2DLayer(conv1, 32, 5, pad=2, nonlinearity="relu", name="conv2", stride=2)
        pool1 = MaxPool2DLayer(conv2, pool_size=3, stride=2, name='pool1')

        conv3 = Conv2DLayer(pool1, 32, 5, pad=2, nonlinearity="relu", name="conv3")
        conv4 = Conv2DLayer(conv3, 32, 5, pad=2, nonlinearity="relu", name="conv4")
        pool2 = MaxPool2DLayer(conv4, pool_size=3, stride=2, name='pool2')

        flat = tf.contrib.layers.flatten(pool2)

        fc = DenseLayer(
            input=flat,
            num_outputs=256,
            nonlinearity="relu",
            name="fc")

    with tf.name_scope("lstm"):
        # Flatten convolutions output to fit fully connected layer
        fc = deflatten(fc, S, B)

        # LSTM-1
        # Concatenate encoder's output (i.e. flattened result from conv net)
        # with previous reward (see https://arxiv.org/abs/1611.03673)
        concat1 = tf.concat(2, [fc, prev_reward])
        lstm1 = LSTM(concat1, 64, scope="LSTM-1")

        # LSTM-2
        # Concatenate previous output with vehicle_state and prev_action
        concat2 = tf.concat(2, [fc, lstm1.output, vehicle_state, prev_action])
        lstm2 = LSTM(concat2, 256, scope="LSTM-2")

        output = lstm2.output

    layers = [conv1, conv2, conv3, conv4, fc,
              concat1, lstm1.output, concat2, lstm2.output]

    if add_summaries:
        with tf.name_scope("summaries"):
            conv1_w = [v for v in tf.trainable_variables() if "conv1/weights"][0]
            grid = put_kernels_on_grid(conv1_w)
            tf.summary.image("conv1/weights", grid)

            tf.summary.image("front_view", front_view, max_outputs=500)

            for layer in layers:
                tf.contrib.layers.summarize_activation(layer)

    for layer in layers:
        print layer

    return output, AttrDict(
        state_in  = lstm1.state_in  + lstm2.state_in,
        state_out = lstm1.state_out + lstm2.state_out,
        prev_state_out = None
    )

def policy_network(input, num_outputs):

    rank = get_rank(input)

    if rank == 3:
        S, B = get_seq_length_batch_size(input)
        input = flatten(input)

    '''
    input = DenseLayer(
        input=input,
        num_outputs=256,
        nonlinearity="relu",
        name="policy-input-dense")
    '''

    # Linear classifiers for mu and sigma
    mu = DenseLayer(
        input=input,
        num_outputs=num_outputs,
        name="policy-mu")

    sigma = DenseLayer(
        input=input,
        num_outputs=num_outputs,
        name="policy-sigma")

    if num_outputs <= 2:
        with tf.name_scope("mu_sigma_constraints"):
            min_mu = tf.constant([[FLAGS.min_mu_vf] + ([FLAGS.min_mu_steer] if num_outputs == 2 else [])], dtype=tf.float32)
            max_mu = tf.constant([[FLAGS.max_mu_vf] + ([FLAGS.max_mu_steer] if num_outputs == 2 else [])], dtype=tf.float32)

            min_sigma = tf.constant([[FLAGS.min_sigma_vf] + ([FLAGS.min_sigma_steer] if num_outputs == 2 else [])], dtype=tf.float32)
            max_sigma = tf.constant([[FLAGS.max_sigma_vf] + ([FLAGS.max_sigma_steer] if num_outputs == 2 else [])], dtype=tf.float32)

            # Clip mu by min and max, use softplus and capping for sigma
            # mu = clip(mu, min_mu, max_mu)
            # sigma = tf.minimum(tf.nn.softplus(sigma) + min_sigma, max_sigma)
            # sigma = tf.nn.sigmoid(sigma) * 1e-20 + max_sigma

            mu = softclip(mu, min_mu, max_mu)
            sigma = softclip(sigma, min_sigma, max_sigma)

    if rank == 3:
        mu = deflatten(mu, S, B)
        sigma = deflatten(sigma, S, B)

    return mu, sigma

def state_value_network(input, num_outputs=1):
    """
    This is state-only value V
    """

    rank = get_rank(input)

    if rank == 3:
        S, B = get_seq_length_batch_size(input)
        input = flatten(input)

    '''
    input = DenseLayer(
        input=input,
        num_outputs=256,
        nonlinearity="relu",
        name="value-input-dense")

    input = DenseLayer(
        input=input,
        num_outputs=256,
        nonlinearity="relu",
        name="value-input-dense-2")
    '''

    # This is just linear classifier
    value = DenseLayer(
        input=input,
        num_outputs=num_outputs,
        name="value-dense")

    value = tf.reshape(value, [-1, 1], name="value")

    if rank == 3:
        value = deflatten(value, S, B)

    return value
