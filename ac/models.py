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

    with tf.variable_scope(scope):
        # Forward LSTM Cell and its initial state placeholder
        lstm_fw = tf.nn.rnn_cell.LSTMCell(
            num_outputs, state_is_tuple=True, use_peepholes=True,
            # cell_clip=10., num_proj=num_outputs, proj_clip=10.
        )

        state_in_fw = [
            tf.placeholder(tf.float32, [batch_size, lstm_fw.state_size.c], name="c_fw"),
            tf.placeholder(tf.float32, [batch_size, lstm_fw.state_size.h], name="h_fw")
        ]

        if FLAGS.bi_directional:
            # Backward LSTM Cell and its initial state placeholder
            lstm_bw = tf.nn.rnn_cell.LSTMCell(
                num_outputs, state_is_tuple=True, use_peepholes=True,
                # cell_clip=10., num_proj=num_outputs, proj_clip=10.
            )

            state_in_bw = [
                tf.placeholder(tf.float32, [batch_size, lstm_bw.state_size.c], name="c_bw"),
                tf.placeholder(tf.float32, [batch_size, lstm_bw.state_size.h], name="h_bw")
            ]

            broadcaster = tf.to_int32(input[0, :, 0]) * 0
            sequence_length = tf.shape(input)[0] + broadcaster

            # bi-directional LSTM (forward + backward)
            lstm_outputs, state_out = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw, lstm_bw, input, dtype=tf.float32, time_major=True, scope=scope,
                sequence_length=sequence_length,
                initial_state_fw=tf.nn.rnn_cell.LSTMStateTuple(*state_in_fw),
                initial_state_bw=tf.nn.rnn_cell.LSTMStateTuple(*state_in_bw)
            )

            # Concate forward/backward output to create a tensor of shape
            # [seq_length, batch_size, 2 * num_outputs]
            lstm_outputs = tf_concat(-1, lstm_outputs)

            # Group forward/backward states into a single list
            state_out = state_out[0] + state_out[1]
            state_in = state_in_fw + state_in_bw
        else:
            # 1-directional LSTM (forward only)
            lstm_outputs, state_out = tf.nn.dynamic_rnn(
                lstm_fw, input, dtype=tf.float32, time_major=True, scope=scope,
                initial_state=tf.nn.rnn_cell.LSTMStateTuple(*state_in_fw)
            )

            state_in = state_in_fw

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

        conv2d = tf.contrib.layers.convolution2d

        # Batch norm is not compatible with LSTM
        # (see Issue: https://github.com/tensorflow/tensorflow/issues/6087)
        batch_norm = None # tf.contrib.layers.batch_norm

        conv1 = conv2d(input, 64, 5, activation_fn=tf.nn.relu, normalizer_fn=batch_norm, scope="conv1")
        conv2 = conv2d(conv1, 64, 5, activation_fn=tf.nn.relu, normalizer_fn=batch_norm, scope="conv2")
        pool1 = MaxPool2DLayer(conv2, pool_size=3, stride=2, name='pool1')

        conv3 = conv2d(pool1, 64, 5, activation_fn=tf.nn.relu, normalizer_fn=batch_norm, scope="conv3")
        conv4 = conv2d(conv3, 64, 5, activation_fn=tf.nn.relu, normalizer_fn=batch_norm, scope="conv4")
        pool2 = MaxPool2DLayer(conv4, pool_size=3, stride=2, name='pool2')

        """
        conv5 = conv2d(pool2, 64, 5, activation_fn=tf.nn.relu, normalizer_fn=batch_norm, scope="conv5")
        conv6 = conv2d(conv5, 64, 5, activation_fn=tf.nn.relu, normalizer_fn=batch_norm, scope="conv6")
        pool3 = MaxPool2DLayer(conv6, pool_size=3, stride=2, name='pool3')
        """

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
        # concat1 = fc
        lstm1 = LSTM(concat1, 128, scope="LSTM-1")

        # LSTM-2
        # Concatenate previous output with vehicle_state and prev_action
        concat2 = tf.concat(2, [fc, lstm1.output, vehicle_state, prev_action])
        # concat2 = lstm1.output
        lstm2 = LSTM(concat2, 256, scope="LSTM-2")

        output = lstm2.output

    layers = [
        conv1, conv2, pool1, conv3, conv4, pool2, # conv5, conv6, pool3,
        flat, fc, concat1, lstm1.output, concat2, lstm2.output
    ]

    if add_summaries:
        with tf.name_scope("summaries"):
            conv1_w = [v for v in tf.trainable_variables() if "conv1/weights"][0]
            grid = put_kernels_on_grid(conv1_w)
            tf.summary.image("conv1/weights", grid)

            tf.summary.image("front_view", front_view, max_outputs=500)

            # for layer in layers: tf.contrib.layers.summarize_activation(layer)

    for layer in layers:
        tf.logging.info(layer)

    return output, AttrDict(
        state_in  = lstm1.state_in  + lstm2.state_in,
        state_out = lstm1.state_out + lstm2.state_out,
        prev_state_out = None
    )

def policy_network(input, num_outputs, clip_mu=True):

    rank = get_rank(input)

    if rank == 3:
        S, B = get_seq_length_batch_size(input)
        input = flatten(input)

    input = DenseLayer(
        input=input,
        num_outputs=256,
        nonlinearity="relu",
        name="policy-input-dense")

    # Linear classifiers for mu and sigma
    mu = DenseLayer(
        input=input,
        num_outputs=num_outputs,
        name="policy-mu")

    sigma = DenseLayer(
        input=input,
        num_outputs=num_outputs,
        name="policy-sigma")

    if rank == 3:
        mu = deflatten(mu, S, B)
        sigma = deflatten(sigma, S, B)

    return tf.unstack(mu, axis=-1), tf.unstack(sigma, axis=-1)

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
