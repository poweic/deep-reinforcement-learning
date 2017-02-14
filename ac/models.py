import tensorflow as tf
from ac.utils import *
FLAGS = tf.flags.FLAGS
batch_size = FLAGS.batch_size
seq_length = FLAGS.seq_length
num_states = FLAGS.num_states
num_actions = FLAGS.num_actions

def get_state_placeholder():
    # Note that placeholder are tf.Tensor not tf.Variable
    state = tf.placeholder(tf.float32, [seq_length, batch_size, num_states], "state")
    prev_action = tf.placeholder(tf.float32, [seq_length, batch_size, num_actions], "prev_action")
    prev_reward = tf.placeholder(tf.float32, [seq_length, batch_size, 1], "prev_reward")

    return AttrDict(
        state = state,
        prev_action = prev_action,
        prev_reward = prev_reward
    )

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

def build_shared_network(input, add_summaries=False):
    """
    Builds a 3-layer network conv -> conv -> fc as described
    in the A3C paper. This network is shared by both the policy and value net.
    Args:
    add_summaries: If true, add layer summaries to Tensorboard.
    Returns:
    Final layer activations.
    """

    S, B = get_seq_length_batch_size(input.state)
    state = input.state
    prev_action = input.prev_action
    prev_reward = input.prev_reward

    with tf.name_scope("lstm"):
        # Flatten convolutions output to fit fully connected layer
        fc = state

        # LSTM-1
        # Concatenate encoder's output (i.e. flattened result from conv net)
        # with previous reward (see https://arxiv.org/abs/1611.03673)
        concat1 = tf.concat(2, [fc, prev_reward])
        # concat1 = fc
        lstm1 = LSTM(concat1, 128, scope="LSTM-1")

        # LSTM-2
        # Concatenate previous output with vehicle_state and prev_action
        concat2 = tf.concat(2, [fc, lstm1.output, prev_action])
        # concat2 = lstm1.output
        lstm2 = LSTM(concat2, 256, scope="LSTM-2")

        output = lstm2.output

    layers = [
        fc, concat1, lstm1.output, concat2, lstm2.output
    ]

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

    input = tf.contrib.layers.fully_connected(
	inputs=input,
	num_outputs=256,
	activation_fn=tf.nn.relu,
	scope="policy-input-dense")

    # Linear classifiers for mu and sigma
    mu = tf.contrib.layers.fully_connected(
        inputs=input,
        num_outputs=num_outputs,
        scope="policy-mu")

    sigma = tf.contrib.layers.fully_connected(
        inputs=input,
        num_outputs=num_outputs,
        scope="policy-sigma")

    if rank == 3:
        mu = deflatten(mu, S, B)
        sigma = deflatten(sigma, S, B)

    return mu, sigma
    # return tf.unstack(mu, axis=-1), tf.unstack(sigma, axis=-1)

def state_value_network(input, num_outputs=1):
    """
    This is state-only value V
    """

    rank = get_rank(input)

    if rank == 3:
        S, B = get_seq_length_batch_size(input)
        input = flatten(input)

    # This is just linear classifier
    value = tf.contrib.layers.fully_connected(
	inputs=input,
	num_outputs=num_outputs,
	scope="value-dense")

    value = tf.reshape(value, [-1, 1], name="value")

    if rank == 3:
        value = deflatten(value, S, B)

    return value
