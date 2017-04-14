import sys
import tensorflow as tf
from drl.ac.utils import *
FLAGS = tf.flags.FLAGS

def naive_mean_steer_policy(front_view):
    H, W = front_view.get_shape().as_list()[2:4]

    yv, xv = np.meshgrid(range(H), range(W))
    xv = xv.astype(np.float32)
    yv = yv.astype(np.float32)

    theta = np.arctan((yv - W / 2 + 0.5) / (H - xv - 0.5))

    # Make sure it's symmetric with negative sign
    assert np.all(theta[:, :W/2] == -theta[:, -1:W/2-1:-1])

    theta = tf_const(theta)[None, None, ..., None]

    # Substract min from front_view to make sure positivity
    r = front_view - tf.reduce_min(front_view, keep_dims=True,
                                   reduction_indices=[2,3,4])

    # Compute average steering angle (weighted sum by reward), add epsilon
    # to avoid divide by zero error
    num   = tf.reduce_sum(r * theta, reduction_indices=[2,3,4])
    denom = tf.reduce_sum(r        , reduction_indices=[2,3,4]) + 1e-10

    return num / denom

def get_state_placeholder():
    S = FLAGS.seq_length
    B = FLAGS.batch_size

    # Note that placeholder are tf.Tensor not tf.Variable
    prev_action = tf.placeholder(FLAGS.dtype, [S, B, FLAGS.num_actions], "prev_action")
    prev_reward = tf.placeholder(FLAGS.dtype, [S, B, 1], "prev_reward")

    placeholder = AttrDict(
        prev_action = prev_action,
        prev_reward = prev_reward
    )

    if "OffRoadNav" in FLAGS.game:
        front_view_shape = list(FLAGS.observation_space.spaces[0].shape)
        front_view    = tf.placeholder(FLAGS.dtype, [S, B] + front_view_shape, "front_view")
        vehicle_state = tf.placeholder(FLAGS.dtype, [S, B, 6], "vehicle_state")

        placeholder.update({
            'front_view': front_view,
            'vehicle_state': vehicle_state
        })
    else:
        state = tf.placeholder(FLAGS.dtype, [S, B, FLAGS.num_states], "state")
        placeholder.update({'state': state})

    return placeholder

def LSTM(input, num_outputs, scope=None, state_in_fw=None, state_in_bw=None):

    batch_size = FLAGS.batch_size

    # starting from TensorFlow v1.0, tf.nn.rnn_cell is moved to tf.contrib.rnn
    rnn_core = tf.contrib.rnn

    with tf.variable_scope(scope):
        # Forward LSTM Cell and its initial state placeholder
        lstm_fw = rnn_core.LSTMCell(
            num_outputs, state_is_tuple=True, use_peepholes=True,
            # cell_clip=10., num_proj=num_outputs, proj_clip=10.
        )

        if state_in_fw is None:
            state_in_fw = [
                tf.placeholder(FLAGS.dtype, [batch_size, lstm_fw.state_size.c], name="c_fw"),
                tf.placeholder(FLAGS.dtype, [batch_size, lstm_fw.state_size.h], name="h_fw")
            ]

        if FLAGS.bi_directional:
            # Backward LSTM Cell and its initial state placeholder
            lstm_bw = rnn_core.LSTMCell(
                num_outputs, state_is_tuple=True, use_peepholes=True,
                # cell_clip=10., num_proj=num_outputs, proj_clip=10.
            )

            if state_in_bw is None:
                state_in_bw = [
                    tf.placeholder(FLAGS.dtype, [batch_size, lstm_bw.state_size.c], name="c_bw"),
                    tf.placeholder(FLAGS.dtype, [batch_size, lstm_bw.state_size.h], name="h_bw")
                ]

            broadcaster = tf.to_int32(input[0, :, 0]) * 0
            sequence_length = tf.shape(input)[0] + broadcaster

            # bi-directional LSTM (forward + backward)
            lstm_outputs, state_out = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw, lstm_bw, input, dtype=FLAGS.dtype, time_major=True, scope=scope,
                sequence_length=sequence_length,
                initial_state_fw=rnn_core.LSTMStateTuple(*state_in_fw),
                initial_state_bw=rnn_core.LSTMStateTuple(*state_in_bw)
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
                lstm_fw, input, dtype=FLAGS.dtype, time_major=True, scope=scope,
                initial_state=rnn_core.LSTMStateTuple(*state_in_fw)
            )

            state_in = state_in_fw

    return AttrDict(
        output    = lstm_outputs,
        state_in  = state_in,
        state_out = state_out
    )

def build_convnet(front_view, params):

    S, B = get_seq_length_batch_size(front_view)
    layers = [flatten(front_view)]

    with tf.name_scope("conv"):

        conv2d = tf.contrib.layers.convolution2d
        max_pool2d = tf.contrib.layers.max_pool2d

        # Batch norm is NOT compatible with LSTM
        # (see Issue: https://github.com/tensorflow/tensorflow/issues/6087)

        conv_options = {"activation_fn": tf.nn.relu, "padding": "VALID"}
        conv_options.update(params)
        layers.append(conv2d(layers[-1], 24, 3, scope="conv-1", stride=2, **conv_options))
        layers.append(conv2d(layers[-1], 32, 3, scope="conv-2", stride=2, **conv_options))
        layers.append(conv2d(layers[-1], 64, 3, scope="conv-3", stride=1, **conv_options))
        layers.append(conv2d(layers[-1], 64, 3, scope="conv-4", **conv_options))
        layers.append(conv2d(layers[-1], 64, 3, scope="conv-5", **conv_options))
        """

        conv_options = {"activation_fn": tf.nn.relu}
        conv_options.update(params)

        layers.append(conv2d(layers[-1], 24, 5, stride=2, scope="conv1", **conv_options))
        layers.append(conv2d(layers[-1], 32, 5, stride=1, scope="conv2", **conv_options))

        layers.append(conv2d(layers[-1], 48, 5, stride=2, scope="conv3", **conv_options))
        layers.append(conv2d(layers[-1], 64, 5, stride=1, scope="conv4", **conv_options))

        layers.append(conv2d(layers[-1], 128, 5, stride=2, scope="conv5", **conv_options))
        layers.append(conv2d(layers[-1], 256, 5, stride=2, scope="conv6", **conv_options))
        """

        # Reshape [seq_len * batch_size, H, W, C] to [seq_len * batch_size, H * W * C]
        layers.append(tf.contrib.layers.flatten(layers[-1]))

    for layer in layers:
        tf.logging.info(layer)

    return layers[-1]

def resnet_block(inputs, num_outputs, activation_fn=tf.nn.relu):
    y = inputs

    y = tf.contrib.layers.fully_connected(
        y, num_outputs, activation_fn=activation_fn,
        normalizer_fn=tf.contrib.layers.batch_norm,
        normalizer_params={'updates_collections': None}
    )

    y = tf.contrib.layers.fully_connected(
        y, num_outputs, activation_fn=None,
        normalizer_fn=tf.contrib.layers.batch_norm,
        normalizer_params={'updates_collections': None}
    )

    y += inputs

    y = activation_fn(y)

    return y

def build_network(input, scope_name, add_summaries=False):
    """
    Builds a 3-layer network conv -> conv -> fc as described
    in the A3C paper. This network is shared by both the policy and value net.
    Args:
    add_summaries: If true, add layer summaries to Tensorboard.
    Returns:
    Final layer activations.
    """

    params = {}
    if FLAGS.batch_norm:
        params.update({
            "normalizer_fn": tf.contrib.layers.batch_norm,
            # "normalizer_params": {'updates_collections': None}
        })

    S, B = get_seq_length_batch_size(input.prev_action)

    layers = []
    other_states = [input.prev_action, input.prev_reward]
    if "OffRoadNav" in FLAGS.game:
        tf.logging.info("building convnet ...")
        layers.append(build_convnet(input.front_view, params))
        other_states += [input.vehicle_state]
    else:
        layers.append(flatten(tf.concat([input.state] + other_states, 2)))

    layers.append(tf.contrib.layers.fully_connected(
        inputs=layers[-1],
        num_outputs=FLAGS.hidden_size,
        activation_fn=tf.nn.tanh,
        scope="state-hidden-dense",
        **params
    ))

    if FLAGS.use_lstm:
        # Flatten convolutions output to fit fully connected layer
        layers.append(deflatten(layers[-1], S, B))

        lstms = []
        # LSTM-1
        # Concatenate encoder's output (i.e. flattened result from conv net)
        # with previous reward (see https://arxiv.org/abs/1611.03673)
        # concat1 = tf.concat(2, [fc, input.prev_reward])
        concat1 = tf.concat([layers[-1]] + other_states, 2)
        lstms.append(LSTM(concat1, FLAGS.hidden_size, scope="LSTM-1"))

        # LSTM-2
        # Concatenate previous output with vehicle_state and prev_action
        # concat2 = tf.concat(2, [fc, lstms[-1].output, vehicle_state, input.prev_action])
        concat2 = tf.concat([lstms[-1].output] + other_states, 2)
        lstms.append(LSTM(concat2, FLAGS.hidden_size, scope="LSTM-2"))

        layers += lstms

        output = layers[-1].output

        state_in, state_out = [], []
        for lstm in lstms:
            state_in  += lstm.state_in
            state_out += lstm.state_out

        # Given a TF variable v, remove the leading scope name and ":0" part
        def _hash_(v):
            return str(v.name).replace(scope_name, "").replace(":0", "")

        keys = [_hash_(v) for v in state_in]

        states = AttrDict(
            inputs = {k: v for k, v in zip(keys, state_in)},
            outputs = {k: v for k, v in zip(keys, state_out)},
        )

    else:

        """
        for i in range(3):
            layers.append(resnet_block(layers[-1], FLAGS.hidden_size, tf.nn.relu))
        """

        layers.append(tf.contrib.layers.fully_connected(
            inputs=layers[-1],
            num_outputs=FLAGS.hidden_size,
            activation_fn=tf.nn.tanh,
            scope="state-hidden-dense-2",
            **params
        ))

        layers.append(deflatten(layers[-1], S, B))

        output = layers[-1]

        states = AttrDict(
            inputs = {},
            outputs = {}
        )

    if add_summaries:
        with tf.name_scope("summaries"):
            summarize_conv_kernels()

    for layer in layers:
        tf.logging.info(layer)

    output = tf_check_numerics(output)

    return output, states

def summarize_conv_kernels():
    conv_kernels = [
        v for v in tf.trainable_variables()
        if "conv" in v.name and "weights" in v.name and "global_net" in v.name
    ]

    if len(conv_kernels) == 0:
        return

    for kernels in conv_kernels:
        grid = put_kernels_on_grid(kernels)
        tf.logging.info("grid of {} has shape = {}".format(kernels.name, grid.get_shape()))
        tf.summary.image(kernels.name.replace(":0", ""), grid)

def policy_network(input, num_outputs, clip_mu=True):

    rank = get_rank(input)

    if rank == 3:
        S, B = get_seq_length_batch_size(input)
        input = flatten(input)

    """
    input = tf.contrib.layers.fully_connected(
        inputs=input,
        num_outputs=256,
        activation_fn=None,
        scope="policy-input-dense")
    """

    # Linear classifiers for mu and sigma
    mu = tf.contrib.layers.fully_connected(
        inputs=input,
        num_outputs=num_outputs,
        activation_fn=None,
        scope="policy-mu")

    sigma = tf.contrib.layers.fully_connected(
        inputs=input,
        num_outputs=num_outputs,
        activation_fn=None,
        scope="policy-sigma")

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

    """
    input = tf.contrib.layers.fully_connected(
        inputs=input,
        num_outputs=FLAGS.hidden_size,
        activation_fn=tf.nn.tanh,
        scope="value-input-dense")
    """

    # This is just linear classifier
    value = tf.contrib.layers.fully_connected(
        inputs=input,
        num_outputs=num_outputs,
        activation_fn=None,
        scope="value-dense")

    if rank == 3:
        value = deflatten(value, S, B)

    return value

def get_lstm_initial_states(lstm_states, batch_size):

    # the shape of s is [batch_size, hidden_size] & batch_size is usually None
    def get_hidden_size(s):
        return s.get_shape().as_list()[1]

    return {
        k: np.zeros((batch_size, get_hidden_size(v)), np.float32)
        for k, v in lstm_states.iteritems()
    }

def get_forward_velocity(state):
    v = state.vehicle_state[..., 4:5]

    # FIXME TensorFlow has bug when dealing with NaN gradient even masked out
    # so I have to make sure abs(v) is not too small
    # v = tf.Print(v, [flatten_all(v)], message="\33[33m before v = \33[0m", summarize=100)
    v = tf.sign(v) * tf.maximum(tf.abs(v), 1e-3)
    # v = tf.Print(v, [flatten_all(v)], message="\33[33m after  v = \33[0m", summarize=100)
    return v

def stochastic_dueling_network(input, value, pi):
    """
    This function wrap advantage, value, policy pi within closure, so that
    the caller doesn't have to pass these as argument anymore
    """

    advantage = advantage_network(input)

    def Q_tilt(action, name, num_samples=FLAGS.num_sdn_samples):
        with tf.name_scope(name):
            # See eq. 13 in ACER
            if len(action.get_shape()) != 4:
                action = action[None, ...]

            adv = tf.squeeze(advantage(action, name="A_action"), 0)

            # TODO Use symmetric low variance sampling !!
            samples = tf.stop_gradient(pi.sample_n(num_samples))
            advs = advantage(samples, "A_sampled", num_samples)
            mean_adv = tf.reduce_mean(advs, axis=0)

            return value + adv - mean_adv

    return Q_tilt

def advantage_network(input):

    rank = get_rank(input)
    if rank == 3:
        S, B = get_seq_length_batch_size(input)

    # Given states
    def advantage(actions, name, num_samples=1):

        with tf.name_scope(name):
            ndims = len(actions.get_shape())
            broadcaster = tf.zeros([num_samples] + [1] * (ndims-1), dtype=FLAGS.dtype)
            input_ = input[None, ...] + broadcaster

            input_with_a = tf.concat([input_, actions], -1)
            input_with_a = flatten_all_leading_axes(input_with_a)

            # 1st fully connected layer
            # fc1 = input_with_a
            fc1 = tf.contrib.layers.fully_connected(
                inputs=input_with_a,
                num_outputs=FLAGS.hidden_size,
                activation_fn=tf.nn.tanh,
                scope="fc1")

            # 2nd fully connected layer that regresses the advantage
            fc2 = tf.contrib.layers.fully_connected(
                inputs=fc1,
                num_outputs=1,
                activation_fn=None,
                scope="fc2")

            output = fc2

            output = tf.reshape(output, [num_samples, -1, 1])

            if rank == 3:
                output = tf.reshape(output, [-1, S, B, 1])

        return output

    return tf.make_template('advantage', advantage)
