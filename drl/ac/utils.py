# -*- coding: utf-8 -*-
import os
import time
import zlib
import psutil
import numpy as np
import scipy.signal
import scipy.interpolate as interpolate
import tensorflow as tf
import inspect
import schedule
import cv2
import gym
import sys
import cPickle
from pprint import pprint
from numbers import Number
from collections import Set, Mapping, deque
from gym import spaces
import itertools
FLAGS = tf.flags.FLAGS

def get_dof(space):
    t = type(space)
    if t == spaces.tuple_space.Tuple:
        return sum([get_dof(s) for s in space.spaces])
    else:
        try: # if Discrete
            return space.n
        except: # else Box
            return np.prod(space.shape)

def tf_const(c):
    return tf.constant(c, dtype=FLAGS.dtype)

def initialize_env_related_flags(env):
    if FLAGS.random_seed is not None:
        env.seed(FLAGS.random_seed)

    FLAGS.action_space = env.action_space
    FLAGS.num_actions = get_dof(env.action_space)

    FLAGS.observation_space = env.observation_space
    FLAGS.num_states = get_dof(env.observation_space)

    FLAGS.featurize_state, FLAGS.num_states = state_featurizer(env)

def warm_up_env():
    # DEPRECATED because I move monitor to the top of train.py
    # If tf.contribs is imported earlier than the first env.render(), contribs
    # will mess up some resource needed by env.render(). YOU MUST call this
    # function before using tf.contribs to warm up.

    env = gym.make(FLAGS.game)
    initialize_env_related_flags(env)
    env.close()

def check_unused_local_variables(grads_and_vars):

    none_grads = [
        (g, v) for g, v in grads_and_vars
        if tf.get_variable_scope().name in v.name and g is None
    ]

    if len(none_grads) > 0:
        tf.logging.warn("\33[33m Detected None in grads_and_vars: \33[0m")
        pprint([(g, v.name) for g, v in none_grads])

        tf.logging.warn("\33[33m All trainable variables:\33[0m")
        pprint([v.name for v in tf.trainable_variables()])
        import ipdb; ipdb.set_trace()

def pretty_float(fmt):
    fmt = fmt.replace("%f", "{:+8.3f}")
    return fmt

def form_state(env_state, prev_action, prev_reward, hidden_states, steps=None):
    env_state = FLAGS.featurize_state(env_state)

    state = AttrDict(
        steps = np.array(steps).reshape(-1, 1),
        prev_action = prev_action.copy().T,
        prev_reward = prev_reward.copy().T
    )

    if hidden_states is not None:
        state.update(hidden_states)

    if "OffRoadNav" not in FLAGS.game:
        env_state = {"state": env_state.reshape(-1, 1).copy().T}

    state.update(env_state)

    return state

class EpisodeStats(object):
    def __init__(self):
        self.episode_lengths = []
        self.episode_rewards = []

        # This is different from OpenAI gym spec because we run multiple agents
        self.episode_rewards_all_agents = []
        self.episode_types = []
        self.initial_reset_timestamp = None
        self.timestamps = []

        # A thread-safe get-and-increment counter
        self.counter1 = itertools.count()
        self.counter2 = itertools.count()

    def set_initial_timestamp(self):
        if self.initial_reset_timestamp is None:
            self.initial_reset_timestamp = time.time()

    def append(self, length, reward, rewards_all_agent):

        self.episode_lengths.append(length)
        self.episode_rewards_all_agents.append(rewards_all_agent)
        self.episode_rewards.append(reward)
        self.timestamps.append(time.time())

        timesteps = np.sum(self.episode_lengths)

        if self.counter1.next() % FLAGS.log_episode_stats_every_nth == 0:
            # set print options
            np.set_printoptions(
                formatter={'float_kind': lambda x: "{:.2f}".format(x)},
                linewidth=1000
            )

            fmt = "Episode {:05d} [{}]: return: {} [mean = {:.2f}], length = {}"
            tf.logging.info(fmt.format(
                self.num_episodes(),timesteps,rewards_all_agent,reward,length
            ))

            # reset print options
            np.set_printoptions()

    def __str__(self):

        stats = zip(
            self.episode_lengths, self.episode_rewards,
            self.timestamps, self.episode_rewards_all_agents
        )

        # set print options
        np.set_printoptions(
            linewidth=1000,
            formatter={'float_kind': lambda x: "{:.5f}".format(x)}
        )

        HEADER = "{}\t{}\t{}\t{}\t{}\t{}\n"
        s = HEADER.format("Episode", "Length", "Reward", "Timestamp", "NumAgents", "Rewards")

        ROW = "{}\t{}\t{:.5f}\t{}\t{}\t{}\n"
        for i, (l, r, t, rs) in enumerate(stats):
            s += ROW.format(i, l, r, t, len(rs), rs)

        # reset print options
        np.set_printoptions()

        return s

    def last_n_stats(self, N=None):
        if N is None:
            N = FLAGS.min_episodes

        if self.num_episodes() == 0:
            mean, std = 0, 0
        else:
            last_n = self.episode_rewards[-N:]
            mean, std = np.mean(last_n), np.std(last_n)

        if self.counter2.next() % FLAGS.log_episode_stats_every_nth == 0:
            fmt = "\33[33mLast {} episodes' score: {:.4f} ± {:.4f}\33[0m"
            tf.logging.info(fmt.format(N, mean, std))

        return mean, std

    def num_episodes(self):
        return len(self.episode_lengths)

    def summary(self):
        _, _, s = self.last_n_stats()
        s += "\nTotal returns: {}".format(self.episode_rewards)
        s += "\nEpisode lengths: {}".format(self.episode_lengths)
        s += "\ninitial_reset_timestamp: {}".format(self.initial_reset_timestamp)
        s += "\ntimestamps: {}".format(self.timestamps)
        return s

def state_featurizer(env):

    if FLAGS.game != "MountainCarContinuous-v0":
        return lambda x: x, FLAGS.num_states

    import sklearn.pipeline
    import sklearn.preprocessing
    from sklearn.kernel_approximation import RBFSampler

    env.reset()

    # Feature Preprocessing: Normalize to zero mean and unit variance
    # We use a few samples from the observation space to do this
    observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(observation_examples)

    # Used to converte a state to a featurizes represenation.
    # We use RBF kernels with different variances to cover different parts of the space
    featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
    ])
    featurizer.fit(scaler.transform(observation_examples))

    def featurize_state(state):
        """ Returns the featurized representation for a state.
        """
        scaled = scaler.transform([state.squeeze()])
        featurized = featurizer.transform(scaled)
        return featurized[0]

    return featurize_state, 400

def mkdir_p(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    return dirname

def show_mem_usage(x=None, subject=None):

    if x is None:
        process = psutil.Process(os.getpid())
        num_bytes = float(process.memory_info().rss)
        subject = "this process"
    else:
        num_bytes = sizeof(x)
        if subject is None:
            subject = varName(x)

    for order, unit in zip([10, 20, 30], ["KB", "MB", "GB"]):
        if num_bytes > 2. ** order:
            usage = "{:.3f} {}".format(num_bytes / (2. ** order), unit)

    tf.logging.info("Memory usage of {}: {} ({} Bytes)".format(
        subject, usage, num_bytes
    ))

def save_model():
    step = FLAGS.sess.run(tf.contrib.framework.get_global_step())
    fn = FLAGS.saver.save(FLAGS.sess, FLAGS.save_path, global_step=step)
    print time.strftime('[%H:%M:%S %Y/%m/%d] model saved to '), fn

def write_statistics():
    # also print experiment configuration in MATLAB parseable JSON
    cfg = "'" + repr(FLAGS.exp_config)[1:-1].replace("'", '"') + "'\n"
    print >> open(FLAGS.stats_file, 'w'), cfg, FLAGS.stats

def to_radian(deg):
    return deg / 180. * np.pi

def to_degree(rad):
    return rad / np.pi * 180.

def tf_shape(x):
    try:
        return x.get_shape().as_list()
    except:
        return x.get_shape()

def get_rank(x):
    try:
        return x.ndim
    except:
        return len(x.get_shape())

def same_rank(x, y):
    return get_rank(x) == get_rank(y)

def get_seq_length_batch_size(x):
    shape = tf.shape(x)
    S = shape[0] if FLAGS.seq_length is None else FLAGS.seq_length
    B = shape[1] if FLAGS.batch_size is None else FLAGS.batch_size
    return S, B

def get_var_list_wrt(loss):
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    grads_and_vars = optimizer.compute_gradients(loss)
    var_list = [v for g, v in grads_and_vars if g is not None]
    return var_list

def s2y(mu, v):
    # Extract steer from input (2nd column), turn it to yawrate, and
    # concatenate (pack) it back
    mu_yawrate = steer_to_yawrate(mu[..., 1:], v)[..., 0]
    mu = tf.stack([mu[..., 0], mu_yawrate], axis=-1)
    return mu

def y2s(mu, v):
    # Extract yawrwate from input (2nd column), turn it to steer, and
    # concatenate (pack) it back
    mu_steer = yawrate_to_steer(mu[..., 1:], v)[..., 0]
    mu = tf.stack([mu[..., 0], mu_steer], axis=-1)
    return mu

def steer_to_yawrate(steer, v):
    '''
    Use Ackerman formula to compute yawrate from steering angle and forward
    velocity v:
      r = wheelbase / tan(steer)
      omega = v / r
    '''
    # assert steer.get_shape().as_list() == v.get_shape().as_list(), "steer = {}, v = {}".format(steer, v)
    # print steer, v
    return v * tf.tan(steer) / FLAGS.wheelbase

def yawrate_to_steer(omega, v):
    '''
    Use Ackerman formula to compute steering angle from yawrate and forward
    velocity v:
      r = v / omega
      steer = atan(wheelbase / r)
    '''
    # assert omega.get_shape().as_list() == v.get_shape().as_list()
    return tf.atan(FLAGS.wheelbase * omega / v) 

def make_train_op(local_estimator, global_estimator, opt=None):
    """
    Creates an op that applies local gradients to the global variables.
    """
    # Get local gradients and global variables
    local_grads, _ = zip(*local_estimator.grads_and_vars)
    _, global_vars = zip(*global_estimator.grads_and_vars)

    # Clip gradients
    max_grad = FLAGS.max_gradient
    local_grads, _ = tf.clip_by_global_norm(local_grads, max_grad)

    # Zip clipped local grads with global variables
    local_grads_global_vars = list(zip(local_grads, global_vars))
    # global_step = tf.contrib.framework.get_global_step()

    if opt is None:
        opt = global_estimator.optimizer

    return opt.apply_gradients(local_grads_global_vars)

def make_copy_params_op(v1_list, v2_list, alpha=0.):
    """
    Creates an operation that copies parameters from variable in v1_list to variables in v2_list.
    The ordering of the variables in the lists must be identical.
    """
    v1_list = list(sorted(v1_list, key=lambda v: v.name))
    v2_list = list(sorted(v2_list, key=lambda v: v.name))

    '''
    for v1, v2 in zip(v1_list, v2_list):
        print "\n\n================================================="
        print "{}: {}".format(v1.name, v1)
        print "{}: {}".format(v2.name, v2)
    print "\33[93mlen(v1_list) = {}, len(v2_list) = {}\33[0m".format(len(v1_list), len(v2_list))
    '''

    if alpha == 0.:
        return tf.group(*[v2.assign(v1) for v1, v2 in zip(v1_list, v2_list)])
    else:
        a = alpha
        b = tf_const(1.) - alpha
        return tf.group(*[v2.assign(a * v2 + b * v1) for v1, v2 in zip(v1_list, v2_list)])

def discount(x, gamma):
    # if x.ndim == 1:
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
    '''
    else:
        return scipy.signal.lfilter([1], [1, -gamma], x[:, ::-1], axis=1)[:, ::-1]
    '''

def flatten_all(x):
    return tf.reshape(x, [-1])

def flatten(x): # flatten the first 2 axes
    try:
        return x.reshape((-1,) + x.shape[2:])
    except:
        return tf.reshape(x, [-1] + x.get_shape().as_list()[2:])

# TODO merge all flatten utilities into one single function
def flatten_all_leading_axes(x):
    return tf.reshape(x, [-1, x.get_shape()[-1].value])

def deflatten(x, n, m=-1): # de-flatten the first axes
    try:
        return x.reshape((n, -1,) + x.shape[1:])
    except:
        shape = tf.concat([[n], [m], x.get_shape().as_list()[1:]], 0)
        return tf.reshape(x, shape)

def inverse_transform_sampling_2d(data, n_samples, version=2):

    M, N = data.shape

    # Construct horizontal cumulative sum for inverse transform sampling
    cumsum_x = np.zeros(N+1)
    cumsum_x[1:] = np.cumsum(np.sum(data, axis=0))
    cumsum_x /= cumsum_x[-1]
    inv_x_cdf = interpolate.interp1d(cumsum_x, np.arange(N+1))

    # Construct vertical cumulative sum for inverse transform sampling
    cumsum_y = np.zeros((M+1, N))
    cumsum_y[1:, :] = np.cumsum(data, axis=0)
    cumsum_y /= cumsum_y[-1, :]
    inv_y_cdf = np.array([interpolate.interp1d(cumsum_y[:, i], np.arange(M+1)) for i in range(N)])

    if version == 2:
        # Version 2 (about 2 times faster)
        rx = np.random.rand(n_samples, 1)
        ry = np.random.rand(n_samples, 1)
        
        L = (rx <= cumsum_x.reshape(1, -1))
        x_indices = [l.index(True)-1 for l in L.tolist()]
        xs = inv_x_cdf(rx)
        ys = np.array([inv_y_cdf[x_ind](r) for x_ind,r in zip(x_indices, ry)])

    else:
        # Version 1
        xs = np.zeros(n_samples)
        ys = np.zeros(n_samples)
        rand_ = np.random.rand(n_samples, 2)
        for i, (rx, ry) in enumerate(rand_):
            x_ind = (rx > cumsum_x).tolist().index(False) - 1
            xs[i] = inv_x_cdf(rx)
            ys[i] = inv_y_cdf[x_ind](ry)

    return xs.squeeze(), ys.squeeze()

def clip(x, min_v, max_v):
    return tf.maximum(tf.minimum(x, max_v), min_v)

def softclip(x, min_v, max_v):
    return (max_v + min_v) / 2 + tf.nn.tanh(x) * (max_v - min_v) / 2

def varName(var):
    lcls = inspect.stack()[2][0].f_locals
    for name in lcls:
        if id(var) == id(lcls[name]):
            return name
    return None

def tf_check_numerics(x, message=None):
    if not FLAGS.debug:
        return x

    if message is None:
        message = varName(x)
        if message is None:
            return x
        else:
            message += " = "

    return tf.check_numerics(x, message)

def tf_print(x, message=None, cond2=None, flat_=True):
    if not FLAGS.debug:
        return x

    step = tf.contrib.framework.get_global_step()

    if step is None:
        return x

    if message is None:
        message = varName(x)
        if message is None:
            return x
        else:
            message += " = "

    cond = tf.equal(tf.mod(step, 1), 0)
    if cond2 is not None:
        cond = tf.logical_and(cond, cond2)

    message = "\33[93m" + message + "\33[0m"
    return tf.cond(cond, lambda: tf.Print(x, [
        x if not flat_ else flatten_all(x)
    ], message=message, summarize=1000), lambda: x)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def put_kernels_on_grid(kernel, pad = 1):

    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.
    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)
    Return:
      Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    '''
    # get shape of the grid. NumKernels == grid_Y * grid_X
    def factorize(n):
        for i in range(int(np.sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1: print('Who would enter a prime number of filters?')
                return (i, int(n / i))

    shape = kernel.get_shape().as_list()
    kernel = tf.reshape(kernel, [
        shape[0], shape[1], 1, shape[2] * shape[3]
    ])

    (grid_Y, grid_X) = factorize(kernel.get_shape()[3].value)
    # print 'grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X)

    # Normalize each filter to [0, 1]
    """
    x_max = tf.reduce_max(kernel, axis=[0, 1], keep_dims=True)
    x_min = tf.reduce_min(kernel, axis=[0, 1], keep_dims=True)
    kernel = (kernel - x_min) / (x_max - x_min)
    """

    # pad X and Y
    x1 = tf.pad(kernel, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel.get_shape()[0] + 2 * pad
    X = kernel.get_shape()[1] + 2 * pad

    channels = kernel.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, channels]))

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scaling to [0, 255] is not necessary for tensorboard
    return x7

def to_image(R, K, interpolation=cv2.INTER_NEAREST):
    R = normalize(R)
    R = cv2.resize(R, (R.shape[1] * K, R.shape[0] * K), interpolation=interpolation)[..., None]
    R = np.concatenate([R, R, R], axis=2)
    return R

def compute_mean_steering_angle(reward):
    rimg = to_image(reward, 20)
    cv2.imshow("reward", rimg)

    H, W = reward.shape
    yv, xv = np.meshgrid(range(H), range(W))
    xv = xv.astype(np.float32)
    yv = yv.astype(np.float32)

    theta = np.arctan((yv - W / 2 + 0.5) / (H - xv))

    mean_steer = np.mean(theta * reward)

    print "mean_steer = {} degree".format(mean_steer / np.pi * 180)

    cv2.imshow("theta", np.abs(theta))
    cv2.waitKey(0)
    sys.exit()

def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape,minval=0,maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        k = tf.shape(logits)[-1]
        #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
        y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
        y = tf.stop_gradient(y_hard - y) + y

    return y

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def sizeof(obj_0):
    try: # Python 2
        zero_depth_bases = (basestring, Number, xrange, bytearray)
        iteritems = 'iteritems'
    except NameError: # Python 3
        zero_depth_bases = (str, bytes, Number, range, bytearray)
        iteritems = 'items'

    """Recursively iterate to sum size of object & members."""
    def inner(obj, _seen_ids = set()):
        obj_id = id(obj)
        if obj_id in _seen_ids:
            return 0
        _seen_ids.add(obj_id)
        size = sys.getsizeof(obj)
        if isinstance(obj, zero_depth_bases):
            pass # bypass remaining control flow and return
        elif isinstance(obj, (tuple, list, Set, deque)):
            size += sum(inner(i) for i in obj)
        elif isinstance(obj, Mapping) or hasattr(obj, iteritems):
            size += sum(inner(k) + inner(v) for k, v in getattr(obj, iteritems)())
        # Check for custom object instances - may subclass above too
        if hasattr(obj, '__dict__'):
            size += inner(vars(obj))
        if hasattr(obj, '__slots__'): # can have __slots__ with __dict__
            size += sum(inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s))
        return size

    return inner(obj_0)

def compress_decompress_dict(rollout, fn):
    return AttrDict({
        k: fn(v.tobytes()) if isinstance(v, np.ndarray) else v
        for k, v in rollout.iteritems()
    })

class Timer(object):
    def __init__(self, message, maxlen=100000):
        self.timer = deque(maxlen=maxlen)
        self.message = message

        # A thread-safe get-and-increment counter
        self.counter = itertools.count()

    def tic(self):
        self.t = time.time()

    def toc(self):
        self.timer.append(time.time() - self.t)

        if self.counter.next() % (self.timer.maxlen / 10) == 0:
            tf.logging.info("average time of \33[93m{} = {:.2f} ms\33[0m".format(
                self.message, np.mean(self.timer) * 1000
            ))

class ReplayBuffer(deque):

    def __init__(self, maxlen=None):
        super(ReplayBuffer, self).__init__(maxlen=maxlen)

        # keep last 1000 compress, decompress time for profiling purpose
        self.timer = AttrDict(
            compress = Timer("compress"),
            decompress = Timer("decompress")
        )

        # A thread-safe get-and-increment counter
        self.counter = itertools.count()

    def compress(self, item):
        self.timer.compress.tic()
        item = zlib.compress(cPickle.dumps(item, protocol=cPickle.HIGHEST_PROTOCOL))
        self.timer.compress.toc()
        return item

    def decompress(self, item):
        self.timer.decompress.tic()
        item = cPickle.loads(zlib.decompress(item))
        self.timer.decompress.toc()
        return item

    def append(self, item):

        if FLAGS.compress:
            item = self.compress(item)

            if self.counter.next() % self.maxlen == 0:
                show_mem_usage(self, "replay buffer")

        super(ReplayBuffer, self).append(item)

    def __getitem__(self, key):
        item = super(ReplayBuffer, self).__getitem__(key)

        if FLAGS.compress:
            item = self.decompress(item)

        return item

def reduce_seq_batch_dim(value, value_sur):
    assert len(value.get_shape()) == 3
    assert len(value_sur.get_shape()) == 3

    # for sequence dimension (0-th axis), we use reduce_sum for surrogate value
    # but reduce_mean for easier human debugging
    value     = tf.reduce_mean(value, axis=0)
    value_sur = tf.reduce_sum(value_sur, axis=0)

    # Reduce the rest axis to a single number
    value     = tf.reduce_mean(value)
    value_sur = tf.reduce_mean(value_sur)

    return value, value_sur

def to_feed_dict(self, state):

    feed_dict = {
        self.state[k]: state[k]
        if same_rank(self.state[k], state[k]) else state[k][None, ...]
        for k in state.keys()
    }

    return feed_dict

def get_regularizable_vars():
    # skip bias and convolution kernels
    vscope = tf.get_variable_scope().name
    weights = [
        v for v in tf.trainable_variables()
        if vscope in v.name and
        ("W" in v.name or "weights" in v.name) and
        "conv" not in v.name
    ]
    return weights

def l1_loss(weights):
    return tf.add_n([tf.reduce_sum(tf.abs(w)) for w in weights])

def l2_loss(weights):
    return tf.add_n([tf.reduce_sum(w * w) for w in weights])

def compute_gradients_with_checks(optimizer, loss, var_list=None):

    tf.logging.info("Computing gradients ...")

    grads_and_vars = optimizer.compute_gradients(loss, var_list=var_list)

    check_unused_local_variables(grads_and_vars)

    # remove none grads
    grads_and_vars = [(g,v) for g,v in grads_and_vars if g is not None]

    # add check_numerics
    grads_and_vars = [
        (tf.check_numerics(g, message=str(v.name)), v)
        for g, v in grads_and_vars
    ]

    global_norm = tf.global_norm([g for g, v in grads_and_vars])

    return grads_and_vars, global_norm
