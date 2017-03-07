# -*- coding: utf-8 -*-
import os
import cv2
import time
import psutil
import collections
import numpy as np
import scipy.signal
import scipy.interpolate as interpolate
import tensorflow as tf
import inspect
import schedule
from gym import spaces
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

def form_state(env, state, prev_action, prev_reward):
    state = FLAGS.featurize_state(state)

    if "OffRoadNav" in FLAGS.game:
        return AttrDict(
            front_view    = env.get_front_view(state).copy(),
            vehicle_state = state.copy().T,
            prev_action   = prev_action.copy().T,
            prev_reward   = prev_reward.copy().T
        )
    else:
        return AttrDict(
            state = state.reshape(-1, 1).copy().T,
            prev_action = prev_action.copy().T,
            prev_reward = prev_reward.copy().T
        )

class EpisodeStats(object):
    def __init__(self):
        self.episode_lengths = []
        self.episode_rewards = []

        # This is different from OpenAI gym spec because we run multiple agents
        self.episode_rewards_all_agents = []
        self.episode_types = []
        self.initial_reset_timestamp = None
        self.timestamps = []

    def set_initial_timestamp(self):
        if self.initial_reset_timestamp is None:
            self.initial_reset_timestamp = time.time()

    def append(self, length, reward, rewards_all_agent):
        self.episode_lengths.append(length)
        self.episode_rewards_all_agents.append(rewards_all_agent)
        self.episode_rewards.append(reward)
        self.timestamps.append(time.time())

        timesteps = np.sum(self.episode_lengths)

        # set print options
        np.set_printoptions(formatter={'float_kind': lambda x: "{:.2f}".format(x)})
        tf.logging.info(
            "Episode {:05d}: total return: {} [mean = {:.2f}], length = {}, timesteps = \33[93m{}\33[0m".format(
                self.num_episodes(), rewards_all_agent, reward, length, timesteps
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

    def last_n_stats(self):
        N = FLAGS.min_episodes
        last_n = self.episode_rewards[-N:]
        mean, std = np.mean(last_n), np.std(last_n)
        return mean, std, "Last {} episodes' score: {} ± {}".format(N, mean, std)

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

def show_mem_usage():
    process = psutil.Process(os.getpid())
    usage = float(process.memory_info().rss)

    for order, unit in zip([10, 20, 30], ["KB", "MB", "GB"]):
        if usage > 2. ** order:
            usage_v = "{:.3f} {}".format(usage / (2. ** order), unit)

    tf.logging.info("Memory usage: {} ({} Bytes)".format(usage_v, usage))

def save_model_every_nth_minutes(sess):
    mkdir_p(FLAGS.checkpoint_dir)
    schedule.every(FLAGS.save_every_n_minutes).minutes.do(
        lambda: save_model(sess)
    )

def save_model(sess):
    step = sess.run(tf.contrib.framework.get_global_step())
    fn = FLAGS.saver.save(sess, FLAGS.save_path, global_step=step)
    print time.strftime('[%H:%M:%S %Y/%m/%d] model saved to '), fn

def to_radian(deg):
    return deg / 180. * np.pi

def to_degree(rad):
    return rad / np.pi * 180.

def tf_shape(x):
    try:
        return x.get_shape().as_list()
    except:
        return x.get_shape()

def tf_concat(axis, tensors):
    if axis == -1:
        axis = get_rank(tensors[0]) - 1

    return tf.concat(axis, tensors)

def get_rank(x):
    return len(x.get_shape())

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
    mu = tf.pack([mu[..., 0], mu_yawrate], axis=-1)
    return mu

def y2s(mu, v):
    # Extract yawrwate from input (2nd column), turn it to steer, and
    # concatenate (pack) it back
    mu_steer = yawrate_to_steer(mu[..., 1:], v)[..., 0]
    mu = tf.pack([mu[..., 0], mu_steer], axis=-1)
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

def make_train_op(local_estimator, global_estimator):
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

    return global_estimator.optimizer.apply_gradients(
        local_grads_global_vars)

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
        a = tf.constant(alpha, tf.float32)
        b = 1. - a
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

def deflatten(x, n, m=-1): # de-flatten the first axes
    try:
        return x.reshape((n, -1,) + x.shape[1:])
    except:
        shape = tf.concat(0, [[n], [m], x.get_shape().as_list()[1:]])
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

    (grid_Y, grid_X) = factorize(kernel.get_shape()[3].value)
    # print 'grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X)

    # pad X and Y
    x1 = tf.pad(kernel, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel.get_shape()[0] + 2 * pad
    X = kernel.get_shape()[1] + 2 * pad

    channels = kernel.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, channels]))

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, channels]))

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scaling to [0, 255] is not necessary for tensorboard
    return x7

def compute_mean_steering_angle(reward):
    from ac.utils import to_image
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
