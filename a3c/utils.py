import numpy as np
import scipy.interpolate as interpolate
import tensorflow as tf
FLAGS = tf.flags.FLAGS

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

def tf_print(x, message):
    if not FLAGS.debug:
        return x

    step = tf.contrib.framework.get_global_step()
    cond = tf.equal(tf.mod(step, 1), 0)
    message = "\33[93m" + message + "\33[0m"
    return tf.cond(cond, lambda: tf.Print(x, [x], message=message, summarize=100), lambda: x)

def multiply(list_a, list_b):
    return [a*b for a, b in zip(list_a, list_b)]

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
