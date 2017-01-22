import numpy as np
import scipy.interpolate as interpolate

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
