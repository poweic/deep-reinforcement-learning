import scipy.io
import numpy as np

class VehicleModel():

    def __init__(self, timestep, noise_level=0.):
        model = scipy.io.loadmat("../vehicle_modeling/vehicle_model_ABCD.mat")
        self.A = model["A"]
        self.B = model["B"]
        self.C = model["C"]
        self.D = model["D"]
        self.timestep = timestep
        self.noise_level = noise_level

        # x = Ax + Bu, y = Cx + Du
        # Turn cm/s, degree/s to m/s and rad/s
        Q = np.diag([100., 180./np.pi]).astype(np.float32)
        Qinv = np.diag([0.01, 0.01, np.pi/180.]).astype(np.float32)
        self.B = self.B.dot(Q)
        self.C = Qinv.dot(self.C)
        self.D = Qinv.dot(self.D).dot(Q)

        # x is the unobservable hidden state, y is the observation
        # u is (v_forward, yaw_rate), y is (vx, vy, w), where
        # vx is v_slide, vy is v_forward, w is yaw rate
        # x' = Ax + Bu (prediction)
        # y' = Cx + Du (measurement)
        self.x = None

    def _predict(self, x, u):
        u = u.reshape(2, 1)
        y = np.dot(self.C, x) + np.dot(self.D, u)
        x = np.dot(self.A, x) + np.dot(self.B, u)
        return y, x

    def predict(self, state, action):
        if self.x is None:
            raise ValueError("self.x is still None. Call reset() first.")

        assert state.shape == (6, 1), "state.shape = {}".format(state.shape)
        assert action.shape == (2, 1), "action.shape = {}".format(action.shape)
        assert self.x.shape == (4, 1), "self.x.shape = {}".format(self.x.shape)

        # y = state[3:6]
        y, self.x = self._predict(self.x, action)
        
        # theta is in radian
        theta = state[2]

        c, s = np.cos(theta)[0], np.sin(theta)[0]
        M = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        # dx = v * dt, where dx is variable delta
        delta = np.dot(M, state[3:6].reshape(3, 1)) * self.timestep

        # Add some noise using delta * (1 + noise) instead of delta + noise
        delta *= 1 + np.random.rand() * self.noise_level

        # x2 = x1 + dx
        state[0:3] += delta
        state[3:6] = y[:]

        return state

    def reset(self, state):
        # state: [x, y, theta, x', y', theta']
        # extract the last 3 elements from state
        y0 = state[3:6].reshape(3, 1)
        self.x = np.dot(np.linalg.pinv(self.C), y0)
