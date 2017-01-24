import scipy.io
import numpy as np

class VehicleModel():

    def __init__(self, timestep, noise_level=0., drift=False):
        model = scipy.io.loadmat("../vehicle_modeling/vehicle_model_ABCD.mat")
        self.A = model["A"]
        self.B = model["B"]
        self.C = model["C"]
        self.D = model["D"]
        self.timestep = timestep
        self.noise_level = noise_level

        if not drift:
            self.A[0][0] = 0
            self.C[0][0] = 0
            self.D[0][1] = 0

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
        u = u.reshape(2, -1)
        y = np.dot(self.C, x) + np.dot(self.D, u)
        x = np.dot(self.A, x) + np.dot(self.B, u)
        return y, x

    def predict(self, state, action):
        if self.x is None:
            raise ValueError("self.x is still None. Call reset() first.")

        assert state.shape[0] == 6, "state.shape = {}".format(state.shape)
        assert action.shape[0] == 2, "action.shape = {}".format(action.shape)
        assert self.x.shape[0] == 4, "self.x.shape = {}".format(self.x.shape)

        # y = state[3:6]
        y, self.x = self._predict(self.x, action)
        
        # theta is in radian
        theta = state[2]

        # c, s = np.cos(theta)[0], np.sin(theta)[0]
        # M = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        # delta = np.dot(M, state[3:6]) * self.timestep

        c, s = np.cos(theta), np.sin(theta)
        M = np.array([[c, -s], [s, c]])
        M = np.rollaxis(M, 2, 0)

        V = np.zeros_like(state[0:3])
        for i in range(V.shape[1]):
            V[0:2, i] = np.dot(M[i], state[3:5, i])
            V[2:3, i] += state[5:6, i]

        # dx = v * dt
        delta = V * self.timestep

        # Add some noise using delta * (1 + noise) instead of delta + noise
        delta *= 1 + np.random.rand(*delta.shape) * self.noise_level

        # x2 = x1 + dx
        state[0:3] += delta
        state[3:6] = y

        return state

    def reset(self, state):
        # state: [x, y, theta, x', y', theta']
        # extract the last 3 elements from state
        y0 = state[3:6].reshape(3, -1)
        self.x = np.dot(np.linalg.pinv(self.C), y0)
