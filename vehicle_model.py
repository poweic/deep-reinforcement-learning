import scipy.io
import numpy as np

class VehicleModel():

    def __init__(self, timestep=0.001):
        model = scipy.io.loadmat("../vehicle_modeling/vehicle_model_ABCD.mat")
        self.A = model["A"]
        self.B = model["B"]
        self.C = model["C"]
        self.D = model["D"]
        self.timestep = timestep

        print "A: {}, B: {}, C: {}, D: {}".format(
            self.A.shape, self.B.shape, self.C.shape, self.D.shape)

        # x is the unobservable hidden state, y is the observation
        # u is (v_forward, yaw_rate), y is (vx, vy, w), where
        # vx is v_slide, vy is v_forward, w is yaw rate
        # x' = Ax + Bu (prediction)
        # y' = Cx + Du (measurement)
        self.x = None

    def _predict(self, x, u):
        y = np.dot(self.C, x) + np.dot(self.D, u)
        x = np.dot(self.A, x) + np.dot(self.B, u)
        return y, x

    def predict(self, state, action):
        # TODO
        # y = state[3:6]
        y, self.x = self._predict(self.x, action)
        
        theta = state[2]
        c, s = np.cos(theta)[0], np.sin(theta)[0]
        M = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

        delta = np.dot(M, state[3:6].reshape(3, 1)) * self.timestep
        # print "(x, y, theta): ({}, {}, {}) => ({}, {}, {})".format(state[0], state[1], state[2], delta[0], delta[1], delta[2])
        state[0:3] += delta
        state[3:6] = y[:]

        return state

    def reset(self, state):
        # state: [x, y, theta, x', y', theta']
        # extract the last 3 elements from state
        y0 = state[3:6].reshape(3, 1)
        self.x = np.dot(np.linalg.pinv(self.C), y0)

