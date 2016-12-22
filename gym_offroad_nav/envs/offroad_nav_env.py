import gym
import cv2
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

class OffRoadNavEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, rewards, vehicle_model):
        self.viewer = None

        A = {
            # maximum speed = 2500 cm/s = 90 km/hr
            'v_forward': {'low': 0, 'high': 2500 / 10},

            # maximum yawrate = +- 360 deg/s
            'yawrate': {'low': -360 / 10, 'high': 360 / 10}
        }
        self.A = A

        self.max_a = np.array([A['v_forward']['high'], A['yawrate']['high']]).reshape((1, 2))
        self.min_a = np.array([A['v_forward']['low'], A['yawrate']['low']]).reshape((1, 2))

        # A tf.tensor (or np) containing rewards, we need a constant version and 
        self.rewards = rewards

        self.vehicle_model = vehicle_model

        self.K = 10

        self.state = None

    def _step(self, action):
        ''' Take one step in the environment
        state is the vehicle state, not the full MDP state including history.

        Parameters
        ----------
        action : Numpy array
            The control input for vehicle. [v_forward, yaw_rate]

        Returns
        -------
        Tuple
            A 4-element tuple (state, reward, done, info)
        '''
        self.state = self.vehicle_model.predict(self.state, action)

        # Y forward, X lateral
        # ix = -19, -18, ...0, 1, 20, iy = 0, 1, ... 39
        x, y = self.state[:2, 0]
        ix, iy = self.get_ixiy(x, y)
        done = (ix < -19) or (ix > 20) or (iy < 0) or (iy > 39)

        reward = self._bilinear_reward_lookup(x, y)
        '''
        print "(ix, iy) = ({:3d}, {:3d}) {}".format(
            ix, iy, "\33[92mDone\33[0m" if done else "")
        '''

        # debug info
        info = {}

        return self.state, reward, done, info

    def _get_reward(self, ix, iy):
        linear_idx = (40 - 1 - iy) * 40 + (ix + 19)
        r = self.rewards.flatten()[linear_idx]
        return r

    def get_ixiy(self, x, y):
        ix = int(np.floor(x / 0.5))
        iy = int(np.floor(y / 0.5))
        return ix, iy

    def _bilinear_reward_lookup(self, x, y, debug=True):
        ix, iy = self.get_ixiy(x, y)
        # print "(x, y) = ({}, {}), (ix, iy) = ({}, {})".format(x, y, ix, iy)

        x0 = int(np.clip(ix, -19, 20))
        y0 = int(np.clip(iy, 0, 39))
        x1 = int(np.clip(ix + 1, -19, 20))
        y1 = int(np.clip(iy + 1, 0, 39))

        f00 = self._get_reward(x0, y0)
        f01 = self._get_reward(x0, y1)
        f10 = self._get_reward(x1, y0)
        f11 = self._get_reward(x1, y1)

        xx = x / 0.5 - ix
        yy = y / 0.5 - iy

        w00 = (1.-xx)*(1.-yy)
        w01 = yy*(1.-xx)
        w10 = xx*(1.-yy)
        w11 = xx*yy

        r = f00*w00 + f01*w01 + f10*w10 + f11*w11
        '''
        if debug:
            print "reward[{:6.2f},{:6.2f}] = {:7.2f}*{:4.2f} + {:7.2f}*{:4.2f} + {:7.2f}*{:4.2f} + {:7.2f}*{:4.2f} = {:7.2f}".format(
                x, y, f00, w00, f01, w01, f10, w10, f11, w11, r
            ),
        '''
        return r

    def _reset(self, s0):
        if not hasattr(self, "R"):
            self.R = self.to_image(self.rewards, self.K)
            self.bR = self.to_image(self.debug_bilinear_R(self.K), self.K)

        self.state = s0
        self.bR_temp = np.copy(self.bR)
        self.vehicle_model.reset(s0)
        return s0

    def to_image(self, R, K):
        R = np.clip((R - np.min(R)) * 255. / (np.max(R) - np.min(R)), 0, 255).astype(np.uint8)
        R = cv2.resize(R, (40*K, 40*K), interpolation=cv2.INTER_NEAREST)[..., None]
        R = np.concatenate([R, R, R], axis=2)
        return R

    def debug_bilinear_R(self, K):
        X = np.linspace(-10, 10, num=40*K)
        Y = np.linspace(0, 20, num=40*K)

        bR = np.zeros((40*K, 40*K))

        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                bR[-j, i] = self._bilinear_reward_lookup(x, y, debug=False)

        return bR

    def _render(self, info, mode='human', close=False):

        ix, iy = np.floor(self.state[:2, 0] * self.K / 0.5).astype(np.int)

        bR = np.copy(self.bR)

        cv2.circle(self.bR_temp, (40*self.K/2-1 + ix, 40*self.K-1-iy), 1, (0, 0, 255), 0)
        cv2.circle(bR, (40*self.K/2-1 + ix, 40*self.K-1-iy), 2, (0, 0, 255), 2)

        text = "max return = {:.3f}".format(info["max_return"])
        cv2.putText(bR, text, (0, 370), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        text = "total_return = {:.3f}".format(info["total_return"])
        cv2.putText(bR, text, (0, 390), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

        img = np.concatenate([bR, self.bR_temp], axis=1)

        # cv2.imshow("trajectories", self.bR_temp)
        cv2.imshow(info["worker_name"], img)
        cv2.waitKey(20)
