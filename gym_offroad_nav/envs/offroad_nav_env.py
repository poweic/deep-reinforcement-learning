import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

class OffRoadNavEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, rewards, vehicle_model):
        self.viewer = None

        self.action_space = {
            # maximum speed = 2500 cm/s = 90 km/hr
            'v_forward': {'low': 0, 'high': 2500},

            # maximum yawrate = +- 360 deg/s
            'yawrate': {'low': -360, 'high': 360}
        }

        self.max_a = np.array([self.action_space['v_forward']['high'], self.action_space['yawrate']['high']]).reshape((1, 2))
        self.min_a = np.array([self.action_space['v_forward']['low'], self.action_space['yawrate']['low']]).reshape((1, 2))

        # A tf.tensor (or np) containing rewards, we need a constant version and 
        self.rewards = rewards

        self.vehicle_model = vehicle_model

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
            A 4-elements tuple (state, reward, done, info)
        '''
        self.state = self.vehicle_model.predict(self.state, action)

        # Y forward, X lateral
        # ix = -19, -18, ...0, 1, 20, iy = 0, 1, ... 39
        x, y = self.state[:2, 0]
        ix, iy = np.floor(self.state[:2, 0] / 0.5).astype(np.int)
        done = (ix < -19) or (ix > 20) or (iy < 0) or (iy > 39)

        reward = self._bilinear_reward_lookup(x, y, ix, iy)
        print "(ix, iy) = ({}, {}), done = {}".format(
            ix, iy, "\33[33mTrue\33[0m" if done else "False")

        # debug info
        info = {}

        return self.state, reward, done, info

    def _get_reward(self, ix, iy):
        linear_idx = (40 - 1 - iy) * 40 + (ix + 19)
        r = self.rewards.flatten()[linear_idx]
        return r

    def _bilinear_reward_lookup(self, x, y, ix, iy):
        x0 = int(np.clip(ix, -19, 20))
        y0 = int(np.clip(iy, 0, 39))
        x1 = int(np.clip(ix + 1, -19, 20))
        y1 = int(np.clip(iy + 1, 0, 39))

        f00 = self._get_reward(x0, y0)
        f01 = self._get_reward(x0, y1)
        f10 = self._get_reward(x1, y0)
        f11 = self._get_reward(x1, y1)

        w00 = (1.-x)*(1.-y)
        w01 = y*(1.-x)
        w10 = x*(1.-y)
        w11 = x*y
        r = f00*w00 + f01*w01 + f10*w10 + f11*w11
        print "reward[{:.4f}, {:.4f}] = {:.4f}*{:.4f} + {:.4f}*{:.4f} + {:.4f}*{:.4f} + {:.4f}*{:.4f} = {}".format(
            x, y, f00, w00, f01, w01, f10, w10, f11, w11, r
        ),
        return r

    def _reset(self, s0):
        # TODO
        # Clear any internal state variables, and return the initial state of MDP
        self.state = s0
        self.vehicle_model.reset(s0)
        return s0

    def _render(self, mode='human', close=False):
        # TODO
        pass
