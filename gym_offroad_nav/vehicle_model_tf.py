import scipy.io
import tensorflow as tf
import numpy as np
FLAGS = tf.flags.FLAGS
batch_size = FLAGS.batch_size
seq_length = FLAGS.seq_length

# batch_size = 13

class VehicleModelGPU():

    def __init__(self, timestep, noise_level=0., drift=False):
        self.timestep = timestep
        self.noise_level = noise_level
        self.drift = drift
        self.init_ABCD()

        self.state  = tf.placeholder(self.dtype, [6, batch_size])
        self.u      = tf.placeholder(self.dtype, [2, batch_size])
        self.x_0    = tf.placeholder(self.dtype, [4, batch_size])
        self.steps = tf.placeholder(tf.int32, [])

        self.predict_op = self.create_predict_op(
            self.state, self.x_0, self.u, self.steps
        )

        # x is the unobservable hidden state, y is the observation
        # u is (v_forward, yaw_rate), y is (vx, vy, w), where
        # vx is v_slide, vy is v_forward, w is yaw rate
        # x' = Ax + Bu (prediction)
        # y' = Cx + Du (measurement)
        self.x = None

    def init_ABCD(self):

        # Load ABCD, and then store in tf.constant
        model = scipy.io.loadmat("../vehicle_modeling/vehicle_model_ABCD.mat")
        self.A = model["A"]
        self.B = model["B"]
        self.C = model["C"]
        self.D = model["D"]

        self.dtype = tf.float64

        if not self.drift:
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

    def predict(self, state, action, steps, sess=None):
        sess = sess or tf.get_default_session()

        state_final, x_final = sess.run(self.predict_op, feed_dict={
            self.state: state,
            self.x_0: self.x,
            self.u: action,
            self.steps: steps
        })

        self.x = x_final
        return state_final

    def create_predict_op(self, state_0, x_0, u, steps):

        i_0 = steps - 1

        # init tf constant, so we don't create them in for loop
        A = tf.constant(self.A, self.dtype)
        B = tf.constant(self.B, self.dtype)
        C = tf.constant(self.C, self.dtype)
        D = tf.constant(self.D, self.dtype)
        timestep = tf.constant(self.timestep, self.dtype)
        noise_level = tf.constant(self.noise_level, self.dtype)

        def cond(i, state_i, xu_):
            return i >= 0

        def body(i, state_i, x_i):
            y   = tf.matmul(C, x_i) + tf.matmul(D, u)
            x_i = tf.matmul(A, x_i) + tf.matmul(B, u)

            # x_i = tf.Print(x_i, [x_i, y], "x_i, y = ")

            theta = state_i[2]
            # theta = tf.Print(theta, [theta], "theta = ")

            c, s = tf.cos(theta), tf.sin(theta)
            # c = tf.Print(c, [c, s], "c, s = ")

            vx_, vy_ = state_i[3], state_i[4]
            # vx_ = tf.Print(vx_, [vx_, vy_], "vx_, vy_ = ")

            vx = c * vx_ - s * vy_
            vy = s * vx_ + c * vy_
            omega = state_i[5]

            delta = tf.pack([vx, vy, omega], axis=0) * timestep
            # delta = tf.Print(delta, [delta], 'delta = ')
            delta *= 1 + tf.random_uniform(tf.shape(delta), dtype=self.dtype) * noise_level

            state_i = tf.concat(0, [state_i[0:3] + delta, y])

            return i-1, state_i, x_i

        i, state_final, x_final = tf.while_loop(
            cond, body, loop_vars=[i_0, state_0, x_0],
        )

        return state_final, x_final

    def reset(self, state):
        # state: [x, y, theta, x', y', theta']
        # extract the last 3 elements from state
        y0 = state[3:6].reshape(3, -1)
        self.x = np.dot(np.linalg.pinv(self.C), y0)
