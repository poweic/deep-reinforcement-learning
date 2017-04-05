#!/usr/bin/python
import numpy as np
import tensorflow as tf
import gym

tf.flags.DEFINE_integer("random-seed", None, "Random seed for gym.env and TensorFlow")
tf.flags.DEFINE_boolean("display", True, "If set, no imshow will be called")
tf.flags.DEFINE_string("game", "Humanoid-v1", "Game environment. Ex: Humanoid-v1, OffRoadNav-v0")
tf.flags.DEFINE_integer("seq-length", None, "sequence length used for construct TF graph")
tf.flags.DEFINE_integer("batch-size", None, "batch size used for construct TF graph")
tf.flags.DEFINE_boolean("bi-directional", False, "If set, use bi-directional RNN/LSTM")
tf.flags.DEFINE_string("log-file", None, "log file")
tf.flags.DEFINE_float("learning-rate", 2e-4, "Learning rate for policy net and value net")

import drl.logger
from drl.ac.utils import get_dof, AttrDict, warm_up_env
from drl.ac.models import LSTM, fill_lstm_state_placeholder
warm_up_env()

FLAGS = tf.flags.FLAGS

def build_network(actions, initial_states):

    layers = [actions]

    lstms = []

    lstms.append(LSTM(
        layers[-1], FLAGS.num_states, scope="LSTM-1",
        state_in_fw = [
            tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.num_states], name="c_fw"),
            initial_states
        ]
    ))

    state_in, state_out = [], []
    for lstm in lstms:
        state_in  += lstm.state_in
        state_out += lstm.state_out

    lstm = AttrDict(
        state_in  = state_in,
        state_out = state_out,
        prev_state_out = None
    )

    layers.extend(lstms)

    """
    layers.append(tf.contrib.layers.fully_connected(
        inputs=layers[-1].output,
        num_outputs=FLAGS.num_states,
        activation_fn=tf.nn.tanh
    ))
    """
    
    output = layers[-1].output

    return output, lstm

def compute_mean_squared_loss(states_pred, states_gnd):
    loss = (states_pred - states_gnd) ** 2

    # take mean on axis 1 (batch_size)
    loss = tf.reduce_mean(loss, axis=[1])

    loss = tf.reduce_sum(loss)

    return loss

def gen_data(env, seq_length=10, batch_size=10):

    actions = np.zeros((seq_length, batch_size, FLAGS.num_actions), dtype=np.float32)
    states  = np.zeros((seq_length, batch_size, FLAGS.num_states),  dtype=np.float32)

    for i in range(batch_size):

        state = env.reset()
        for j in range(seq_length):
            states[j, i] = state[:]

            action = env.action_space.sample()
            actions[j, i] = action[:]
            env.step(action)

            state, reward, done, _ = env.step(action)
            env.render()

    return (actions, states)

class NNSID(object):
    def __init__(self, sess):

        self.actions = tf.placeholder(tf.float32, [None, None, FLAGS.num_actions], name="actions")
        self.states = tf.placeholder(tf.float32, [None, None, FLAGS.num_states], name="states")
        self.sess = sess

        states_pred, self.lstm = build_network(
            actions=self.actions[:-1],
            initial_states=self.states[0]
        )

        self.loss = compute_mean_squared_loss(states_pred, self.states[1:])

        self.lr = FLAGS.learning_rate

        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def update(self, actions, states):

        feed_dict = {
            self.actions: actions,
            self.states: states
        }
        batch_size = actions.shape[1]
        fill_lstm_state_placeholder(self.lstm, feed_dict, batch_size)

        _, loss = self.sess.run([self.train_step, self.loss], feed_dict)

        return loss

def train(nnsid, data):

    for i, (actions, states) in enumerate(data):
        loss = nnsid.update(actions, states)
        tf.logging.info("#{:04d}: loss = {}".format(i, loss))

def main():

    env = gym.make("Humanoid-v1")

    actions, states = gen_data(env, seq_length=20, batch_size=1000)

    import scipy.io
    scipy.io.savemat("/home/boton/humanoid_data.mat", dict(actions=actions, states=states))

    return

    with tf.Session() as sess:
        with tf.device('/cpu:0'):
            nnsid = NNSID(sess)

        tf.logging.info("Initializing all TensorFlow variables ...")
        sess.run(tf.global_variables_initializer())

        for epoch in range(100):
            train(nnsid, data)

if __name__ == "__main__":
    main()
