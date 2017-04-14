#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from drl.optimizers import TrpoOptimizer
from packaging import version
from attrdict import AttrDict
assert version.parse(tf.__version__) > version.parse("1.0.0"), \
    "Tensorflow version >= 1.0.0 required"

lr = 1e-5
input_dim = 64
hidden_size = 64
rnd1 = np.random.randn(input_dim, hidden_size).astype(np.float32)
rnd2 = np.random.randn(hidden_size, input_dim).astype(np.float32)

def build_model(sess, optimizer):

    input = tf.placeholder(tf.float32, [1, input_dim], "input")

    w1 = tf.Variable(rnd1)

    w2 = tf.Variable(rnd2)

    output = tf.nn.tanh(tf.matmul(tf.nn.tanh(tf.matmul(input, w1)), w2))

    loss = tf.reduce_sum((input - output)**2)

    step = optimizer.minimize(loss)

    train = lambda x: sess.run([loss, step], feed_dict={input: x})[0]

    return AttrDict(input=input, train=train, loss=loss)

with tf.Session() as sess:

    model1 = build_model(sess, TrpoOptimizer(lr))
    model2 = build_model(sess, tf.train.AdamOptimizer(lr))

    # initialize
    sess.run(tf.global_variables_initializer())

    x = np.random.randn(1, input_dim).astype(np.float32)

    for itr in range(5000):

        loss1 = model1.train(x)
        loss2 = model2.train(x)

        print "#{:04d} loss (Trpo) = {}, loss (Adam) = {}".format(itr, loss1, loss2)
