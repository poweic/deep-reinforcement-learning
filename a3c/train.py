#!/usr/bin/python
import colored_traceback.always

import unittest
import sys
import os
import cv2
import scipy.io
import numpy as np
import tensorflow as tf
import itertools
import shutil
import threading
import multiprocessing

from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
    sys.path.append(import_path)

from a3c.estimators import ValueEstimator, PolicyEstimator
from a3c.policy_monitor import PolicyMonitor
from worker import Worker
from gym_offroad_nav.envs import OffRoadNavEnv
from gym_offroad_nav.vehicle_model import VehicleModel

tf.flags.DEFINE_string("model_dir", "/tmp/a3c-offroad", "Directory to write Tensorboard summaries and videos to.")
tf.flags.DEFINE_integer("t_max", 10000, "Number of steps before performing an update")
tf.flags.DEFINE_integer("max_global_steps", None, "Stop training after this many steps in the environment. Defaults to running indefinitely.")
tf.flags.DEFINE_integer("eval_every", 300, "Evaluate the policy every N seconds")
tf.flags.DEFINE_boolean("reset", False, "If set, delete the existing model directory and start training from scratch.")
tf.flags.DEFINE_integer("parallelism", 6, "Number of threads to run. If not set we run [num_cpu_cores] threads.")
tf.flags.DEFINE_float("timestep", 0.01, "Maximum forward velocity of vehicle")
tf.flags.DEFINE_float("max_forward_speed", 25, "Maximum forward velocity of vehicle (m/s)")
tf.flags.DEFINE_float("min_forward_speed", 1, "Minimum forward velocity of vehicle (m/s)")
tf.flags.DEFINE_float("max_yaw_rate", 360, "Maximum yaw rate (omega) of vehicle (degree / sec)")
tf.flags.DEFINE_float("min_yaw_rate", -360, "Minimum yaw rate (omega) of vehicle (degree / sec)")

FLAGS = tf.flags.FLAGS

envs = []
disp_img = np.zeros((800, 800*2, 3), dtype=np.uint8)
def imshow4(idx, img):
    global disp_img
    x = idx / 4
    y = idx % 4
    disp_img[x*400:(x+1)*400, y*400:(y+1)*400, :] = np.copy(img)
cv2.imshow4 = imshow4

def make_env(name=None):
    global envs
    vehicle_model = VehicleModel(FLAGS.timestep)
    # rewards = scipy.io.loadmat("data/circle2.mat")["reward"].astype(np.float32) - 100
    rewards = scipy.io.loadmat("data/maze.mat")["reward"].astype(np.float32) - 15
    env = OffRoadNavEnv(rewards, vehicle_model)
    if name is not None:
        envs.append([name, env])
    return env

# Set the number of workers
NUM_WORKERS = multiprocessing.cpu_count()
if FLAGS.parallelism:
    NUM_WORKERS = FLAGS.parallelism

MODEL_DIR = FLAGS.model_dir
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")

# Optionally empty model directory
if FLAGS.reset:
    shutil.rmtree(MODEL_DIR, ignore_errors=True)

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

summary_writer = tf.summary.FileWriter(os.path.join(MODEL_DIR, "train"))

with tf.device("/cpu:0"):

    # Keeps track of the number of updates we've performed
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Global policy and value nets
    with tf.variable_scope("global") as vs:
        rewards = tf.Variable(tf.zeros([1, 40, 40, 1]), name="rewards", trainable=False)
        policy_net = PolicyEstimator()
        value_net = ValueEstimator(policy_net.state, reuse=True)

    # Global step iterator
    global_counter = itertools.count()

    # Create worker graphs
    workers = []
    for worker_id in range(NUM_WORKERS):
        # We only write summaries in one of the workers because they're
        # pretty much identical and writing them on all workers
        # would be a waste of space
        worker_summary_writer = None
        if worker_id == 0:
            worker_summary_writer = summary_writer

        name = "worker_{}".format(worker_id)
        worker = Worker(
            name=name,
            env=make_env(name),
            rewards=rewards,
            policy_net=policy_net,
            value_net=value_net,
            global_counter=global_counter,
            discount_factor = 0.99,
            summary_writer=worker_summary_writer,
            max_global_steps=FLAGS.max_global_steps)

        workers.append(worker)

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.01, max_to_keep=10)

    # Used to occasionally save videos for our policy net
    # and write episode rewards to Tensorboard
    pe = PolicyMonitor(
        env=make_env(),
        rewards=rewards,
        policy_net=policy_net,
        summary_writer=summary_writer,
        saver=saver)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()

    # Copy rewards to GPU variable
    sess.run(rewards.assign(
        make_env().rewards.reshape(rewards.get_shape().as_list())
    ))
    # saver.save(sess, 'models/test')

    # Load a previous checkpoint if it exists
    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest_checkpoint:
        print("Loading model checkpoint: {}".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    # Start worker threads
    worker_threads = []
    for worker in workers:
        worker_fn = lambda: worker.run(sess, coord, FLAGS.t_max)
        t = threading.Thread(target=worker_fn)
        t.start()
        worker_threads.append(t)

    # Start a thread for policy eval task
    '''
    monitor_thread = threading.Thread(target=lambda: pe.continuous_eval(FLAGS.eval_every, sess, coord))
    monitor_thread.start()
    '''

    # Show how agent behaves in envs in main thread
    while True:
        for name, env in envs:
            env._render({"worker": name})
            cv2.imshow("result", disp_img)
            cv2.waitKey(5)

    # Wait for all workers to finish
    coord.join(worker_threads)
