#!/usr/bin/python
import colored_traceback.always

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
import time
from pprint import pprint

from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
    sys.path.append(import_path)

from a3c.estimators import PolicyValueEstimator
from a3c.policy_monitor import PolicyMonitor
from worker import Worker
from gym_offroad_nav.envs import OffRoadNavEnv
from gym_offroad_nav.vehicle_model import VehicleModel

tf.flags.DEFINE_string("model_dir", "/tmp/a3c-offroad", "Directory to write Tensorboard summaries and videos to.")
tf.flags.DEFINE_string("game", "circle4", "Game environment")

tf.flags.DEFINE_integer("t_max", 200, "Number of steps before performing an update")
tf.flags.DEFINE_integer("max_global_steps", None, "Stop training after this many steps in the environment. Defaults to running indefinitely.")
tf.flags.DEFINE_integer("eval_every", 30, "Evaluate the policy every N seconds")
tf.flags.DEFINE_integer("parallelism", 6, "Number of threads to run. If not set we run [num_cpu_cores] threads.")
tf.flags.DEFINE_integer("downsample", 10, "Downsample transitions to reduce sample correlation")
tf.flags.DEFINE_integer("n_agents_per_worker", 32, "Downsample transitions to reduce sample correlation")

tf.flags.DEFINE_boolean("reset", False, "If set, delete the existing model directory and start training from scratch.")
tf.flags.DEFINE_boolean("debug", False, "If set, turn on the debug flag")

tf.flags.DEFINE_float("learning_rate", 1e-2, "Learning rate for policy net and value net")
tf.flags.DEFINE_float("max_gradient", 40, "Threshold for gradient clipping used by tf.clip_by_global_norm")
tf.flags.DEFINE_float("timestep", 0.025, "Simulation timestep")
tf.flags.DEFINE_float("wheelbase", 2.00, "Wheelbase of the vehicle in meters")
tf.flags.DEFINE_float("vehicle_model_noise_level", 0.05, "level of white noise (variance) in vehicle model")
tf.flags.DEFINE_float("entropy_cost_mult", 1e-2, "multiplier used by entropy regularization")
tf.flags.DEFINE_float("discount_factor", 0.99, "discount factor in Markov decision process (MDP)")
tf.flags.DEFINE_float("lambda_", 0.99, "lambda in TD-Lambda (temporal difference learning)")

tf.flags.DEFINE_float("min_mu_vf", 5  / 3.6, "Minimum forward velocity of vehicle (m/s)")
tf.flags.DEFINE_float("max_mu_vf", 40 / 3.6, "Maximum forward velocity of vehicle (m/s)")
tf.flags.DEFINE_float("min_mu_steer", -30 * np.pi / 180, "Minimum steering angle (rad)")
tf.flags.DEFINE_float("max_mu_steer", +30 * np.pi / 180, "Maximum steering angle (rad)")

tf.flags.DEFINE_float("min_sigma_vf", 1  / 3.6, "Minimum variance of forward velocity")
tf.flags.DEFINE_float("max_sigma_vf", 10 / 3.6, "Maximum variance of forward velocity")
tf.flags.DEFINE_float("min_sigma_steer", 1 * np.pi / 180, "Minimum variance of steering angle (rad)")
tf.flags.DEFINE_float("max_sigma_steer", 10 * np.pi / 180, "Maximum variance of steering angle (rad)")

'''
tf.flags.DEFINE_float("max_forward_speed", 2 + 0.0001, "Maximum forward velocity of vehicle (m/s)")
tf.flags.DEFINE_float("min_forward_speed", 2 - 0.0001, "Minimum forward velocity of vehicle (m/s)")
tf.flags.DEFINE_float("max_steering", -16. * np.pi / 180 + 0.0001, "Maximum steering angle (rad)")
tf.flags.DEFINE_float("min_steering", -16. * np.pi / 180 - 0.0001, "Minimum steering angle (rad)")
'''

FLAGS = tf.flags.FLAGS
print "Simulation time for each episode = {} secs (timestep = {})".format(
    FLAGS.t_max * FLAGS.timestep, FLAGS.timestep)
pprint(vars(FLAGS))

W = 400
disp_img = np.zeros((2*W, 2*W*2, 3), dtype=np.uint8)
disp_lock = threading.Lock()
def imshow4(idx, img):
    global disp_img
    x = idx / 4
    y = idx % 4
    with disp_lock:
        disp_img[x*W:(x+1)*W, y*W:(y+1)*W, :] = np.copy(img)

cv2.imshow4 = imshow4

def make_env(name=None):
    vehicle_model = VehicleModel(FLAGS.timestep, FLAGS.vehicle_model_noise_level)
    reward_fn = "data/{}.mat".format(FLAGS.game)
    rewards = scipy.io.loadmat(reward_fn)["reward"].astype(np.float32)
    # rewards -= 100
    # rewards -= 15
    rewards = (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards))
    rewards = (rewards - 0.5) * 10
    # rewards[rewards < 0.1] = -1
    env = OffRoadNavEnv(rewards, vehicle_model, name)
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

with tf.Session() as sess:
    # Keeps track of the number of updates we've performed
    global_step = tf.Variable(0, name="global_step", trainable=False)
    max_return = 0

    # Global step iterator
    global_counter = itertools.count()

    # Create worker graphs
    workers = []
    for worker_id in range(NUM_WORKERS):
        # We only write summaries in one of the workers because they're
        # pretty much identical and writing them on all workers
        # would be a waste of space
        worker_summary_writer = None

        name = "worker_{}".format(worker_id)
        print "Initializing {} ...".format(name)
        worker = Worker(
            name=name,
            env=make_env(name),
            global_counter=global_counter,
            add_summaries=(worker_id == 0),
            n_agents=FLAGS.n_agents_per_worker,
            discount_factor=FLAGS.discount_factor,
            max_global_steps=FLAGS.max_global_steps)

        workers.append(worker)

    summary_writer = tf.summary.FileWriter(os.path.join(MODEL_DIR, "train"), sess.graph)

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.01, max_to_keep=10)

    # Global policy and value nets
    with tf.variable_scope("global") as vs:
        global_net = PolicyValueEstimator()

    workers[0].summary_writer = summary_writer
    for worker in workers:
        worker.set_global_net(global_net)

    # Used to occasionally save videos for our policy net
    # and write episode rewards to Tensorboard
    '''
    pe = PolicyMonitor(
        env=make_env("policy_monitor"),
        global_net=global_net,
        summary_writer=summary_writer,
        saver=saver)
    '''

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()

    # saver.save(sess, 'models/test')

    # Load a previous checkpoint if it exists
    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest_checkpoint:
        print("Loading model checkpoint: {}".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    # Start worker threads
    worker_threads = []
    for i in range(len(workers)):
        worker_fn = lambda j=i: workers[j].run(sess, coord, FLAGS.t_max)
        t = threading.Thread(target=worker_fn)
        time.sleep(0.5)
        t.start()
        worker_threads.append(t)

    # Start a thread for policy eval task
    '''
    monitor_thread = threading.Thread(target=lambda: pe.continuous_eval(FLAGS.eval_every, sess, coord))
    monitor_thread.start()
    '''

    # Show how agent behaves in envs in main thread
    counter = 0
    while True:
        for worker in workers:
            if worker.max_return > max_return:
                max_return = worker.max_return
                print "max_return = \33[93m{}\33[00m".format(max_return)

            worker.env._render({"worker": worker})

        # cv2.imwrite("/share/Research/Yamaha/dirl-exp/{:06d}.png".format(counter), disp_img)
        cv2.imshow("result", disp_img)
        cv2.waitKey(10)
        counter += 1

    # Wait for all workers to finish
    coord.join(worker_threads)
