#!/usr/bin/python
import colored_traceback.always

import os
import cv2
import scipy.io
import numpy as np
import tensorflow as tf
import itertools
import shutil
import threading
import time
import schedule
from pprint import pprint

from a3c.estimators import PolicyValueEstimator
from a3c.policy_monitor import PolicyMonitor
from a3c.worker import Worker
from gym_offroad_nav.envs import OffRoadNavEnv
from gym_offroad_nav.vehicle_model import VehicleModel

tf.flags.DEFINE_string("model_dir", "/Data3/a3c-offroad/", "Directory to write Tensorboard summaries and models to.")
tf.flags.DEFINE_string("game", "line", "Game environment")

tf.flags.DEFINE_integer("max_global_steps", None, "Stop training after this many steps in the environment. Defaults to running indefinitely.")
tf.flags.DEFINE_integer("eval_every", 30, "Evaluate the policy every N seconds")
tf.flags.DEFINE_integer("parallelism", 1, "Number of threads to run. If not set we run [num_cpu_cores] threads.")
tf.flags.DEFINE_integer("downsample", 5, "Downsample transitions to reduce sample correlation")
tf.flags.DEFINE_integer("n_agents_per_worker", 16, "Downsample transitions to reduce sample correlation")
tf.flags.DEFINE_integer("save_every_n_minutes", 10, "Save model every N minutes")

tf.flags.DEFINE_boolean("drift", False, "If set, turn on drift")
tf.flags.DEFINE_boolean("reset", False, "If set, delete the existing model directory and start training from scratch.")
tf.flags.DEFINE_boolean("debug", False, "If set, turn on the debug flag")

tf.flags.DEFINE_float("t_max", 5, "Maximum elasped time per simulation (in seconds)")
tf.flags.DEFINE_float("command_freq", 5, "How frequent we send command to vehicle (in Hz)")

tf.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate for policy net and value net")
tf.flags.DEFINE_float("l2_reg", 1e-4, "L2 regularization multiplier")
tf.flags.DEFINE_float("max_gradient", 40, "Threshold for gradient clipping used by tf.clip_by_global_norm")
tf.flags.DEFINE_float("timestep", 0.0001, "Simulation timestep")
tf.flags.DEFINE_float("wheelbase", 2.00, "Wheelbase of the vehicle in meters")
tf.flags.DEFINE_float("vehicle_model_noise_level", 0.1, "level of white noise (variance) in vehicle model")
tf.flags.DEFINE_float("entropy_cost_mult", 1e-3, "multiplier used by entropy regularization")
tf.flags.DEFINE_float("discount_factor", 0.99, "discount factor in Markov decision process (MDP)")
tf.flags.DEFINE_float("lambda_", 0.90, "lambda in TD-Lambda (temporal difference learning)")

tf.flags.DEFINE_float("min_mu_vf", 10. / 3.6, "Minimum forward velocity of vehicle (m/s)")
tf.flags.DEFINE_float("max_mu_vf", 11. / 3.6, "Maximum forward velocity of vehicle (m/s)")
tf.flags.DEFINE_float("min_mu_steer", -30 * np.pi / 180, "Minimum steering angle (rad)")
tf.flags.DEFINE_float("max_mu_steer", +30 * np.pi / 180, "Maximum steering angle (rad)")

tf.flags.DEFINE_float("min_sigma_vf", 3. / 3.6, "Minimum variance of forward velocity")
tf.flags.DEFINE_float("max_sigma_vf", 7. / 3.6, "Maximum variance of forward velocity")
tf.flags.DEFINE_float("min_sigma_steer", 3. * np.pi / 180, "Minimum variance of steering angle (rad)")
tf.flags.DEFINE_float("max_sigma_steer", 7 * np.pi / 180, "Maximum variance of steering angle (rad)")
'''
tf.flags.DEFINE_float("min_mu_vf", 7  / 3.6 - 0.0001, "Minimum forward velocity of vehicle (m/s)")
tf.flags.DEFINE_float("max_mu_vf", 7  / 3.6 + 0.0001, "Maximum forward velocity of vehicle (m/s)")
tf.flags.DEFINE_float("min_mu_steer", -30 * np.pi / 180, "Minimum steering angle (rad)")
tf.flags.DEFINE_float("max_mu_steer", +30 * np.pi / 180, "Maximum steering angle (rad)")

tf.flags.DEFINE_float("min_sigma_vf", 0.05 / 3.6 - 0.001, "Minimum variance of forward velocity")
tf.flags.DEFINE_float("max_sigma_vf", 0.05 / 3.6 + 0.001, "Maximum variance of forward velocity")
tf.flags.DEFINE_float("min_sigma_steer", 1 * np.pi / 180 - 0.001, "Minimum variance of steering angle (rad)")
tf.flags.DEFINE_float("max_sigma_steer", 15 * np.pi / 180 + 0.001, "Maximum variance of steering angle (rad)")
'''

# Parse command line arguments, add some additional flags, and print them out
FLAGS = tf.flags.FLAGS
FLAGS.checkpoint_dir = FLAGS.model_dir + "/checkpoints/" + FLAGS.game
FLAGS.save_path = FLAGS.checkpoint_dir + "/model"
pprint(FLAGS.__flags)

W = 400
disp_img = np.zeros((2*W, 2*W*2, 3), dtype=np.uint8)
disp_lock = threading.Lock()
def imshow4(idx, img):
    global disp_img
    x = idx / 4
    y = idx % 4
    with disp_lock:
        if img.shape[0] == W:
            disp_img[x*W:(x+1)*W, y*W:(y+1)*W, :] = np.copy(img)
        else:
            disp_img[x*W:(x+2)*W, y*W:(y+2)*W, :] = np.copy(img)

cv2.imshow4 = imshow4

def make_env():
    vehicle_model = VehicleModel(FLAGS.timestep, FLAGS.vehicle_model_noise_level)
    reward_fn = "data/{}.mat".format(FLAGS.game)
    rewards = scipy.io.loadmat(reward_fn)["reward"].astype(np.float32)
    # rewards -= 100
    # rewards -= 15
    rewards = (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards))
    rewards = (rewards - 0.6) * 2
    # rewards[rewards < 0.1] = -1
    env = OffRoadNavEnv(rewards, vehicle_model)
    return env

# Optionally empty model directory
if FLAGS.reset:
    shutil.rmtree(FLAGS.model_dir, ignore_errors=True)

def mkdir_p(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def save_model_every_nth_minutes(sess, saver):

    mkdir_p(FLAGS.checkpoint_dir)

    def save_model():
        step = sess.run(tf.contrib.framework.get_global_step())
        fn = saver.save(sess, FLAGS.save_path, global_step=step)
        print time.strftime('[%H:%M:%S %Y/%m/%d] model saved to '), fn

    schedule.every(FLAGS.save_every_n_minutes).minutes.do(save_model)

with tf.Session() as sess:
    # Keeps track of the number of updates we've performed
    global_step = tf.Variable(0, name="global_step", trainable=False)
    max_return = 0

    # Global step iterator
    global_counter = itertools.count()

    # Create worker graphs
    workers = []
    for i in range(FLAGS.parallelism):
        name = "worker_%d" % i
        print "Initializing {} ...".format(name)

        worker = Worker(
            name=name,
            env=make_env(),
            global_counter=global_counter,
            add_summaries=(i == 0),
            n_agents=FLAGS.n_agents_per_worker,
            discount_factor=FLAGS.discount_factor,
            max_global_steps=FLAGS.max_global_steps)

        workers.append(worker)

    summary_dir = os.path.join(FLAGS.model_dir, "train")
    summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)

    # Global policy and value nets
    with tf.variable_scope("global") as vs:
        global_net = PolicyValueEstimator()

    workers[0].summary_writer = summary_writer
    for worker in workers:
        worker.set_global_net(global_net)

    saver = tf.train.Saver(max_to_keep=10, var_list=[
        v for v in tf.trainable_variables() if "global" in v.name
    ] + [global_step])

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

    save_model_every_nth_minutes(sess, saver)

    coord = tf.train.Coordinator()

    # Load a previous checkpoint if it exists
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
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
                # print "max_return = \33[93m{}\33[00m".format(max_return)

            worker.env._render({"worker": worker})

        cv2.imshow("result", disp_img)
        cv2.waitKey(10)
        counter += 1

        schedule.run_pending()

    # Wait for all workers to finish
    coord.join(worker_threads)
