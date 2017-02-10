#!/usr/bin/python
import colored_traceback.always

import os
import sys
import cv2
import scipy.io
import numpy as np
import tensorflow as tf

tf.flags.DEFINE_string("model-dir", "/Data3/a3c-offroad/", "Directory to write Tensorboard summaries and models to.")
tf.flags.DEFINE_string("tag", None, "Optional experiment tag")
tf.flags.DEFINE_string("game", "line", "Game environment")
tf.flags.DEFINE_string("estimator-type", "A3C", "Choose A3C or ACER")

tf.flags.DEFINE_integer("max-global-steps", None, "Stop training after this many steps in the environment. Defaults to running indefinitely.")
tf.flags.DEFINE_integer("seq-length", None, "sequence length used for construct TF graph")
tf.flags.DEFINE_integer("batch-size", None, "batch size used for construct TF graph")

tf.flags.DEFINE_integer("eval-every", 30, "Evaluate the policy every N seconds")
tf.flags.DEFINE_integer("parallelism", 1, "Number of threads to run. If not set we run [num_cpu_cores] threads.")
tf.flags.DEFINE_integer("downsample", 5, "Downsample transitions to reduce sample correlation")
tf.flags.DEFINE_integer("n-agents-per-worker", 16, "Downsample transitions to reduce sample correlation")
tf.flags.DEFINE_integer("save-every-n-minutes", 10, "Save model every N minutes")

tf.flags.DEFINE_integer("field-of-view", 20, "size of front view (N x N) passed to network")

tf.flags.DEFINE_float("replay-ratio", 10, "off-policy memory replay ratio, choose a number from {0, 1, 4, 8}")
tf.flags.DEFINE_integer("max-replay-buffer-size", 100, "off-policy memory replay buffer")
tf.flags.DEFINE_float("avg-net-momentum", 0.995, "soft update momentum for average policy network in TRPO")
tf.flags.DEFINE_float("max-Q-diff", None, "Maximum Q difference (for robustness)")
tf.flags.DEFINE_boolean("mixture-model", False, "Use single Gaussian if set to True, use GMM otherwise")
tf.flags.DEFINE_string("policy-dist", "Gaussian", "Either Gaussian, Beta, or StudentT")
tf.flags.DEFINE_integer("bucket-width", 10, "bucket_width")

tf.flags.DEFINE_boolean("bi-directional", False, "If set, use bi-directional RNN/LSTM")
tf.flags.DEFINE_boolean("drift", False, "If set, turn on drift")
tf.flags.DEFINE_boolean("reset", False, "If set, delete the existing model directory and start training from scratch.")
tf.flags.DEFINE_boolean("display", True, "If set, no imshow will be called")
tf.flags.DEFINE_boolean("resume", False, "If set, resume training from the corresponding last checkpoint file")
tf.flags.DEFINE_boolean("debug", False, "If set, turn on the debug flag")
tf.flags.DEFINE_boolean("dump-crash-report", False, "If set, dump mdp_states and internal TF variables when crashed.")

tf.flags.DEFINE_float("t-max", 30, "Maximum elasped time per simulation (in seconds)")
tf.flags.DEFINE_float("command-freq", 20, "How frequent we send command to vehicle (in Hz)")

tf.flags.DEFINE_float("learning-rate", 2e-4, "Learning rate for policy net and value net")
tf.flags.DEFINE_float("l2-reg", 1e-4, "L2 regularization multiplier")
tf.flags.DEFINE_float("max-gradient", 10, "Threshold for gradient clipping used by tf.clip_by_global_norm")
tf.flags.DEFINE_float("timestep", 0.025, "Simulation timestep")
tf.flags.DEFINE_float("wheelbase", 2.00, "Wheelbase of the vehicle in meters")
tf.flags.DEFINE_float("vehicle-model-noise-level", 0.1, "level of white noise (variance) in vehicle model")
tf.flags.DEFINE_float("entropy-cost-mult", 1e-3, "multiplier used by entropy regularization")
tf.flags.DEFINE_float("discount-factor", 0.995, "discount factor in Markov decision process (MDP)")
tf.flags.DEFINE_float("lambda_", 0.50, "lambda in TD-Lambda (temporal difference learning)")

tf.flags.DEFINE_float("min-mu-vf", 6. / 3.6, "Minimum forward velocity of vehicle (m/s)")
tf.flags.DEFINE_float("max-mu-vf", 14. / 3.6, "Maximum forward velocity of vehicle (m/s)")
tf.flags.DEFINE_float("min-mu-steer", -30 * np.pi / 180, "Minimum steering angle (rad)")
tf.flags.DEFINE_float("max-mu-steer", +30 * np.pi / 180, "Maximum steering angle (rad)")

tf.flags.DEFINE_float("min-sigma-vf", 1.0 / 3.6, "Minimum variance of forward velocity")
tf.flags.DEFINE_float("max-sigma-vf", 1.1 / 3.6, "Maximum variance of forward velocity")
tf.flags.DEFINE_float("min-sigma-steer", 3. * np.pi / 180, "Minimum variance of steering angle (rad)")
tf.flags.DEFINE_float("max-sigma-steer", 20 * np.pi / 180, "Maximum variance of steering angle (rad)")

import itertools
import shutil
import threading
import time
import schedule
from pprint import pprint

from ac.estimators import get_estimator
from ac.worker import Worker
from ac.utils import make_copy_params_op, AttrDict, save_model_every_nth_minutes
# from ac.a3c.monitor import server

from gym_offroad_nav.envs import OffRoadNavEnv
from gym_offroad_nav.vehicle_model import VehicleModel

import my_logger

# Parse command line arguments, add some additional flags, and print them out
FLAGS = tf.flags.FLAGS
FLAGS.checkpoint_dir = "{}/checkpoints/{}{}".format(
    FLAGS.model_dir, FLAGS.game, "-" + FLAGS.tag if FLAGS.tag is not None else ""
)
FLAGS.save_path = FLAGS.checkpoint_dir + "/model"
FLAGS.action_space = AttrDict(
    n_actions   = 2,
    low        = [FLAGS.min_mu_vf   , FLAGS.min_mu_steer   ],
    high       = [FLAGS.max_mu_vf   , FLAGS.max_mu_steer   ],
    sigma_low  = [FLAGS.min_sigma_vf, FLAGS.min_sigma_steer],
    sigma_high = [FLAGS.max_sigma_vf, FLAGS.max_sigma_steer],
)
pprint(FLAGS.__flags)

# 
W = 400
disp_img = np.zeros((2*W, 2*W*2, 3), dtype=np.uint8)
disp_lock = threading.Lock()
def imshow4(idx, img):
    global disp_img
    x = idx / 4
    y = idx % 4
    with disp_lock:
        disp_img[x*W:x*W+img.shape[0], y*W:y*W+img.shape[1], :] = np.copy(img)

cv2.imshow4 = imshow4

def make_env():
    vehicle_model = VehicleModel(FLAGS.timestep, FLAGS.vehicle_model_noise_level)
    reward_fn = "data/{}.mat".format(FLAGS.game)
    rewards = scipy.io.loadmat(reward_fn)["reward"].astype(np.float32)
    # rewards -= 100
    # rewards -= 15
    rewards = (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards))
    # rewards = (rewards - 0.5) * 2 # 128
    rewards = (rewards - 0.7) * 2
    rewards[rewards > 0] *= 10

    # rewards[rewards < 0.1] = -1
    env = OffRoadNavEnv(rewards, vehicle_model)
    return env

# Optionally empty model directory
'''
if FLAGS.reset:
    shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
'''

def compute_mean_steering_angle(reward):
    from ac.utils import to_image
    rimg = to_image(reward, 20)
    cv2.imshow("reward", rimg)

    H, W = reward.shape
    yv, xv = np.meshgrid(range(H), range(W))
    xv = xv.astype(np.float32)
    yv = yv.astype(np.float32)

    theta = np.arctan((yv - W / 2 + 0.5) / (H - xv))

    mean_steer = np.mean(theta * reward)

    print "mean_steer = {} degree".format(mean_steer / np.pi * 180)

    cv2.imshow("theta", np.abs(theta))
    cv2.waitKey(0)
    sys.exit()

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
with tf.Session(config=config) as sess:

    # Keeps track of the number of updates we've performed
    global_step = tf.Variable(0, name="global_step", trainable=False)
    max_return = 0

    # Get estimator class by type name
    Estimator = get_estimator(FLAGS.estimator_type)

    # Global policy and value nets
    with tf.variable_scope("global_net"):
        global_net = Estimator(trainable=False)

    # Global step iterator
    global_counter = itertools.count()

    # Create worker graphs
    workers = []
    for i in range(FLAGS.parallelism):
        name = "worker_%d" % i
        tf.logging.info("Initializing {} ...".format(name))

        worker = Estimator.Worker(
            name=name,
            env=make_env(),
            global_counter=global_counter,
            global_net=global_net,
            add_summaries=(i == 0),
            n_agents=FLAGS.n_agents_per_worker)

        workers.append(worker)

    summary_dir = os.path.join(FLAGS.model_dir, "train")
    summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)

    workers[0].summary_writer = summary_writer

    FLAGS.saver = tf.train.Saver(max_to_keep=10, var_list=[
        v for v in tf.trainable_variables() if "worker" not in v.name
    ] + [global_step])

    sess.run(tf.global_variables_initializer())

    save_model_every_nth_minutes(sess)

    coord = tf.train.Coordinator()

    # Load a previous checkpoint if it exists
    if FLAGS.resume:
        latest_checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if latest_checkpoint:
            tf.logging.info("Loading model checkpoint: {}".format(latest_checkpoint))
            FLAGS.saver.restore(sess, latest_checkpoint)

    # Start worker threads
    worker_threads = []
    for i in range(len(workers)):
        worker_fn = lambda j=i: workers[j].run(sess, coord)
        t = threading.Thread(target=worker_fn)
        time.sleep(0.5)
        t.start()
        worker_threads.append(t)

    # server.start()

    # Show how agent behaves in envs in main thread
    if FLAGS.display:
        counter = 0
        while not Worker.stop:
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
