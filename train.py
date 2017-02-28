#!/usr/bin/env python
import colored_traceback.always

import os
import sys
import cv2
import scipy.io
import numpy as np
import tensorflow as tf
import itertools
import shutil
import threading
import time
import schedule
from my_config import parse_flags
import multiprocessing

tf.flags.DEFINE_integer("max-steps", "1000", "Maximum steps per episode")
tf.flags.DEFINE_integer("random-seed", None, "Random seed for gym.env and TensorFlow")

FLAGS = parse_flags()

import gym
from gym import wrappers

def make_env():
    env = gym.envs.make(FLAGS.game)

    if FLAGS.random_seed is not None:
        env.seed(FLAGS.random_seed)

    env.reset()
    # Add monitor (None will use default video recorder, False will disable video recording)
    # env = wrappers.Monitor(env, FLAGS.exp, force=True, video_callable=None if FLAGS.record_video else False)

    if FLAGS.record_video:
        FLAGS.render_every = None

    FLAGS.action_space = env.action_space
    FLAGS.num_actions = env.action_space.shape[0]
    FLAGS.num_states = env.observation_space.shape[0]

    if FLAGS.game == "MountainCarContinuous-v0":
	import sklearn.pipeline
	import sklearn.preprocessing
	from sklearn.kernel_approximation import RBFSampler

	# Feature Preprocessing: Normalize to zero mean and unit variance
	# We use a few samples from the observation space to do this
	observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
	scaler = sklearn.preprocessing.StandardScaler()
	scaler.fit(observation_examples)

	# Used to converte a state to a featurizes represenation.
	# We use RBF kernels with different variances to cover different parts of the space
	featurizer = sklearn.pipeline.FeatureUnion([
	    ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
	    ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
	    ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
	    ("rbf4", RBFSampler(gamma=0.5, n_components=100))
	])
	featurizer.fit(scaler.transform(observation_examples))

	def featurize_state(state):
	    """ Returns the featurized representation for a state.
	    """
	    scaled = scaler.transform([state])
	    featurized = featurizer.transform(scaled)
	    return featurized[0]

	FLAGS.featurize_state = featurize_state
	FLAGS.num_states = 400

    return env

env = make_env()
if FLAGS.display:
    env.render()
env.close()

from ac.estimators import get_estimator
from ac.worker import Worker
from ac.utils import (
    make_copy_params_op, save_model_every_nth_minutes, EpisodeStats
)

from gym import wrappers
from gym_offroad_nav.envs import OffRoadNavEnv
from gym_offroad_nav.vehicle_model import VehicleModel
from gym_offroad_nav.vehicle_model_tf import VehicleModelGPU

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
    # vehicle_model_gpu = VehicleModelGPU(FLAGS.timestep, FLAGS.vehicle_model_noise_level)
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
    # env = OffRoadNavEnv(rewards, vehicle_model, vehicle_model_gpu)
    # env = wrappers.Monitor(env, FLAGS.monitor_dir, video_callable=False)
    return env

# Optionally empty model directory
if FLAGS.reset:
    shutil.rmtree(FLAGS.base_dir, ignore_errors=True)

tf.logging.info("Number of cpus = {}".format(multiprocessing.cpu_count()))

# Optionally empty model directory
cfg = tf.ConfigProto()
cfg.gpu_options.per_process_gpu_memory_fraction = FLAGS.per_process_gpu_memory_fraction
with tf.Session(config=cfg) as sess:

    global_episode_stats = EpisodeStats()

    # Keeps track of the number of updates we've performed
    global_step = tf.Variable(0, name="global_step", trainable=False)
    FLAGS.global_step = global_step

    t = int(time.time())
    global_time_init = tf.Variable(t, name="global_time_init", dtype=tf.int32, trainable=False)
    global_time      = tf.Variable(t, name="global_time"     , dtype=tf.int32, trainable=False)
    FLAGS.global_timestep_placeholder = tf.placeholder(tf.int32, [])
    FLAGS.set_time_op = tf.assign(global_time, FLAGS.global_timestep_placeholder)

    FLAGS.global_timestep = global_time - global_time_init

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
            global_episode_stats=global_episode_stats,
            global_net=global_net,
            add_summaries=(i == 0),
            n_agents=FLAGS.n_agents_per_worker)

        workers.append(worker)

    summary_dir = os.path.join(FLAGS.base_dir, "train")
    summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)

    workers[0].summary_writer = summary_writer

    tf.logging.info("Creating TensorFlow graph Saver ...")
    FLAGS.saver = tf.train.Saver(max_to_keep=10, var_list=[
        v for v in tf.trainable_variables() if "worker" not in v.name
    ] + [global_step])

    tf.logging.info("Initializing all TensorFlow variables ...")
    sess.run(tf.global_variables_initializer())
    tf.get_default_graph().finalize()

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
    tf.logging.info("Launching worker threads ...")
    for i in range(len(workers)):
        worker_fn = lambda j=i: workers[j].run(sess, coord)
        t = threading.Thread(target=worker_fn)
        t.start()
        if i == 0:
            time.sleep(5)
        worker_threads.append(t)

    # server.start()

    # Show how agent behaves in envs in main thread
    """
    if FLAGS.display:
        while not Worker.stop:
            for worker in workers:
                if worker.max_return > max_return:
                    max_return = worker.max_return
                    # print "max_return = \33[93m{}\33[00m".format(max_return)

                worker.env._render({"worker": worker})

            cv2.imshow("result", disp_img)
            cv2.waitKey(10)

            schedule.run_pending()
    """
    while not Worker.stop:
        if FLAGS.display:
            workers[0].env.render()

        """
        for worker in workers:
            worker.env.render()
        """

        time.sleep(1)
        schedule.run_pending()

    # Wait for all workers to finish
    coord.join(worker_threads)