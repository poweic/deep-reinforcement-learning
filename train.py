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
import gym_offroad_nav.envs

from ac.estimators import get_estimator
from ac.worker import Worker
from ac.utils import save_model, write_statistics, EpisodeStats, get_dof, state_featurizer

def make_env(worker=None):
    env = gym.make(FLAGS.game)

    if FLAGS.random_seed is not None:
        env.seed(FLAGS.random_seed)

    FLAGS.action_space = env.action_space
    FLAGS.num_actions = get_dof(env.action_space)
    FLAGS.num_states = get_dof(env.observation_space)

    FLAGS.featurize_state, FLAGS.num_states = state_featurizer(env)

    return env

env = make_env()
if FLAGS.display:
    env.render()

# Optionally empty model directory
if FLAGS.reset:
    shutil.rmtree(FLAGS.base_dir, ignore_errors=True)

tf.logging.info("Number of cpus = {}".format(multiprocessing.cpu_count()))

# Optionally empty model directory
cfg = tf.ConfigProto()
cfg.gpu_options.per_process_gpu_memory_fraction = FLAGS.per_process_gpu_memory_fraction
with tf.Session(config=cfg) as sess:

    FLAGS.sess = sess
    FLAGS.stats = EpisodeStats()

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
            make_env=make_env,
            global_counter=global_counter,
            global_episode_stats=FLAGS.stats,
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

    # Save model and dump statistics every n minutes
    schedule.every(FLAGS.save_every_n_minutes).minutes.do(save_model)
    schedule.every(FLAGS.save_every_n_minutes).minutes.do(write_statistics)

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
    while not Worker.stop:
        if FLAGS.display:
            last = Worker.replay_buffer[-1]

            env.seed(last.seed)
            env.reset()

            for action in last.action:
                env.step(action.T)
                env.render()
                cv2.waitKey(10)
        else:
            time.sleep(1)

        schedule.run_pending()

    # Wait for all workers to finish
    coord.join(worker_threads)

    save_model()
    write_statistics()

env.close()
