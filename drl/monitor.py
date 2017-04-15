import gym
import time
import Queue
import traceback
import multiprocessing
import numpy as np
import tensorflow as tf
from gym import wrappers
FLAGS = tf.flags.FLAGS

# Global method take take a queue "q" as input, consume data in the queue
# aggressively (i.e. get_nowait()), set the random seed, and replay those
# actions and render.
def renderer(q):

    env = None

    while True:
        try:
            seed, actions = q.get_nowait()

            # env was initially set to None, because we start rendering only
            # after we receive data
            if env is None:
                env = gym.make(FLAGS.game)
                env = wrappers.Monitor(env, FLAGS.video_dir)

            env.seed(seed[0])
            env.reset()

            for action in actions:
                env.render()
                _, _, done, _ = env.step(action.T)

            assert done

        except Queue.Empty:
            # Nothing in queue to render, wait 5 seconds ...
            time.sleep(5)

        except Exception as e:
            print "\33[91m"
            traceback.print_exc()
            raise e

class Monitor(object):
    def __init__(self):
        self.queue = multiprocessing.Manager().Queue(maxsize=5)
        self.render_process = multiprocessing.Process(
            target=renderer, args=(self.queue,))

    def send(self, data):
        try:
            self.queue.put_nowait(data)
        except:
            pass

    def monitor(self, workers):
        self.workers = workers
        self.prev_data = [None] * len(workers)

    def refresh(self):
        # Iterate all workers and see whether there's new rollout to render
        # We keep track the previous data we sent for each worker so that we
        # won't render same rollout twice.
        for i, worker in enumerate(self.workers):
            if len(worker.seed_actions_buffer) == 0:
                continue

            data = worker.seed_actions_buffer[-1]
            if self.prev_data[i] is data:
                continue

            self.send(data)
            self.prev_data[i] = data
    
    def start(self):
        self.render_process.start()

    def join(self):
        self.render_process.join()
