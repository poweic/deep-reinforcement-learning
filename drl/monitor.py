import gym
import time
import Queue
import multiprocessing
import tensorflow as tf
FLAGS = tf.flags.FLAGS

# Global method take take a queue "q" as input, consume data in the queue
# aggressively (i.e. get_nowait()), set the random seed, and replay those
# actions and render.
def renderer(q):

    env = gym.make(FLAGS.game)

    while True:
        try:
            rollout = q.get_nowait()

            env.seed(rollout.seed[0])
            env.reset()

            for action in rollout.action:
                env.render()
                env.step(action.T)
                time.sleep(0.01)

        except Queue.Empty:
            tf.logging.info("Nothing in queue to render, wait 5 seconds ...")
            time.sleep(5)

        except Exception as e:
            tf.logging.info("\33[31m[Exception]\33[0m {}".format(e))


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
    
    def start(self):
        self.render_process.start()

    def join(self):
        self.render_process.join()
