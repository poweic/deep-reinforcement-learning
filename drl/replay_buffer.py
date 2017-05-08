import zlib
import cPickle
import itertools
import tensorflow as tf
from attrdict import AttrDict
from collections import deque
from drl.ac.utils import Timer, show_mem_usage
FLAGS = tf.flags.FLAGS

class ReplayBuffer(object):

    def __init__(self, fn=None, maxlen=None):

        if fn:
            self.load(fn)
        else:
            self.maxlen = maxlen
            self.deque = deque(maxlen=maxlen)

        # keep last 1000 compress, decompress time for profiling purpose
        self.timer = AttrDict(
            compress = Timer("compress"),
            decompress = Timer("decompress")
        )

        # A thread-safe get-and-increment counter
        self.counter = itertools.count()

        # fixed items will not be pushed away by append method
        self.fixed_items = []

    def load(self, fn, fixed=False):
        tf.logging.info("Loading replay from {}".format(fn))
        data = cPickle.load(open(fn, 'rb'))
        tf.logging.info("{} experiences loaded".format(len(data)))

        if fixed:
            self.fixed_items = list(data)
        else:
            self.deque = data
            self.maxlen = self.deque.maxlen

    def dump(self, fn):
        cPickle.dump(self.deque, open(fn, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

    def compress(self, item):
        self.timer.compress.tic()
        item = zlib.compress(cPickle.dumps(item, protocol=cPickle.HIGHEST_PROTOCOL))
        self.timer.compress.toc()
        return item

    def decompress(self, item):
        self.timer.decompress.tic()
        item = cPickle.loads(zlib.decompress(item))
        self.timer.decompress.toc()
        return item

    def append(self, item):

        if FLAGS.compress:
            item = self.compress(item)

            if self.counter.next() % self.maxlen == 0:
                show_mem_usage(self, "replay buffer")

        self.deque.append(item)

    def __len__(self):
        return len(self.deque) + len(self.fixed_items)

    def get(self, key):
        return self.deque[key]

    def __getitem__(self, key):
        n_fixed = len(self.fixed_items)
        if key < n_fixed:
            item = self.fixed_items[key]
        else:
            item = self.deque[key - n_fixed]

        if FLAGS.compress:
            item = self.decompress(item)

        return item
