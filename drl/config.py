import os
import numpy as np
import tensorflow as tf
import pprint

tf.flags.DEFINE_integer("max-steps", "1000", "Maximum steps per episode")
tf.flags.DEFINE_integer("n-steps", "100", "Number of steps in environment before we process them and use that to update the network")
tf.flags.DEFINE_integer("random-seed", None, "Random seed for gym.env and TensorFlow")

tf.flags.DEFINE_string("base-dir", "exp/", "Directory to write Tensorboard summaries and models to.")
tf.flags.DEFINE_string("exp", None, "Optional experiment tag")
tf.flags.DEFINE_string("log-file", None, "log file")
tf.flags.DEFINE_string("stats-file", None, "stats file")
tf.flags.DEFINE_string("game", None, "Game environment. Ex: Humanoid-v1, OffRoadNav-v0")
tf.flags.DEFINE_string("estimator-type", "ACER", "Choose A3C or ACER")
tf.flags.DEFINE_string("qprop-type", "adaptive", "Choose \"adaptive\", \"conservative\", or \"aggressive\"")

tf.flags.DEFINE_integer("min-episodes", 100, "minimum episodes to play")
tf.flags.DEFINE_float("score-to-win", 100.0, "score to win if the average of last N episodes is greater than this number")

tf.flags.DEFINE_integer("max-global-steps", None, "Stop training after this many updates of neural network. Defaults to run forever.")
tf.flags.DEFINE_integer("seq-length", None, "sequence length used for construct TF graph")
tf.flags.DEFINE_integer("batch-size", None, "batch size used for construct TF graph")
tf.flags.DEFINE_integer("max-seq-length", 1024, "maximum sequence length without out of memory")
tf.flags.DEFINE_integer("decay-steps", 1000, "Decay learning using exponential_decay with staircase=True")
tf.flags.DEFINE_float("decay-rate", 0.7071, "Decay learning using exponential_decay with staircase=True")
tf.flags.DEFINE_float("per-process-gpu-memory-fraction", 1.0, "per_process_gpu_memory_fraction for TF session config")
tf.flags.DEFINE_boolean("staircase", False, "Set exponential_decay with staircase=True/False")
tf.flags.DEFINE_boolean("timestep-as-feature", False, "Add timestep as feature by adding an extra dimension")
tf.flags.DEFINE_boolean("use-lstm", True, "Use LSTM when set True")
tf.flags.DEFINE_boolean("batch-norm", False, "Use batch norm whenever possible")
tf.flags.DEFINE_boolean("double-precision", False, "Use tf.float64")
tf.flags.DEFINE_boolean("summarize", False, "Create summary writer")
tf.flags.DEFINE_boolean("debug-dump", False, "dump debugging information to *.mat file")

tf.flags.DEFINE_integer("log-episode-stats-every-nth", 20, "Print stats of episode every nth")
tf.flags.DEFINE_integer("parallelism", 1, "Number of threads to run. If not set we run [num_cpu_cores] threads.")
tf.flags.DEFINE_integer("save-every-n-minutes", 10, "Save model every N minutes")
tf.flags.DEFINE_integer("off-policy-batch-size", 1, "batch rollouts when performing off-policy updates")

tf.flags.DEFINE_float("replay-ratio", 10, "off-policy memory replay ratio, choose a number from {0, 1, 4, 8}")
tf.flags.DEFINE_integer("max-replay-buffer-size", 100, "off-policy memory replay buffer")
tf.flags.DEFINE_boolean("prioritize-replay", False, "Use choice(length) to sample replay")
tf.flags.DEFINE_boolean("compress", True, "compress memory replay using cPickle + zlib")
tf.flags.DEFINE_integer("regenerate-size", 1000, "number of episodes experience to regenerate after resuming")

tf.flags.DEFINE_float("avg-net-momentum", 0.995, "soft update momentum for average policy network in TRPO")
tf.flags.DEFINE_float("importance-weight-truncation-threshold", 5, "soft update momentum for average policy network in TRPO")
tf.flags.DEFINE_boolean("mixture-model", False, "Use single Gaussian if set to True, use GMM otherwise")
tf.flags.DEFINE_string("policy-dist", "Gaussian", "Either Gaussian, Beta, or StudentT")
tf.flags.DEFINE_integer("bucket-width", 10, "bucket_width")
tf.flags.DEFINE_integer("num-sdn-samples", 8, "soft update momentum for average policy network in TRPO")

tf.flags.DEFINE_boolean("share-network", True, "If set, value net and policy net will share a common network")
tf.flags.DEFINE_boolean("bi-directional", False, "If set, use bi-directional RNN/LSTM")
tf.flags.DEFINE_boolean("reset", False, "If set, delete the existing model directory and start training from scratch.")
tf.flags.DEFINE_boolean("display", True, "If set, no imshow will be called")
tf.flags.DEFINE_boolean("show-memory-usage", False, "If set, show memory usage during training")
tf.flags.DEFINE_boolean("resume", False, "If set, resume training from the corresponding last checkpoint file")
tf.flags.DEFINE_boolean("debug", False, "If set, turn on the debug flag")
tf.flags.DEFINE_boolean("dump-crash-report", False, "If set, dump mdp_states and internal TF variables when crashed.")
tf.flags.DEFINE_boolean("bootstrap", True, "If set, bootstrap V from the next state")
tf.flags.DEFINE_boolean("train-value-scale", False, "If set, add additional trainable value scale TF variable")

tf.flags.DEFINE_float("t-max", 30, "Maximum elasped time per simulation (in seconds)")

tf.flags.DEFINE_integer("hidden-size", 128, "size of hidden layer")
tf.flags.DEFINE_float("learning-rate", 2e-4, "Learning rate for policy net and value net")
tf.flags.DEFINE_float("lr-vp-ratio", 1, "Learning rate of value:Learning rate of policy")
tf.flags.DEFINE_boolean("random-learning-rate", False, "Random sample learning rate from LogUniform(min, max)")
tf.flags.DEFINE_boolean("min-learning-rate", 1e-4, "min learning rate used in LogUniform")
tf.flags.DEFINE_boolean("max-learning-rate", 5e-4, "max learning rate used in LogUniform")

tf.flags.DEFINE_float("l2-reg", 1e-4, "L2 regularization multiplier")
tf.flags.DEFINE_float("max-gradient", 10, "Threshold for gradient clipping used by tf.clip_by_global_norm")
tf.flags.DEFINE_float("timestep", 0.025, "Simulation timestep")
tf.flags.DEFINE_float("wheelbase", 2.00, "Wheelbase of the vehicle in meters")
tf.flags.DEFINE_float("vehicle-model-noise-level", 0.1, "level of white noise (variance) in vehicle model")
tf.flags.DEFINE_float("entropy-cost-mult", 1e-3, "multiplier used by entropy regularization")
tf.flags.DEFINE_float("discount-factor", 0.995, "discount factor in Markov decision process (MDP)")
tf.flags.DEFINE_float("lambda_", 0.95, "lambda in TD-Lambda (temporal difference learning)")

tf.flags.DEFINE_string("map-def", "map0", "map definition file in *.yaml format for OffRoadNav-v0")
tf.flags.DEFINE_float("command-freq", 20, "How frequent we send command to vehicle (in Hz)")
tf.flags.DEFINE_integer("field-of-view", 20, "size of front view (N x N) passed to network")
tf.flags.DEFINE_integer("downsample", 1, "downsample front view by this scale")
tf.flags.DEFINE_integer("n-agents-per-worker", 1, "number of agents per worker thread")
tf.flags.DEFINE_integer("viewport-scale", 4, "number of agents per worker thread")
tf.flags.DEFINE_boolean("drift", False, "If set, turn on drift")

tf.flags.DEFINE_float("min-mu-vf", 2. / 3.6, "Minimum forward velocity of vehicle (m/s)")
tf.flags.DEFINE_float("max-mu-vf", 14. / 3.6, "Maximum forward velocity of vehicle (m/s)")
tf.flags.DEFINE_float("min-mu-steer", -30 * np.pi / 180, "Minimum steering angle (rad)")
tf.flags.DEFINE_float("max-mu-steer", +30 * np.pi / 180, "Maximum steering angle (rad)")

tf.flags.DEFINE_float("min-sigma-vf", 1.0 / 3.6, "Minimum variance of forward velocity")
tf.flags.DEFINE_float("max-sigma-vf", 1.1 / 3.6, "Maximum variance of forward velocity")
tf.flags.DEFINE_float("min-sigma-steer", 3. * np.pi / 180, "Minimum variance of steering angle (rad)")
tf.flags.DEFINE_float("max-sigma-steer", 20 * np.pi / 180, "Maximum variance of steering angle (rad)")

def parse_flags():
    # Parse command line arguments, add some additional flags, and print them out
    FLAGS = tf.flags.FLAGS

    # If not starts with "/", treat it as relative path
    if not FLAGS.base_dir.startswith("/"):
        FLAGS.base_dir = os.getcwd() + "/" + FLAGS.base_dir

    FLAGS.exp_dir = "{}/{}{}".format(
        FLAGS.base_dir, FLAGS.game, "-" + FLAGS.exp if FLAGS.exp is not None else ""
    )

    FLAGS.log_dir        = FLAGS.exp_dir + "/log/"
    # FLAGS.monitor_dir    = FLAGS.exp_dir + "/monitor"
    FLAGS.checkpoint_dir = FLAGS.exp_dir + "/checkpoint"
    FLAGS.save_path      = FLAGS.checkpoint_dir + "/model"
    FLAGS.debug_dir      = FLAGS.exp_dir + "/debug"

    FLAGS.dtype = tf.float64 if FLAGS.double_precision else tf.float32

    from drl.ac.utils import AttrDict, mkdir_p

    mkdir_p(FLAGS.checkpoint_dir)
    mkdir_p(FLAGS.debug_dir)

    if FLAGS.random_learning_rate:
        low  = np.log10(FLAGS.min_learning_rate)
        high = np.log10(FLAGS.max_learning_rate)
        FLAGS.learning_rate = 10. ** np.random.uniform(low=low, high=high)

    FLAGS.regenerate_exp_after_resume = FLAGS.resume

    import drl.logger

    exp_config = pprint.pformat(FLAGS.__flags)
    FLAGS.exp_config = exp_config
    tf.logging.info(exp_config)

    return FLAGS
