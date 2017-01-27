import tensorflow as tf
from tensorflow_helnet.layers import DenseLayer, Conv2DLayer, MaxPool2DLayer
from ac.utils import *
from ac.models import *
import ac.acer.worker

FLAGS = tf.flags.FLAGS
batch_size = FLAGS.batch_size

def flatten_all_leading_axes(x):
    return tf.reshape(x, [-1, x.get_shape()[-1].value])

class AcerEstimator():
    def __init__(self, add_summaries=False, trainable=True):

        self.trainable = trainable

        with tf.name_scope("inputs"):
            self.state = get_state_placeholder()

            self.actions         = tf.placeholder(tf.float32, [batch_size, 2], "actions")
            self.sampled_actions = tf.placeholder(tf.float32, [batch_size, 2], "sampled_actions")
            self.Q_ret           = tf.placeholder(tf.float32, [batch_size, 1], "Q_ret")
            self.Q_opc           = tf.placeholder(tf.float32, [batch_size, 1], "Q_opc")
            self.rho             = tf.placeholder(tf.float32, [batch_size, 1], "rho")
            self.rho_prime       = tf.placeholder(tf.float32, [batch_size, 1], "rho_prime")

        with tf.variable_scope("shared"):
            shared = build_shared_network(self.state, add_summaries=add_summaries)

        with tf.variable_scope("pi"):
            self.pi = self.policy_network(shared, 2)

        with tf.variable_scope("V"):
            self.value = state_value_network(shared)

        with tf.variable_scope("A"):
            self.adv = self.advantage_network(shared)
            self.Q_tilt = self.SDN_network(self.adv, self.value, self.pi)

        self.avg_net = getattr(AcerEstimator, "average_net", self)

        with tf.name_scope("ACER"):
            self.acer_g = self.compute_ACER_gradient(
                self.rho, self.pi, self.actions, self.Q_opc, self.value,
                self.rho_prime, self.Q_tilt, self.sampled_actions)

        with tf.name_scope("losses"):
            self.pi_loss = self.get_policy_loss(self.pi, self.acer_g)
            self.vf_loss = self.get_value_loss(
                self.Q_ret, self.Q_tilt, self.actions, self.rho, self.value)

            self.loss = tf.reduce_mean(self.pi_loss + self.vf_loss)

        self.optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate)
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        '''
        print "grads_and_vars = \33[93m{}\33[0m".format(grads_and_vars)
        '''
        self.grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]

        # Get all trainable variables initialized in this
        self.var_list = [v for g, v in self.grads_and_vars]
        '''
        print "var_list = \33[93m{}\33[0m".format(self.var_list)
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          tf.get_variable_scope().name)
        '''

    def compute_trust_region_update(self, g, pi_avg, pi, delta=0.5):
        """
        In ACER's original paper, they use delta uniformly sampled from [0.1, 2]
        """
        # Compute the KL-divergence between the policy distribution of the
        # average policy network and those of this network, i.e. KL(avg || this)
        KL_divergence = tf.contrib.distributions.kl(
            pi_avg.dist, pi.dist, allow_nan=False)

        # Take the partial derivatives w.r.t. phi (i.e. mu and sigma)
        k = tf.concat(1, tf.gradients(KL_divergence, pi.phi))

        # z* is the TRPO regularized gradient
        z_star = g - tf.maximum(0., (k * g - delta) / (g ** 2)) * k

        # By using stop_gradient, we make z_star being treated as a constant
        z_star = tf.stop_gradient(z_star)

        return z_star

    def get_policy_loss(self, pi, acer_g):

        with tf.name_scope("TRPO"):
            self.z_star = self.compute_trust_region_update(
                acer_g, self.avg_net.pi, self.pi)

        phi = tf.concat(1, pi.phi)
        losses = -tf.reduce_sum(phi * self.z_star, axis=1)

        return losses

    def get_value_loss(self, Q_ret, Q_tilt, actions, rho, value):

        with tf.name_scope("Q_target"):
            Q_tilt_a = Q_tilt(actions, name="Q_tilt_a")
            Q_target = tf.stop_gradient(Q_ret - Q_tilt_a)

        losses = - Q_target * (Q_tilt_a + tf.minimum(1., rho) * value)

        return losses

    def compute_ACER_gradient(self, rho, pi, actions, O_opc, value, rho_prime,
                              Q_tilt, sampled_actions):

        def d_log_prob(actions):
            return tf.concat(1, tf.gradients(pi.dist.log_prob(actions), pi.phi))

        # compute gradient with importance weight truncation using c = 10
        c = tf.constant(10, tf.float32, name="importance_weight_truncation_thres")

        with tf.name_scope("truncation"):
            with tf.name_scope("truncated_importance_weight"):
                rho_bar = tf.minimum(c, rho)

            with tf.name_scope("d_log_prob_a"):
                g_a = d_log_prob(self.actions)

            with tf.name_scope("target_1"):
                target_1 = self.Q_opc - self.value

            truncation = (rho_bar * target_1) * g_a

        # compute bias correction term
        with tf.name_scope("bias_correction"):
            with tf.name_scope("bracket_plus"):
                plus = tf.nn.relu(1. - c / rho_prime)

            with tf.name_scope("d_log_prob_a_prime"):
                g_ap = d_log_prob(sampled_actions)

            with tf.name_scope("target_2"):
                target_2 = Q_tilt(sampled_actions, name="Q_tilt_a_prime") - value

            bias_correction = (plus * target_2) * g_ap

        # g is called "truncation with bias correction" in ACER
        g = truncation + bias_correction

        return g

    def SDN_network(self, advantage, value, pi):
        """
        This function wrap advantage, value, policy pi within closure, so that
        the caller doesn't have to pass these as argument anymore
        """

        def Q_tilt(action, name, num_samples=5):
            with tf.name_scope(name):
                # See eq. 13 in ACER
                adv = advantage(action, name="A_action")
                advs = advantage(pi.dist.sample_n(num_samples), "A_sampled", num_samples)
                mean_adv = tf.reduce_mean(advs, axis=0)
                return value + adv - mean_adv

        return Q_tilt

    def policy_network(self, input, num_outputs):

        pi = AttrDict()

        # mu: [B, 2], sigma: [B, 2], phi is just a syntatic sugar
        pi.mu, pi.sigma = policy_network(input, num_outputs)
        pi.phi = [pi.mu, pi.sigma]

        # Reshape & create normal distribution and sample some actions
        pi.dist = tf.contrib.distributions.MultivariateNormalDiag(pi.mu, pi.sigma)

        return pi

    def advantage_network(self, input):

        # Given states
        def advantage(actions, name, num_samples=1):

            with tf.name_scope(name):
                ndims = len(actions.get_shape())
                broadcaster = tf.zeros([num_samples] + [1] * (ndims-1))
                input_ = input + broadcaster

                input_with_a = tf.concat(ndims - 1, [input_, actions])
                input_with_a = flatten_all_leading_axes(input_with_a)

                # 1st fully connected layer
                fc1 = DenseLayer(
                    input=input_with_a,
                    num_outputs=256,
                    nonlinearity="relu",
                    name="fc1")

                # 2nd fully connected layer that regresses the advantage
                fc2 = DenseLayer(
                    input=fc1,
                    num_outputs=1,
                    nonlinearity=None,
                    name="fc2")

                output = fc2
                if ndims == 3:
                    output = tf.reshape(output, [num_samples, -1, 1])

            return output

        return tf.make_template('advantage', advantage)

    @staticmethod
    def create_averge_network():
        if "average_net" not in AcerEstimator.__dict__:
            with tf.variable_scope("average_net"):
                AcerEstimator.average_net = AcerEstimator(add_summaries=True, trainable=False)

# AcerEstimator.create_averge_network = create_averge_network

AcerEstimator.Worker = ac.acer.worker.Worker
