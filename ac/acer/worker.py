import tensorflow as tf
import ac.acer.estimators
from ac.worker import Worker
from ac.utils import *
import time

class AcerWorker(Worker):
    """
    An A3C worker thread. Runs episodes locally and updates global shared value and policy nets.

    Args:
    name: A unique name for this worker
    env: The Gym environment used by this worker
    global_net: Instance of the globally shared network
    global_counter: Iterator that holds the global step
    summary_writer: A tf.train.SummaryWriter for Tensorboard summaries
    """
    def __init__(self, **kwargs):
        self.Estimator = ac.acer.estimators.AcerEstimator
        super(AcerWorker, self).__init__(**kwargs)

    def set_global_net(self, global_net):
        # Get global, local, and the average net var_list
        avg_vars = self.Estimator.average_net.var_list
        global_vars = global_net.var_list
        local_vars = self.local_net.var_list

        # Operation to copy params from global net to local net
        self.copy_params_op = make_copy_params_op(global_vars, local_vars)

        self.global_net = global_net

        def copy_global_to_avg():
            msg = "\33[94mInitialize average net when global_step = \33[0m"
            disp_op = tf.Print(self.global_step, [self.global_step], msg)
            copy_op = make_copy_params_op(global_vars, avg_vars)
            return tf.group(*[copy_op, disp_op])

        init_avg_net = tf.cond(
            tf.equal(self.global_step, 0),
            copy_global_to_avg,
            lambda: tf.no_op()
        )

        with tf.control_dependencies([init_avg_net]):
            train_op = make_train_op(self.local_net, global_net)

            with tf.control_dependencies([train_op]):

                self.train_and_update_avgnet_op = make_copy_params_op(
                    global_vars, avg_vars, alpha=FLAGS.avg_net_momentum
                )

        self.inc_global_step = tf.assign_add(self.global_step, 1)

    def reset_env(self):

        self.state = np.array([0, 2, 0, 0, 0, 0])
        self.action = np.array([0, 0])

        # Reshape to compatiable format
        self.state = self.state.astype(np.float32).reshape(6, -1)
        self.action = np.zeros((2, self.n_agents), dtype=np.float32)
        self.total_return = np.zeros((1, self.n_agents))
        self.current_reward = np.zeros((1, self.n_agents))

        # Add some noise to have diverse start points
        noise = np.random.randn(6, self.n_agents).astype(np.float32) * 0.5
        noise[2, :] /= 10

        self.state = self.state + noise

        self.env._reset(self.state)

    def _run(self):

        # Run on-policy ACER
        self._run_on_policy()

        # Run off-policy ACER N times
        N = np.random.poisson(FLAGS.replay_ratio)

        for i in range(N):
            self._run_off_policy()

        if self.max_global_steps is not None and global_t >= self.max_global_steps:
            tf.logging.info("Reached global step {}. Stopping.".format(global_t))
            self.coord.request_stop()
            return

    def copy_params_from_global(self):
        # Copy Parameters from the global networks
        self.sess.run(self.copy_params_op)

    def store_experience(self, transitions):
        # Store transitions in the replay buffer, discard the oldest by popping
        # the 1st element if it exceeds maximum buffer size
        rp = AcerWorker.replay_buffer

        rp.append(transitions)
        if len(rp) > FLAGS.max_replay_buffer_size:
            rp.pop(0)

        print "len(replay_buffer) = \33[93m{}\33[0m".format(len(rp))

    def _run_on_policy(self):
        self.copy_params_from_global()

        # Collect transitions {(s_0, a_0, r_0, mu_0), (s_1, ...), ... }
        n = int(np.ceil(FLAGS.t_max * FLAGS.command_freq))
        transitions, local_t, global_t = self.run_n_steps(n)

        # Compute gradient and Perform update
        self.update(transitions)

        self.store_experience(transitions)

    def _run_off_policy(self):
        rp = AcerWorker.replay_buffer

        if len(rp) <= 0:
            return

        self.copy_params_from_global()

        # Random select on episode from past experiences
        idx = np.random.randint(len(rp))

        # Compute gradient and Perform update
        self.update(rp[idx], off_policy=True)

    def run_n_steps(self, n_steps):

        transitions = []

        # Initial state
        self.reset_env()
        self.local_net.reset_lstm_state()

        reward = np.zeros((1, self.n_agents), dtype=np.float32)
        for i in range(n_steps):

            mdp_state = form_mdp_state(self.env, self.state, self.action, reward)

            # Predict an action
            self.action, pi = self.local_net.predict_actions(mdp_state, self.sess)

            self.action = self.action.T
            pi.mu = pi.mu.T
            pi.sigma = pi.sigma.T

            # print "action.shape = {}, mu.shape = {}, sigma.shape = {}".format(self.action.shape, pi.mu.shape, pi.sigma.shape)
            assert not np.any(np.isnan(self.action)), "i = {}, self.action = {}, mdp_state = {}".format(i, self.action, mdp_state)

            # Take several sub-steps in environment (the smaller the timestep,
            # the smaller each sub-step, the more accurate the simulation
            n_sub_steps = int(1. / FLAGS.command_freq / FLAGS.timestep)
            for j in range(n_sub_steps):
                next_state, reward, done, _ = self.env.step(self.action)

            self.current_reward = reward
            self.total_return += reward
            if np.max(self.total_return) > self.max_return:
                self.max_return = np.max(self.total_return)

            # Store transition
            # Down-sample transition to reduce correlation between samples
            transitions.append(AttrDict(
                mdp_state=mdp_state,
                pi=pi,
                action=self.action.copy(),
                next_state=next_state.copy(),
                reward=reward.copy(),
                done=done.copy()
            ))

            # Increase local and global counters
            local_t = next(self.local_counter)
            global_t = next(self.global_counter)

            if local_t % 100 == 0:
                tf.logging.info("{}: local Step {}, global step {}".format(self.name, local_t, global_t))

            if np.any(done):
                break
            else:
                self.state = next_state

        return transitions, local_t, global_t

    def update(self, transitions, off_policy=False):

        mdp_states = AttrDict({
            key: np.concatenate([
                t.mdp_state[key][None, ...] for t in transitions
            ], axis=0)
            for key in transitions[0].mdp_state.keys()
        })

        action = np.concatenate([t.action.T[None, ...]   for t in transitions ], axis=0)
        reward = np.concatenate([t.reward.T[None, ...]   for t in transitions ], axis=0)
        mu     = np.concatenate([t.pi.mu.T[None, ...]    for t in transitions ], axis=0)
        sigma  = np.concatenate([t.pi.sigma.T[None, ...] for t in transitions ], axis=0)

        net = self.local_net
        avg_net = self.Estimator.average_net

        feed_dict = {
            net.state.front_view        : mdp_states.front_view,
            net.state.vehicle_state     : mdp_states.vehicle_state,
            net.state.prev_action       : mdp_states.prev_action,
            net.state.prev_reward       : mdp_states.prev_reward,
            net.r                       : reward,
            net.mu.mu                   : mu,
            net.mu.sigma                : sigma,
            avg_net.state.front_view    : mdp_states.front_view,
            avg_net.state.vehicle_state : mdp_states.vehicle_state,
            avg_net.state.prev_action   : mdp_states.prev_action,
            avg_net.state.prev_reward   : mdp_states.prev_reward,
            net.a                       : action
        }

        '''
        for k, v in feed_dict.viewitems():
            print "feed_dict[{}].shape = {}".format(k.name, v.shape)
        '''

        ops = [
            {
                'pi': net.pi_loss,
                'vf': net.vf_loss,
                'total': net.loss
            },
            self.train_and_update_avgnet_op
        ]

        """
        ops.insert(1, {
            'Q_ret': net.Q_ret,
            'Q_opc': net.Q_opc,
            'value': net.value,
            'c': net.c,
            'r': net.r,
            'Q_tilt_a': net.Q_tilt_a,
            'rho': net.rho,
            'rho_prime': net.rho_prime,
            'pi_a': net.pi_a,
            'mu_a': net.mu_a,
            'pi_mean':  net.pi.mu,
            'pi_sigma': net.pi.sigma,
            'mu_mean':  net.mu.mu,
            'mu_sigma': net.mu.sigma,
        })

        loss, Q, _ = self.sess.run(ops, feed_dict=feed_dict)
        """

        net.reset_lstm_state()
        avg_net.reset_lstm_state()

        loss, _ = net.predict(ops, feed_dict, self.sess)
        # loss, _ = self.sess.run(ops, feed_dict=feed_dict)
        loss = AttrDict(loss)

        self.counter += 1
        gstep = self.sess.run(self.inc_global_step)
        print "#{:6d}, pi_loss = {:+12.3f}, vf_loss = {:+12.3f}, loss = {:+12.3f} {}".format(
            gstep, loss.pi, loss.vf, loss.total,
            "\33[93m[off policy]\33[0m" if off_policy else "\33[92m[on  policy]\33[0m"
        ),

        print "mdp_states.front_view.shape = ", mdp_states.front_view.shape


        """
        if off_policy:

            Q = AttrDict(Q)
            for k, v in Q.viewitems():
                Q[k] = np.squeeze(v)

            print "FLAGS.discount_factor = ", FLAGS.discount_factor
            Q.discounted_r = discount(Q.r.T, FLAGS.discount_factor).T

            for k, v in Q.viewitems():
                print "{} [{}]: \n{}".format(k, v.shape, v)

            import ipdb; ipdb.set_trace()
            aaaaaaa = 10
        """

AcerWorker.replay_buffer = []
