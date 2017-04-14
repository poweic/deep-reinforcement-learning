import numpy as np
import tensorflow as tf

def flatten_tensor_variables(ts):
    if isinstance(ts, list):
        return tf.concat(axis=0, values=[tf.reshape(x, [-1]) for x in ts])
    else:
        # already flattened
        return ts

def unflatten_tensor_variables(tvars_ref, t):

    tvars = []

    n = 0
    for v in tvars_ref:
        shape = v.get_shape()
        size = np.prod(shape.as_list())

        reshaped = tf.reshape(t[n:n+size], shape)
        tvars.append(reshaped)
        n += size

    return tvars

class TrpoOptimizer(tf.train.GradientDescentOptimizer):
    def __init__(self, learning_rate, reg_coeff=1e-5, cg_iters=10,
                 residual_tol=1e-10, use_locking=False, name="Trpo"):
        super(TrpoOptimizer, self).__init__(learning_rate, use_locking, name)
        self.reg_coeff = reg_coeff
        self.cg_iters = cg_iters
        self.residual_tol = residual_tol

    def compute_gradients(self, loss, var_list=None, **kwargs):
        grads_and_vars = super(TrpoOptimizer, self).compute_gradients(
            loss, var_list, **kwargs
        )

        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]

        grads = [g for g, v in grads_and_vars]
        tvars = [v for g, v in grads_and_vars]

        self.hvp = self._hvp_builder(grads, tvars)

        self.Hx = self.hvp(tvars)

        A = self.hvp
        b = flatten_tensor_variables(grads)
        cgrads_flat = self._conjugate_gradient(A, b)

        self.cgrads = cgrads = unflatten_tensor_variables(tvars, cgrads_flat)

        cgrads_and_vars = zip(cgrads, tvars)

        # this should return a step-adjusted cgrads_and_vars ??
        # or just overwrite apply_gradients ??
        # cgrads_and_vars = self._backtracking_line_search(loss, tvars, cgrads_and_vars)

        return cgrads_and_vars

    def _hvp_builder(self, grads, tvars):

        reg_coeff = self.reg_coeff

        #  See https://github.com/openai/rllab/blob/master/sandbox/rocky/tf/optimizers/conjugate_gradient_optimizer.py#L32
        def hessian_vector_prodcut(vector):

            g_flat = flatten_tensor_variables(grads)
            v_flat = flatten_tensor_variables(vector)

            Hx = tf.gradients(tf.reduce_sum(g_flat * tf.stop_gradient(v_flat)), tvars)

            Hx = [g if g is not None else tf.zeros_like(v) for g, v in zip(Hx, grads)]

            return flatten_tensor_variables(Hx) + reg_coeff * v_flat

        return tf.make_template('hessian_vector_prodcut', hessian_vector_prodcut)

    def _backtracking_line_search(self, loss, params, cgrads_and_vars):

        loss_graph_def = tf.graph_util.extract_sub_graph(
            loss.graph.as_graph_def(), [loss.name.replace(':0', '')])

        placeholder_names = [
            n.name + ":0" for n in loss_graph_def.node
            if n.op == "Placeholder"
        ]

        placeholders = [
            tf.get_default_graph().get_tensor_by_name(name)
            for name in placeholder_names
        ]

        loss_2 = tf.import_graph_def(
            loss_graph_def, input_map=dict(zip(placeholder_names, placeholders))
            , return_elements=[loss.name]
        )

        theta = params + cgrads_and_vars

        # filter name using "import/.../weights" in loss_2, and set those weights to
        # theta
        set_params = tf.group(*[v.assign(v + g) for g, v in cgrads_and_vars])

        with tf.control_dependencies([loss]):
            with tf.control_dependencies([set_params]):
                cgrads_and_vars = [(g + tf.reshape(loss_2, [-1]) * 0, v) for g, v in cgrads_and_vars]

        return cgrads_and_vars

    def _conjugate_gradient(self, A, b):

        def dot(a, b):
            return tf.reduce_sum(a * b)

        p_0 = b
        r_0 = b
        x_0 = tf.zeros_like(b)
        rdotr_0 = dot(r_0, r_0)

        def cond(i, p, r, x, rdotr):
            return tf.logical_and(i < self.cg_iters, rdotr > self.residual_tol)

        # See https://en.wikipedia.org/wiki/Derivation_of_the_conjugate_gradient_method#The_direct_Lanczos_method
        # and https://github.com/openai/rllab/blob/master/rllab/misc/krylov.py#L7
        def body(i, p, r, x, rdotr):

            i = i + 1
            z = A(p)
            v = rdotr / dot(p, z)
            x += v * p
            r -= v * z
            newrdotr = dot(r, r)

            mu = newrdotr / rdotr
            p = r + mu * p

            rdotr = newrdotr

            return i, p, r, x, rdotr

        i, p, r, x, rdotr = tf.while_loop(
            cond, body,
            loop_vars=[
                tf.constant(0, tf.int32), p_0, r_0, x_0, rdotr_0
            ]
        )

        return x
