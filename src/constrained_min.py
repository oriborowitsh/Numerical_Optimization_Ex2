import numpy as np

STEP_TOLERANCE = 1e-8
OBJECTIVE_TOLERANCE = 1e-12
MAX_ITER= 100
X_O= np.array([1, 1])

MU=10
EPSILON= 1e-10

class ConstrainedMin:

    def __init__(self, func, x0, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, obj_tol=OBJECTIVE_TOLERANCE,
                 param_tol=STEP_TOLERANCE, max_iter=MAX_ITER):
        self.f = func
        self.x = x0

        self.ineq_constraints = ineq_constraints
        self.A = eq_constraints_mat
        self.b = eq_constraints_rhs

        self.obj_tol = obj_tol
        self.param_tol = param_tol
        self.max_iter = max_iter

        self.n= self.x.shape[0]


        self.grad = np.zeros_like(self.x)
        self.hessian = np.zeros((self.n,self.n))

        self.history = []

    def compute_f_k_grad_k_hessian(self, func, compute_hessian):
        f_k, self.grad, self.hessian= func(self.x, compute_hessian)
        return f_k

    def compute_a_k(self, func, f_k, p_k, wolfe_condition=0.01, back_tracking=0.5):
            alpha=1
            new_x= self.x+ alpha*p_k
            l_x= f_k+ wolfe_condition* np.dot(self.grad.T, p_k) * alpha
            while func(new_x,False)[0] > l_x:
                alpha= alpha*back_tracking
                new_x = self.x + alpha * p_k
                l_x = f_k + wolfe_condition * np.dot(self.grad.T, p_k) * alpha
            return alpha



    def check_termination_conditions(self, f_k, prev_x, prev_f_k, p_k):

        #small enough distance between iterations
        if np.linalg.norm(self.x- prev_x) < self.param_tol:
            return True

        #small enough objective change
        if abs(f_k- prev_f_k) < self.obj_tol:
            return True

        #Newton Decrement
        if 0.5*np.dot(np.dot(p_k.T, self.hessian), p_k) < self.obj_tol:
            return True

        return False



    def compute_direction(self):
        if self.A is None or self.A.size == 0:
            # No equality constraints: standard Newton direction
            return -np.linalg.solve(self.hessian, self.grad)

        p=self.A.shape[0]
        zero_block = np.zeros((p, p))


        top = np.hstack((self.hessian, self.A.T))
        bottom = np.hstack((self.A, zero_block))
        block_matrix_lhs= np.vstack((top, bottom))

        zero_array= np.zeros((p))
        block_matrix_rhs= np.hstack((-self.grad, zero_array))
        solution= np.linalg.solve(block_matrix_lhs, block_matrix_rhs)
        return solution[:self.n]


    def inner_iterations(self, func, for_test=True):
        f_k = self.compute_f_k_grad_k_hessian(func, True)
        for i in range(self.max_iter):
            try:
                if not for_test:
                    print(f"inner iteration {i}, current location {self.x}, current objective value {f_k}")

                p_k = self.compute_direction()
                a_k = self.compute_a_k(func,f_k, p_k)


                prev_x = self.x.copy()
                prev_f_k = f_k


                self.x = self.x + a_k * p_k
                f_k = self.compute_f_k_grad_k_hessian(func, True)


                if self.check_termination_conditions(f_k, prev_x, prev_f_k, p_k):
                    return self.x, f_k, (i + 1), True
            except:
                return self.x, f_k, (i + 1), False
        return self.x, f_k, (i + 1), False


    def phi(self, x):
        barrier_val = 0
        barrier_grad = np.zeros_like(x)
        barrier_hess = np.zeros((self.n, self.n))

        for g in self.ineq_constraints:
            g_val, g_grad, g_hess = g(x, True)
            if g_val >= 0:
                return np.inf, barrier_grad, barrier_hess  # outside feasible region

            barrier_val -= np.log(-g_val)
            barrier_grad += (1 / -g_val) * g_grad
            barrier_hess += ((1 / (g_val ** 2)) * np.outer(g_grad, g_grad)) + (1 / -g_val) * g_hess

        return barrier_val, barrier_grad, barrier_hess


    def augmented_objective(self, t):
        def augmented(x, compute_hess):
            f_val, f_grad, f_hess = self.f(x, True)
            barrier_val, barrier_grad, barrier_hess = self.phi(x)
            return f_val+(1/t)*barrier_val, f_grad+(1/t)*barrier_grad, f_hess+(1/t)*barrier_hess
        return augmented

    def outer_iterations(self, mu, epsilon, for_test=True):
        t = 1
        f_k = self.f(self.x, False)[0]
        self.history.append((self.x.copy(), f_k))

        m = len(self.ineq_constraints) if self.ineq_constraints else 0
        if m == 0:
            # No inequality constraints: solve f with equality constraints only
            self.inner_iterations(self.f, True)
            f_k = self.f(self.x, False)[0]
            self.history.append((self.x.copy(), f_k))
            return self.x, f_k, self.history


        while (m / t) >= epsilon:
            if not for_test:
                print(f"Outer iteration {t}, current location {self.x}, current objective value {f_k}")
            self.inner_iterations(self.augmented_objective(t), True)
            t = mu * t
            f_k = self.f(self.x, True)[0]
            self.history.append((self.x.copy(), f_k))

        print(f"After last outer iteration {t/mu}, last location x {np.round(self.x,8)}, last objective value {
        np.round(f_k,8)}")
        if self.A is not None and self.A.size > 0:
            print(f"Equality constraint Ax (should be ~{self.b}): {np.dot(self.A,self.x)}")
            residual = self.A @ self.x - self.b
            print("Equality constraint residual (Ax - b):", np.round(residual, 5))

        if self.ineq_constraints:
            print("Inequality constraint values (should be ≤ 0):")
            for g in self.ineq_constraints:
                g_val = g(self.x, False)[0]
                print(f"  {g.__name__}: {round(g_val, 5)}")
        print("\n")
        return self.x, f_k, self.history


def interior_pt(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0):
    constrained_min=ConstrainedMin(func, x0, ineq_constraints, eq_constraints_mat, eq_constraints_rhs)
    return constrained_min.outer_iterations(epsilon=EPSILON, mu=MU)




