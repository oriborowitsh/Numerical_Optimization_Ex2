import numpy as np

def quadratic(x, evaluate_hessian, Q, q, c):
    obj_val = 0.5*np.dot(np.dot(x.T, Q), x)+ np.dot(q.T,x)+c
    grad=np.dot(Q,x)+q
    hessian = None
    if evaluate_hessian:
        hessian = Q
    return obj_val,grad , hessian

def quadratic_example_f(x, evaluate_hessian):
    Q = 2 * np.identity(3)
    q = np.array([0.0, 0.0, 2.0])
    c = 1.0
    return quadratic(x, evaluate_hessian, Q, q, c)

quadratic_example_lhs=np.array([[1,1,1]])
quadratic_example_rhs=1

def quadratic_example_g1(x, compute_hessian):
    return -x[0], np.array([-1.0, 0.0, 0.0]), np.zeros((3, 3))
def quadratic_example_g2(x, compute_hessian):
    return -x[1], np.array([0.0, -1.0, 0.0]), np.zeros((3, 3))
def quadratic_example_g3(x, compute_hessian):
    return -x[2], np.array([0.0, 0.0, -1.0]), np.zeros((3, 3))


def linear_example_f(x, compute_hessian):
    return quadratic(x, compute_hessian, np.zeros((2,2)), np.array([-1.0,-1.0]), 0.0)

def linear_example_g1(x, compute_hessian):
    return quadratic(x, compute_hessian, np.zeros((2,2)), np.array([-1.0, -1.0]), 1.0)

def linear_example_g2(x, compute_hessian):
    return quadratic(x, compute_hessian, np.zeros((2,2)), np.array([0.0, 1.0]), -1.0)

def linear_example_g3(x, compute_hessian):
    return quadratic(x, compute_hessian, np.zeros((2,2)), np.array([1.0, 0.0]), -2.0)

def linear_example_g4(x, compute_hessian):
    return quadratic(x, compute_hessian, np.zeros((2,2)), np.array([0.0, -1.0]), 0.0)