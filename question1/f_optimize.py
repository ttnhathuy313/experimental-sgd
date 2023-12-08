import numpy as np
import numpy.linalg as la
import time
from copy import deepcopy


def calculateGrad(x, f, calcHessian=False, diff=1e-5):
    """

    :param x: shape (D)
    :param f: function takes x as input and return value shape (M)
    :param diff: different before after (differential)
    :return: gradient of f at x with shape (D)
    """

    dimInput = x.shape[0]
    dimOutput = f(x).shape[0]

    if (calcHessian):
        grad = np.zeros((dimOutput, dimInput))
    elif (dimOutput == 1):
        grad = np.zeros(dimInput, )
    elif (dimOutput > 1):
        grad = np.zeros((dimOutput, dimInput))

    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])

    while not it.finished:
        ix = it.multi_index

        oldVal = x[ix]
        x[ix] = oldVal + diff
        pos = f(x).copy()
        x[ix] = oldVal - diff
        neg = f(x).copy()
        x[ix] = oldVal
        grad[ix] = (pos - neg) / (2 * diff)
        it.iternext()

    return grad


def sgdUpdate(x, grad, stepSize, config):
    return x - stepSize * grad(x), config


def momentumUpdate(x, grad, stepSize, config):
    m = config['momentum']
    beta = config['damping coefficient']

    m = (beta * m) - (stepSize * grad(x))
    updatedX = x + m

    config['momentum'] = m

    return updatedX, config


def nesterovUpdate(x, grad, stepSize, config):
    m = config['momentum']
    beta = config['damping coefficient']

    m = (beta * m) - (stepSize * grad(x + (beta * m)))
    updatedX = x + m

    config['momentum'] = m

    return updatedX, config


def adamUpdate(x, grad, stepSize, config):
    beta1 = config['beta1']
    beta2 = config['beta2']
    epsilon = config['epsilon']

    m = config['m']
    v = config['v']
    t = config['t']

    t = t + 1

    g = grad(x)

    m = (beta1 * m) + ((1 - beta1) * g)
    mt = m / (1 - (beta1 ** t))
    v = (beta2 * v) + ((1 - beta2) * (g ** 2))
    vt = v / (1 - (beta2 ** t))
    updatedX = x - (stepSize * mt / (np.sqrt(vt) + epsilon))

    config['t'] = t

    config['m'] = m
    config['v'] = v

    return updatedX, config


def back_tracking(f, df, x, p, alpha0, rho, c):
    f_value = f(x)
    second_term = c * np.dot(df(x), p)
    alpha = alpha0
    while (f(x + alpha * p) > f_value + alpha * second_term):
        alpha = alpha * rho
    return alpha


def f_optimize(f, df, x0, time_limit):
    start_time = time.time()

    max_iterations = 100000
    tol = 1e-8

    x = x0  # initial solution
    xPrev = deepcopy(x) + 1e2

    iteration_count = 0

    # hessian = calculateGrad(x, df, calcHessian = True)
    # invHessian = la.pinv(hessian)

    sgdConfig = {}

    momentumConfig = {
        'momentum': np.zeros_like(x),

        'damping coefficient': 0.9
    }

    adamConfig = {
        't': 0,

        'm': np.zeros_like(x),
        'v': np.zeros_like(x),

        'beta1': 0.9,
        'beta2': 0.999,
        'epsilon': 1e-8
    }

    config = adamConfig

    current_time = time.time()

    while ((current_time - start_time < time_limit)
           & (iteration_count < max_iterations)
           & (np.linalg.norm(x - xPrev, 2) > tol)):
        # alpha = back_tracking(f, df, x, -gradient, 1.0, 0.5, 1e-3)
        alpha = 1e-3

        xPrev = deepcopy(x)
        x, config = adamUpdate(x, df, alpha, config)

        # x = x - alpha * (invHessian.dot(df(x))) # Newton
        # hessian = calculateGrad(x, df, calcHessian = True)
        # invHessian = la.pinv(hessian)

        iteration_count = iteration_count + 1

        current_time = time.time()

    print('After: %d iterations - %.8f seconds' % (iteration_count, current_time - start_time))
    print('Objective value: %.8f' % (f(x)))

    return x