import numpy as np
import numpy.linalg as la
import time
from copy import deepcopy


def calculateGrad(x, f, calcHessian=False, diff=1e-5):
    """
    Calculate gradient of a multivariable function, the function can be scalar-valued or vector-valued.
    This works well with second order, just use gradient function as the input function. But remember
    to specify calcHessian to adjust for dimensionality.

    This is inspired by CS231n homework 1, see the link https://cs231n.github.io/

    :param x: shape (D)
    :param f: function takes x as input and return value shape (M)
    :param diff: different before after (differential), see more below
    :return: gradient of f at x with shape (M, D)
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
    """
    Perform an update using sgd/gd rule

    :param x: the initial point shape (D)
    :param grad: gradient function to calculate, that is we have to call grad(x)
    :param stepSize: a constant learning rate
    :param config: This dictionary is left as None but will be used for other optimizers.
    (We put it as empty to synchronize with the design of other optimizers. So we do not have
    to go back and delete too much)
    :return: tuple of size (2,). The first element will be the new_x of shape (D), second element
    will be the config dictionary.
    """

    return x - stepSize * grad(x), config


def momentumUpdate(x, grad, stepSize, config):
    """
    Perform an update using momentum rule

    :param x: the initial point shape (D)
    :param grad: gradient function to calculate, that is we have to call grad(x)
    :param stepSize: a constant learning rate
    :param config: Dictionary contains 'momentum' and 'damping coefficient' hyperparameters
    for momentum optimizer. The values are corresponding to the keys above.
    :return: tuple of size (2,). The first element will be the new_x of shape (D), second element
    will be the config dictionary.
    """
    m = config['momentum']
    beta = config['damping coefficient']

    m = (beta * m) - (stepSize * grad(x))
    updatedX = x + m

    config['momentum'] = m

    return updatedX, config


def nesterovUpdate(x, grad, stepSize, config):
    """
    Perform an update using Nesterov rule

    :param x: the initial point shape (D)
    :param grad: gradient function to calculate, that is we have to call grad(x)
    :param stepSize: a constant learning rate
    :param config: Dictionary contains 'momentum' and 'damping coefficient' hyperparameters
    for Nesterov optimizer. The values are corresponding to the keys above.
    :return: tuple of size (2,). The first element will be the new_x of shape (D), second element
    will be the config dictionary.
    """
    m = config['momentum']
    beta = config['damping coefficient']

    m = (beta * m) - (stepSize * grad(x + (beta * m)))
    updatedX = x + m

    config['momentum'] = m

    return updatedX, config


def adamUpdate(x, grad, stepSize, config):
    """
    Perform an update using Adam rule optimizer

    :param x: the initial point shape (D)
    :param grad: gradient function to calculate, that is we have to call grad(x)
    :param stepSize: a constant learning rate
    :param config: Dictionary contains hyperparameters for Adam optimizer.
    There will be 'beta1', 'beta2', 'epsilon' are main hyperparameters.
    The memory will include 'm', 'v' and 't' and all are initialized zeros.
    :return: tuple of size (2,). The first element will be the new_x of shape (D), second element
    will be the config dictionary.
    """
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
    """
    Perform back tracking line search (exact line search method).
    :param f:
    :param df:
    :param x:
    :param p:
    :param alpha0:
    :param rho:
    :param c:
    :return:
    """
    f_value = f(x)
    second_term = c * np.dot(df(x), p)
    alpha = alpha0
    while (f(x + alpha * p) > f_value + alpha * second_term):
        alpha = alpha * rho
    return alpha


def f_optimize(f, df, x0, time_limit):
    """
    Trying to optimize (minimize) a (multivariable) scalar-valued function in a specific time limit.
    Currently: using Adam. The function will stop if time limit exceeded or norm of gradient close to zero
    for a specific threshold.

    :param f: a (multivariable) scalar-valued function that takes in an input of shape (D, ) and
    return a scalar value shape (1, )
    :param df: gradient function to calculate, that is we have to call grad(x)
    :param x0: initial starting point of shape (D, )
    :param time_limit: time limiting for optimizing the function
    :return: x solution output of shape (D, ).
    """

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