"""
Sparse optimization.
"""
import numpy as np
from scipy.optimize import minimize
from icecream import ic
from contexttimer import Timer


def solve_l1(y, theta, maxiter=300):
    """Solve the problem:

        min ||s||_1 s.t. y = theta @ s,

    using SLSQP.

    Args:
        y: Sample vector.
        theta: Library matrix.

    Returns:
        s: Sparse vector.
    """
    l1 = lambda x: np.linalg.norm(x, 1)
    constraints = {'type': 'eq', 'fun': lambda x: theta @ x - y}
    s0 = solve_l2(y, theta)
    s = minimize(l1, s0, method='SLSQP', constraints=constraints, options=dict(maxiter=maxiter))
    return s


def solve_l2(y, theta):
    """Solve the problem:

        min ||x||_2 s.t. y = theta @ x

    Args:
        y: Sample vector.
        theta: Library matrix.

    Returns:
        x: L2 solution.
    """
    return np.linalg.pinv(theta) @ y


def cosamp(phi, u, s, epsilon=1e-10, max_iter=1000):
    """Return an `s`-sparse approximation of the target signal.
    
    Args:
        phi: sampling matrix.
        u: noisy sample vector.
        s: sparsity level.

    Adapted from: https://github.com/avirmaux/CoSaMP
    """
    a = np.zeros(phi.shape[1])
    v = u
    it = 0 # count
    halt = False
    while True:
        it += 1
        
        y = np.dot(np.transpose(phi), v)
        omega = np.argsort(y)[-(2*s):] # large components
        omega = np.union1d(omega, a.nonzero()[0])
        phiT = phi[:, omega]
        b = np.zeros(phi.shape[1])
        # Solve Least Square
        b[omega], _, _, _ = np.linalg.lstsq(phiT, u)
        
        # Get new estimate
        b[np.argsort(b)[:-s]] = 0
        a = b
        
        # Halt criterion
        v_old = v
        v = u - np.dot(phi, a)

        if (np.linalg.norm(v - v_old) < epsilon) or \
            np.linalg.norm(v) < epsilon:
            ic(f'Converged. Iteration {it}.')
            break

        if it > max_iter:
            ic(f'Max. iterations reached: {max_iter}.')
            break
        
    return a
    

if __name__ == '__main__':
    n = 1200
    p = 300
    theta = np.random.randn(p, n)
    y = np.random.randn(p)
    ic(y.shape, theta.shape)
    with Timer() as t:
        s = solve_l1(y, theta, maxiter=10000)
        print(n, p, s, t.elapsed)

    from matplotlib import pyplot as plt
    plt.plot(s.x)
    plt.show()

    plt.hist(s.x)
    plt.show()
