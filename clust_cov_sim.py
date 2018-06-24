import numpy as np
from scipy import special


class clustcovsimGender:
    """Class for simulating data"""

    def __init__(self, pi, K=3, M=5, N=100):
        # Number of clusters
        self.R = len(pi)
        # Number of categories
        self.K = K
        # Number of columns
        self.M = M
        # Number of rows
        self.N = N
        # clsuter proportion
        self.pi = pi
        # number of parameters
        self.D = 2 * self.R + self.R - 1 + M - 1 + K - 1

    @property
    def simData(self):
        Data = np.zeros((self.N, self.M + 1))
        beta = np.zeros((2, self.M))
        # beta[0, :-1] = np.random.randint(self.M - 1)
        # beta[1, :-1] = -np.random.rand(self.M - 1)
        beta[0, :] = np.array([2, 6, 3, 1, -12]) / 10
        beta[1, :] = np.array([-1, -4, -3, -2, 10]) / 10
        # alpha = np.random.rand(self.R)
        alpha = np.array([3, 2, -5]) / 10
        # temp = np.random.rand(self.K)
        mu = np.array([2, 5, 0]) / 10
        theta = np.zeros(self.K)
        rx = np.zeros(self.N)
        for i in range(self.N):
            if np.random.rand(1) > 0.5:
                betatemp = beta[1]
                Data[i, - 1] = 2
            else:
                betatemp = beta[0]
                Data[i, -1] = 1
            idx = np.random.multinomial(1, self.pi).astype(bool)
            alpha_temp = alpha[idx]
            rx[i] = np.argmax(idx.astype(int))
            for m in range(self.M):
                eta = mu + betatemp[m] + alpha_temp
                theta_temp = special.expit(eta)
                theta[0] = theta_temp[0]
                theta[1] = theta_temp[1] - theta_temp[0]
                theta[-1] = 1 - np.sum(theta)
                theta[theta <= 0] = 0.0001
                Data[i, m] = np.argmax(np.random.multinomial(1, theta))
        return Data, rx
