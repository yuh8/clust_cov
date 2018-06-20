import numpy as np
import scipy as sp


class cluscov:
    def __init__(self, Ytrain, K, R, G):
        self.K = K
        self.R = R
        self.Ytrain = Ytrain
        self.N = np.size(Ytrain, axis=0)
        # The last column is the gender
        self.M = np.size(Ytrain, axis=1) - 1
        self.G = G

    def computeTheta(self, par):
        # size of par is  G*R + R + R*M + M + K
        # par is flattened for scipy.optimize.minimize
        temp1 = self.G * self.R
        temp2 = self.R * self.M
        # (G,R)
        delta = par[:temp1].reshape(self.G, self.R)
        # (R,)
        alpha = par[temp1:temp1 + self.R]
        temp1 += self.R
        # (R,M)
        gam = par[temp1:temp1 + temp2].reshape(self.R, self.M)
        temp1 += temp2
        # (M,)
        beta = par[temp1:temp1 + self.M]
        temp1 += self.M
        # (K,)
        mu = par[temp1:temp1 + self.K]
        temp1 += self.K
        # Columnize + add + reshape
        eta = delta + alpha
        eta = eta.reshape(-1, 1) + beta
        eta = eta.reshape(self.G, self.R, self.M)
        eta += gam
        eta = eta.reshape(-1, 1) + mu
        # Reshape to form tensors (G,R,M,K)
        eta = eta.reshape(self.G, self.R, self.M, self.K)
        # insert zero column before
        eta = np.insert(eta, 0, 0, axis=-1)
        logistic_eta = sp.special.expit(eta)
        theta = np.diff(logistic_eta)
        return theta

    @staticmethod
    def log_sum_exp(x):
        k = np.exp(-20)
        e = x - np.max(x)
        y = np.exp(e) / sum(np.exp(e))
        y[e < k] = np.exp(k)
        y = y / sum(y)
        return y

    def E_step(self, theta, pi):
        ytrain = self.ytrain[:, :-1].reshape(-1, 1)
        x = self.ytrain[:, -1].reshape(-1, 1)
        K = np.arange(1, self.K + 1)
        G = np.arange(1, self.G + 1)
        I_NMK = ytrain == K
        I_NG = x == G
        I_NMK = I_NMK.astype(int).reshape(self.N, self.M, self.K)
        I_NG = I_NG.astype(int).reshape(self.N, self.G)
        # log_theta is (G,R,M,K)
        log_theta = np.log(theta)
        # (N,G,R)
        log_pi_N = np.tensordot(I_NMK, log_theta, ((1, 2), (2, 3)))
        # (N,R)
        log_pi_N = np.sum(log_pi_N, 1)
        log_pi_N_out = log_pi_N
        log_pi_N += np.log(pi)
        pi_N = self.log_sum_exp(log_pi_N)
        return pi_N, log_pi_N_out

    def negloglik(self, par, pi, pi_N, log_pi_N):
        ll = -np.sum(pi_N * np.log(pi)) - np.sum(pi_N * log_pi_N)
        return ll

    def EM_step(self, nstarts=10, itermax=100):
        count = 0
        iter_burn = 5
        min_fun = np.inf
        # Burnin
        while count < nstarts:
            par0 = np.random.randn(self.G * self.R + self.R + self.R * self.M + self.M + self.K)
            pi0 = np.random.dirichlet(np.ones(self.R) / self.R)
            count1 = 0
            while count1 < iter_burn:
                theta = self.computeTheta(par0)
                pi_N, log_pi_N = self.E_step(theta, pi0)
                res = sp.optimize.minimize(self.negloglik, par0, args=(pi0, pi_N, log_pi_N), method='L-BFGS-B', options={'disp': False})
                par0 = res.x
                pi0 = np.sum(pi_N, axis=0) / np.sum(pi_N)
                count1 += 1
            # Choose the start yielding the Max LL
            if res.fun < min_fun:
                min_fun = res.fun
                par = res.x
                pi = pi0
            count += 1

        # Final tune
        count = 0
        while count < itermax:
            theta = self.computeTheta(par)
            pi_N, log_pi_N = self.E_step(theta, pi)
            res = sp.optimize.minimize(self.negloglik, par, args=(pi, pi_N, log_pi_N), method='L-BFGS-B', options={'disp': False})
            par = res.x
            pi = np.sum(pi_N, axis=0) / np.sum(pi_N)
            count += 1
