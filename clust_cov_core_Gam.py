import numpy as np
from scipy.optimize import minimize
from scipy import special


class cluscovGam:
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
        # (G,R) + (R,) = (G,R)
        eta = delta + alpha
        # (G*R,) + (M,) = (G*R,M)
        eta = eta.reshape(-1, 1) + beta
        # (G*R,M) = (G,R,M)
        eta = eta.reshape(self.G, self.R, self.M)
        # (G,R,M) +(R,M) = (G,R,M)
        eta += gam
        # (G*R*M,) + (K,) = (G*R*M,K)
        eta = eta.reshape(-1, 1) + mu
        # Reshape to form tensors (G,R,M,K)
        eta = eta.reshape(self.G, self.R, self.M, self.K)
        # Padding matrix for diff computation
        eta = np.insert(eta, 0, 0, axis=-1)
        # logistic function
        logistic_eta = special.expit(eta)
        # theta_GRMK
        theta = np.diff(logistic_eta)
        return theta

    @staticmethod
    def log_sum_exp(x):
        k = -20
        if len(x.shape) <= 1:
            e = x - np.max(x)
            y = np.exp(e) / sum(np.exp(e))
            y[e < k] = np.exp(k)
            y = y / sum(y)
        else:
            D = x.shape[1]
            e = x - np.max(x, axis=1)
            temp = np.sum(np.exp(e), axis=1)
            temp = np.tile(temp, (D, 1)).T
            y = np.exp(e) / temp
            y[e < k] = np.exp(k)
            y = y / sum(y)
        return y

    def E_step(self, theta, pi):
        # Reshape training data to column vector
        ytrain = self.ytrain[:, :-1].reshape(-1, 1)
        # Reshape gender/feature column to column vector
        x = self.ytrain[:, -1].reshape(-1, 1)
        # Categories of each column
        K = np.arange(1, self.K + 1)
        # Number of categories of feature/gender
        G = np.arange(1, self.G + 1)
        # (N*M,K) check for each row and column the matching category
        I_NMK = ytrain == K
        # (N,G) check for each row the matching feature category
        I_NG = x == G
        # (N,M,K)
        I_NMK = I_NMK.astype(int).reshape(self.N, self.M, self.K)
        # log_theta is (G,R,M,K)
        log_theta = np.log(theta)
        # (N,G,R)
        log_pi_N = np.tensordot(I_NMK, log_theta, ((1, 2), (2, 3)))
        # (R,G,N)
        log_pi_N = np.swapaxes(log_pi_N, 0, 2)
        # (R,N,G)
        log_pi_N = np.swapaxes(log_pi_N, 1, 2)
        log_pi_N *= I_NG
        # (R,N)
        log_pi_N = np.sum(log_pi_N, axis=2)
        # (N,R)
        log_pi_N = log_pi_N.T
        log_pi_N_out = log_pi_N
        log_pi_N += np.log(pi)
        pi_N = self.log_sum_exp(log_pi_N)
        return pi_N, log_pi_N_out

    def negloglik(self, par, pi, pi_N, log_pi_N):
        ll = -np.sum(pi_N * (np.log(pi) + log_pi_N))
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
                res = minimize(self.negloglik, par0, args=(pi0, pi_N, log_pi_N), method='L-BFGS-B', options={'disp': False})
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
            res = minimize(self.negloglik, par, args=(pi, pi_N, log_pi_N), method='L-BFGS-B', options={'disp': False})
            par = res.x
            pi = np.sum(pi_N, axis=0) / np.sum(pi_N)
            count += 1
