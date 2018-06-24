import numpy as np
from scipy import special
from scipy.optimize import minimize


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
        # size of par is  G*R + (R-1) + (M-1) + (K-1)
        # par is flattened for scipy.optimize.minimize
        alpha = np.zeros(self.R)
        beta = np.zeros(self.M)
        temp = self.G * self.R
        # (G,R)
        delta = par[:temp].reshape(self.G, self.R)
        # (R-1,)
        alpha[:-1] = par[temp:temp + self.R - 1]
        # sum to zero constraint
        alpha[-1] = -np.sum(alpha)
        temp += self.R - 1
        # (M-1,)
        beta[:-1] = par[temp:temp + self.M - 1]
        # sum to zero constraint
        beta[-1] = -np.sum(beta)
        temp += self.M - 1
        # (K,) ensure monotonicity of mu
        mu = np.zeros(self.K - 1)
        mu[0] = par[temp]
        mu[1:] = np.exp(par[temp + 1:])
        mu = np.cumsum(mu)
        # columnize + add + reshape
        # (G,R) + (R,) = (G,R)
        eta = delta + alpha
        # (G*R,) + (M,) = (G*R,M)
        eta = eta.reshape(-1, 1) + beta
        # (G*R,M) = (G,R,M)
        eta = eta.reshape(self.G, self.R, self.M)
        # (G*R*M,) + (K-1,) = (G*R*M,K-1)
        eta = eta.reshape(-1, 1) + mu
        # reshape to form tensors (G,R,M,K-1)
        eta = eta.reshape(self.G, self.R, self.M, self.K - 1)
        # padding matrix for diff computation
        eta = np.insert(eta, 0, 0, axis=-1)
        # logistic function
        logistic_eta = special.expit(eta)
        # theta_GRMK
        theta = np.diff(logistic_eta)
        # sum to one constraint
        temp1 = 1 - np.sum(theta, axis=-1).reshape(self.G, self.R, self.M, 1)
        # append to the last column theta is (G,R,M,K)
        theta = np.append(theta, temp1, axis=-1)
        theta[theta <= 0] = 0.00001
        return theta

    @staticmethod
    def log_sum_exp(x):
        k = -100
        if len(x.shape) <= 1:
            e = x - np.max(x)
            y = np.exp(e) / sum(np.exp(e))
            y[e < k] = 0
            y = y / sum(y)
        else:
            D = x.shape[1]
            e = x - np.max(x, axis=1).reshape(-1, 1)
            temp = np.sum(np.exp(e), axis=1)
            temp = np.tile(temp, (D, 1)).T
            y = np.exp(e) / temp
            y[e < k] = 0
            y = y / np.sum(y, axis=1).reshape(-1, 1)
        return y

    def E_step(self, theta, pi):
        # reshape training data to column vector
        Ytrain = self.Ytrain[:, :-1].reshape(-1, 1)
        # reshape gender/feature column to column vector
        x = self.Ytrain[:, -1].reshape(-1, 1)
        # categories of each column
        K = np.arange(0, self.K)
        # Number of categories of feature/gender
        G = np.arange(1, self.G + 1)
        # (N*M,K) check for each row and column the matching category
        I_NMK = Ytrain == K
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
        # (N,R) + (R,)
        log_pi_N += np.log(pi)
        pi_N = self.log_sum_exp(log_pi_N)
        return pi_N, log_pi_N

    def negloglik(self, par, pi, pi_N):
        theta = self.computeTheta(par)
        _, log_pi_N = self.E_step(theta, pi)
        ll = -np.sum(pi_N * log_pi_N)
        return ll

    def EM_step(self, nstarts=100, itermax=500):
        count = 0
        iter_burn = 1
        min_fun = np.inf
        # Burnin
        while count < nstarts:
            par0 = np.random.rand(self.G * self.R + self.R - 1 + self.M - 1 + self.K - 1)
            pi0 = np.random.dirichlet(np.ones(self.R) / self.R)
            count1 = 0
            while count1 < iter_burn:
                theta = self.computeTheta(par0)
                pi_N, _ = self.E_step(theta, pi0)
                res = minimize(self.negloglik, par0, args=(pi0, pi_N), method='L-BFGS-B', options={'disp': False})
                par0 = res.x
                pi0 = np.sum(pi_N, axis=0) / np.sum(pi_N)
                count1 += 1
            # Choose the start yielding the Max LL
            if not res.success:
                continue
            if res.fun < min_fun:
                min_fun = res.fun
                par = par0
                pi = pi0
            count += 1

        # Final tune
        count = 0
        ll = np.zeros(itermax)
        while count < itermax:
            theta = self.computeTheta(par)
            pi_N, _ = self.E_step(theta, pi)
            res = minimize(self.negloglik, par, args=(pi, pi_N), method='L-BFGS-B', options={'disp': False})
            par = res.x
            pi = np.sum(pi_N, axis=0) / np.sum(pi_N)
            ll[count] = res.fun
            count += 1
        return np.argmax(pi_N, axis=1), par, pi, ll
