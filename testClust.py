import matplotlib.pyplot as plt
from clust_cov_sim import clustcovsimGender
from clust_cov_core import cluscov
pi = [0.1, 0.7, 0.2]
clust = clustcovsimGender(pi, N=100)
Data, idx = clust.simData
# print(Data)
clust = cluscov(Data, 3, 3, 2)
pi_N, par, pi, ll = clust.EM_step()
print(pi_N)
print(idx)
print(par)
print(pi)
plt.plot(ll)
plt.show()
