import matplotlib.pyplot as plt
from clust_cov_sim import clustcovsimGender
from clust_cov_core import cluscov
pi = [0.1, 0.7, 0.2]
clust = clustcovsimGender(pi, N=1000)
Data, idx = clust.simData
# print(Data)
clust = cluscov(Data, 3, 3, 2)
idx_gen, par, pi, ll = clust.EM_step()
print(idx_gen)
print(idx)
print(par)
print(pi)
plt.plot(ll)
plt.show()
