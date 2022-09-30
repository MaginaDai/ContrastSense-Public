import numpy as np
import torch
from sklearn.decomposition import PCA


def PCA_numpy(neg_num, key):
    A = torch.einsum('ab,ac->bc', [key, key]).numpy()
    w, v = np.linalg.eig(A)
    idx = np.argsort(w)
    topv = v[:, idx[:-neg_num-1:-1]]
    V = torch.from_numpy(topv)
    c = torch.einsum('ab,bc->ac', [key, V])
    return c


def PCA_torch(neg_num, key):
    U, S, V = torch.pca_lowrank(key, q=neg_num)
    p = torch.matmul(key, V[:, :neg_num])
    return p


# def objective_function():
#     m = Model()
#     n = 4096
#     K = 10
#     y = [m.add_var(var_type=BINARY) for i in range(n)]
#     key = torch.randn(128, 4096)
#     m += sum(y) == K
#     # m.objective = torch.einsum('ab,ac->', [key[:, y], key[:, y]]).numpy()
#     m.objective = xsum(y[i]*y[j]*torch.einsum('i, i->', [key[:, i], key[:, j]]).numpy() for i in range(n) for j in range(n))
#
#     m.max_gap = 0.05
#     status = m.optimize(max_seconds=300)
#     if status == OptimizationStatus.OPTIMAL:
#         print('optimal solution cost {} found'.format(m.objective_value))
#     elif status == OptimizationStatus.FEASIBLE:
#         print('sol.cost {} found, best possible: {}'.format(m.objective_value, m.objective_bound))
#     elif status == OptimizationStatus.NO_SOLUTION_FOUND:
#         print('no feasible solution found, lower bound is: {}'.format(m.objective_bound))
#     if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
#         print('solution:')
#         for v in m.vars:
#             if abs(v.x) > 1e-6:  # only printing non-zeros
#                 print('{} : {}'.format(v.name, v.x))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    key = torch.randn(512, 1024)
    PCA_torch(128, key)
