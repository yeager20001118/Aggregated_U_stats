import torch
cov_matrix = torch.tensor([[4.1796e-02, 1.6543e-03, 5.3318e-02, 6.9098e-06],
        [1.6543e-03, 8.6678e-04, 1.6481e-03, 6.5070e-06],
        [5.3318e-02, 1.6481e-03, 7.5120e-02, 7.1990e-06],
        [6.9098e-06, 6.5070e-06, 7.1990e-06, 1.1287e-05]])

variances = torch.diag(cov_matrix)

n = cov_matrix.shape[0]
corr_matrix = torch.zeros_like(cov_matrix)

for i in range(n):
    for j in range(n):
        corr_matrix[i, j] = cov_matrix[i, j] / \
            (variances[i])

print(corr_matrix)

corr_matrix = 1 - corr_matrix / torch.sqrt(torch.diag(corr_matrix).sum())
print("Correlation Matrix:")
print(corr_matrix)