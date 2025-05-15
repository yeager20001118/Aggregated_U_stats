import torch
import numpy as np
import time
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath('/data/gpfs/projects/punim2335/baselines'))
from dataloader_TST import load_data
from torchvision.models import resnet18
from torch.autograd import Function
import scipy
import sys
import os
sys.path.append(os.path.abspath('..'))


class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.
    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """
    @staticmethod
    def forward(ctx, input):
        m = input.detach().cpu().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)
            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


sqrtm = MatrixSquareRoot.apply

KERNEL_SET = ["gaussian", "laplacian", "deep_gaussian", "deep_laplacian"]


def generate_kernel_bandwidth(n_bandwidth, X, Y, reg, is_cov, way, device):
    assert set(way) <= {'Agg', 'Boost', 'Grid'}
    k_b_pair = []
    Z = torch.cat([X, Y], dim=0)
    # dist = torch.cdist(Z, Z, p=2)
    # dist[dist < 0] = 0
    # for i in range(len(n_bandwidth)):
    #     kernel = KERNEL_SET[i]
    #     if way[i] == 'Boost':
    #         bandwidths = get_bandwidth_boost(
    #             n_bandwidth[i], torch.median(dist).item())
    #         for b in bandwidths:
    #             k_b_pair.append((kernel, b))
    #     elif way[i] == 'Grid':
    #         k_b_pair = get_bandwidth_grid(
    #             X, Y, k_b_pair, kernel, n_bandwidth[i], reg, is_cov, device)
    #     elif way[i] == 'Agg':
    #         m = dist.shape[0]
    #         indices = torch.triu_indices(m, m, offset=0)
    #         dist_v = dist[indices[0], indices[1]]
    #         bandwidths = get_bandwidth_agg(dist_v, n_bandwidth[i])
    #         for b in bandwidths:
    #             k_b_pair.append((kernel, b))

    # test
    # k_b_pair = []
    k_b_pair.append((KERNEL_SET[0], torch.tensor(0.072).to(Z.device)))
    # k_b_pair.append((KERNEL_SET[0], torch.tensor(0.049).to(Z.device)))
    # k_b_pair.append((KERNEL_SET[0], torch.tensor(0.009).to(Z.device)))
    # k_b_pair.append((KERNEL_SET[0], torch.tensor(0.11).to(Z.device)))
    # k_b_pair.append((KERNEL_SET[0], torch.tensor(0.658).to(Z.device)))
    k_b_pair.append((KERNEL_SET[1], torch.tensor(0.00074).to(Z.device)))
    return k_b_pair


def get_bandwidth_agg(dist, n_bandwidth):
    device = dist.device
    # Replace zero distances with median
    dist = dist + (dist == 0) * torch.median(dist)
    # Sort distances
    dd = torch.sort(dist)[0]  # torch.sort returns (values, indices)
    idx = torch.floor(torch.tensor(len(dd) * 0.05)).to(torch.int64)
    if torch.min(dist) < 10 ** (-1):
        lambda_min = torch.maximum(
            dd[idx],
            torch.tensor(10 ** (-1)).to(device)
        )
    else:
        lambda_min = torch.min(dist)
    # Adjust lambda_min and compute lambda_max
    lambda_min = lambda_min / 2
    lambda_max = torch.maximum(
        torch.max(dist),
        torch.tensor(3 * 10 ** (-1)).to(device)
    )
    lambda_max = lambda_max * 2
    # Compute power sequence
    power = (lambda_max / lambda_min) ** (1 / (n_bandwidth - 1))
    # Generate geometric sequence of bandwidths
    bandwidths = torch.pow(power, torch.arange(
        n_bandwidth, device=device)) * lambda_min

    return bandwidths


def get_bandwidth_grid(X, Y, k_b_pair, kernel, n_bandwidth, reg, is_cov, device):
    Dxy = torch.cdist(X, Y, p=2)
    Dxy[Dxy < 0] = 0
    list_bandwidths = Dxy.median() * (2.0 ** torch.linspace(0, 100,
                                                            500).to(device, torch.float32))
    list_bandwidths = list_bandwidths.sort()[0].reshape(-1, 1)
    bandwidths = torch.zeros(n_bandwidth, device=device, dtype=torch.float32)
    for k in range(n_bandwidth):
        MAXMUM = 0
        for i in range(len(list_bandwidths)):
            model_au = MMD_AU(
                X, Y, k_b_pair+[(kernel, list_bandwidths[i])], reg, is_cov).to(device, torch.float32)
            U = model_au.compute_U_stats()
            Sigma = model_au.compute_Sigma()
            try:
                T = len(X)**2 * U.T @ torch.inverse(Sigma) @ U
                if T.item() > MAXMUM and list_bandwidths[i] not in bandwidths:
                    MAXMUM = T.item()
                    bandwidths[k] = list_bandwidths[i]
            except:
                pass
        k_b_pair.append((kernel, bandwidths[k]))
    return k_b_pair


def get_bandwidth_boost(n_bandwidth, median):
    scales = get_scales(n_bandwidth)
    return scales * median


def get_scales(n_bandwidth, base=2):
    ratio = torch.sqrt(torch.Tensor([base]))
    if n_bandwidth % 2 == 1:  # odd n
        half = (n_bandwidth - 1) // 2
        powers = torch.tensor(list(range(-half, half + 1)), dtype=torch.float)
    else:  # even n
        half = n_bandwidth // 2
        powers = torch.tensor(
            [-half + i + 0.5 for i in range(n_bandwidth)], dtype=torch.float)
    scales = torch.pow(ratio, powers)
    return scales


def kernel_matrix(kernel, bandwidth, X, Y=None, dist=None, model=None):
    # If Y is not provided, use X
    if Y is None:
        Y = X
    # Compute pairwise distances
    if dist is None:
        dist = torch.cdist(X, Y, p=2)  # Euclidean distance (L2 norm)
        dist[dist < 0] = 0
    # Compute kernel matrix based on the specified kernel
    if kernel.lower() == "gaussian":
        # K_GAUSS(x,y) = exp(-||x-y||²/σ²)
        K = torch.exp(-torch.pow(dist, 2) / torch.pow(bandwidth, 2))
    elif kernel.lower() == "laplacian":
        # K_LAP(x,y) = exp(-||x-y||/σ)
        K = torch.exp(-dist / bandwidth)
    elif kernel.lower()[:4] == "deep":
        if model is None:
            raise ValueError("Model must be provided for deep kernel.")
        X_rep = model(X)
        Y_rep = model(Y)
        dist = torch.cdist(X_rep, Y_rep, p=2)
        dist[dist < 0] = 0
        K = kernel_matrix(kernel[5:], bandwidth, X_rep, dist=dist)
    else:
        raise ValueError(
            f"Unknown kernel: {kernel}. Use 'gaussian' or 'laplacian'.")
    return K


def check_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def mmd_u(K, n):
    """
    Compute the unbiased MMD^2 statistic given the kernel matrix K, if m = n.
    """
    # Extract submatrices for XX, YY, and XY
    K_XX = K[:n, :n]
    K_YY = K[n:, n:]
    K_XY = K[:n, n:]
    # Ensure diagonal elements are zero for XX, YY
    K_XX = K_XX - torch.diag(torch.diag(K_XX))
    K_YY = K_YY - torch.diag(torch.diag(K_YY))
    # Remove diagonal from K_XY (where i=j)
    K_XY = K_XY - torch.diag(torch.diag(K_XY))
    h_matrix = K_XX + K_YY - K_XY - K_XY.t()
    # Calculate each term of the MMD_u^2
    mmd_u_squared = h_matrix.sum() / (n * (n - 1))
    return mmd_u_squared, h_matrix


def MatConvert(x, device, dtype):
    """convert the numpy to a torch tensor."""
    x = torch.from_numpy(x).to(device, dtype)
    return x


class MMD_AU(nn.Module):
    def __init__(self, X, Y, k_b_pair, reg=1e-5, is_cov=True, is_cov_tr=True, encoder=None):
        super(MMD_AU, self).__init__()
        self.X = X
        self.Y = Y
        self.k_b_pair = k_b_pair
        self.kernels = [pair[0] for pair in k_b_pair]
        self.bandwidths = nn.ParameterList()
        self.encoder = encoder
        self.is_cov = is_cov
        self.is_cov_tr = is_cov_tr
        self.reg = reg

        for _, bandwidth in k_b_pair:
            self.bandwidths.append(nn.Parameter(bandwidth.clone()))

    def forward():
        pass

    def compute_U_stats(self, X_test=None, Y_test=None, n_per=None):
        device = self.X.device
        if X_test is not None and Y_test is not None:
            Z = torch.cat([X_test, Y_test], dim=0)
            n = len(X_test)
        else:
            Z = torch.cat([self.X, self.Y], dim=0)
            n = len(self.X)
        dist = torch.cdist(Z, Z, p=2)
        dist[dist < 0] = 0
        n_b = len(self.bandwidths)

        U = []  # U.size() = (c, 1) column vector
        U_b = []  # U_b.size() = (c, n_per)

        if n_per is not None:
            B = torch.randint(0, 2, (n_per, n), dtype=torch.float).to(
                device) * 2.0 - 1.0
            B = torch.einsum('bi,bj->bij', B, B)
        for i in range(n_b):
            K = kernel_matrix(
                self.kernels[i], self.bandwidths[i], Z, dist=dist, model=self.encoder)
            mmd, h_matrix = mmd_u(K, n)
            U.append(mmd)
            if n_per is not None:
                mmd_b = torch.sum(B*h_matrix, dim=(1, 2)) / (n * (n - 1))
                U_b.append(mmd_b)

        U = torch.stack(U)
        if n_per is not None:
            U_b = torch.stack(U_b)  # U_b.size() = (c, n_per) matrix
            return U.unsqueeze(1), U_b

        return U.unsqueeze(1)

    def compute_Sigma(self, X_test=None, Y_test=None):
        device = self.X.device
        n_b = len(self.bandwidths)
        if not self.is_cov:
            return torch.eye(n_b).to(device, torch.float32)

        if X_test is not None and Y_test is not None:
            Z = torch.cat([X_test, Y_test], dim=0)
            n = len(X_test)
        else:
            Z = torch.cat([self.X, self.Y], dim=0)
            n = len(self.X)

        torch.manual_seed(Z.size(0))
        indices = torch.randperm(Z.size(0), device=Z.device)
        Z_null = Z[indices]
        dist = torch.cdist(Z_null, Z_null, p=2)
        dist[dist < 0] = 0
        Sigma = torch.zeros(n_b, n_b).to(device, torch.float32)
        C = self.get_C(2, Z_null.size(0))
        for i in range(n_b):
            for j in range(i, n_b):
                K1 = kernel_matrix(
                    self.kernels[i], self.bandwidths[i], Z_null, dist=dist, model=self.encoder)
                _, h_matrix1 = mmd_u(K1, n)

                K2 = kernel_matrix(
                    self.kernels[j], self.bandwidths[j], Z_null, dist=dist, model=self.encoder)
                _, h_matrix2 = mmd_u(K2, n)

                mask = torch.triu(torch.ones(n, n), diagonal=1).to(device)

                Sigma[i, j] = C * (h_matrix1 * h_matrix2 * mask).sum()
                Sigma[j, i] = Sigma[i, j]
        return Sigma + self.reg * torch.eye(n_b).to(device, torch.float32)

    def train_kernel(self, N_epoch, batch_size, learning_rate):
        print(self.parameters)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        n = self.X.size(0)
        batches = n//batch_size
        with torch.no_grad():
            U = self.compute_U_stats()
            Sigma = self.compute_Sigma()
            T = n**2 * U.T @ torch.inverse(Sigma) @ U
            print(f"Epoch 0: T = {T.item()}")
        for epoch in range(N_epoch):
            indices = torch.randperm(n)
            X = self.X[indices]
            indices = torch.randperm(n)
            Y = self.Y[indices]
            for idx in range(batches):
                if n - idx*batch_size < batch_size:
                    break
                optimizer.zero_grad()
                U = self.compute_U_stats(
                    X_test=X[idx*batch_size:idx*batch_size+batch_size], Y_test=Y[idx*batch_size:idx*batch_size+batch_size])
                Sigma = self.compute_Sigma(
                    X_test=X[idx*batch_size:idx*batch_size+batch_size], Y_test=Y[idx*batch_size:idx*batch_size+batch_size])
                T = batch_size**2 * U.T @ torch.inverse(Sigma) @ U
                loss = - T
                loss.backward(retain_graph=True)
                optimizer.step()
            if (epoch+1) % 100 == 0:
                with torch.no_grad():
                    U = self.compute_U_stats()
                    Sigma = self.compute_Sigma()
                    T = n**2 * U.T @ torch.inverse(Sigma) @ U
                    print(f"Epoch {epoch+1}: T = {T.item()}")

    def get_C(self, m, n):
        import math
        return (
            (n**2)
            / (math.comb(n, m))
            * math.comb(m, 2)
            * math.comb(n - m, m - 2)
            / (math.comb(n - 2, m - 2) ** 2)
            / math.comb(n, 2)
        )

    def get_kernel_bandwidth_pairs(self):
        k_b_pair = [(kernel, bandwidth)
                    for kernel, bandwidth in zip(self.kernels, self.bandwidths)]
        print(k_b_pair)
        return k_b_pair

    def check_requires_grad(self):
        # print("Method 1: Check each parameter's requires_grad attribute")
        for name, param in self.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}, shape={param.shape}")

    def test_bootstrap(self, X_test, Y_test, n_te, L_tr, F_tr, n_per):
        U_te, U_b = self.compute_U_stats(
            X_test=X_test, Y_test=Y_test, n_per=n_per)
        if self.is_cov_tr:
            L_te = L_tr  # L_te.size() = (c, c) column vector
        else:
            L_te = torch.inverse(
                sqrtm(self.compute_Sigma(X_test=X_test, Y_test=Y_test)))

        half_te = L_te @ U_te
        half_te_b = L_te @ U_b
        T_obs_no_select = n_te**2 * torch.norm(half_te, p=2)**2
        T_b_no_select = n_te**2 * torch.norm(half_te_b, p=2, dim=0)**2
        p_value_no_select = torch.sum(T_b_no_select > T_obs_no_select) / n_per

        F_te = torch.sign(half_te)
        F_te_b = torch.sign(half_te_b)  # [:, 0:2]
        F = F_te == F_tr
        F_b = F_te_b == F_tr
        T_obs_select = n_te**2 * torch.norm(F * half_te, p=2)**2
        T_b_select = n_te**2 * torch.norm(F_b * half_te_b, p=2, dim=0)**2
        p_value_select = torch.sum(T_b_select > T_obs_select) / n_per

        return p_value_select, p_value_no_select


def TST_MMD_AU(name, N1, rs, check, n_test, n_per, alpha, N_epoch, batch_size, learning_rate, n_bandwidth, reg=1e-5, way=['Agg', 'Agg', 'Agg', 'Agg'], is_cov=True, is_cov_tr=True, encoder=None):
    """
    Two-sample testing method based on the Aggregated kernel tests with non-consistent U-statistics.

    Parameters
    ----------
    n_bandwidth : (int, int, int, int)
        The number of bandwidths to be used for the (gaussian, laplacian, deep_gaussian, deep_laplacian) kernels.
    reg: a small constant
        The parameter of a scaled identity matrix used in the computation of Sigma matrix.
    way: The way to select the initial bandwidths of (gaussian, laplacian, deep_gaussian, deep_laplacian) kernels.
    is_cov : bool
        Whether to use the covariance matrix.
    is_cov_tr : bool
        Whether to use the covariance matrix computed from the training data for the test data.
    """
    device = check_device()
    np.random.seed(rs)
    torch.manual_seed(rs)
    X_train, Y_train = load_data(name, N1, rs, check)
    X_train, Y_train = MatConvert(X_train, device, torch.float32), MatConvert(
        Y_train, device, torch.float32)

    k_b_pair = generate_kernel_bandwidth(
        n_bandwidth, X_train, Y_train, reg, is_cov, way, device)

    model_au = MMD_AU(X_train, Y_train, k_b_pair, reg, is_cov,
                      is_cov_tr, encoder).to(device, torch.float32)
    start_time = time.time()
    # model_au.train_kernel(N_epoch, batch_size, learning_rate)
    train_time = time.time() - start_time
    
    print(model_au.compute_Sigma())
    model_au.get_kernel_bandwidth_pairs()

    model_au.eval()
    with torch.no_grad():
        U_tr = model_au.compute_U_stats()
        L_tr = torch.inverse(sqrtm(model_au.compute_Sigma()))
        F_tr = torch.sign(L_tr @ U_tr)

    n_te = len(X_train)
    H_MMD_AU_select = np.zeros(n_test)
    H_MMD_AU_no_select = np.zeros(n_test)
    P_MMD_AU_select = np.zeros(n_test)
    P_MMD_AU_no_select = np.zeros(n_test)
    N_test_all = 10 * N1
    X_test_all, Y_test_all = load_data(name, N_test_all, rs + 283, check)
    test_time = 0
    for k in range(n_test):
        ind_test = np.random.choice(N_test_all, N1, replace=False)
        X_test = X_test_all[ind_test]
        Y_test = Y_test_all[ind_test]
        X_test, Y_test = MatConvert(X_test, device, torch.float32), MatConvert(
            Y_test, device, torch.float32)

        start_time = time.time()
        with torch.no_grad():
            p_value_select, p_value_no_select = model_au.test_bootstrap(
                X_test, Y_test, n_te, L_tr, F_tr, n_per)
        test_time += time.time() - start_time

        P_MMD_AU_select[k] = p_value_select
        P_MMD_AU_no_select[k] = p_value_no_select
        H_MMD_AU_select[k] = p_value_select < alpha
        H_MMD_AU_no_select[k] = p_value_no_select < alpha

    return H_MMD_AU_select, H_MMD_AU_no_select, P_MMD_AU_select, P_MMD_AU_no_select, train_time, test_time

# diversity calculation see "diversity.py"
result = []
for N in [50, 100, 150, 200, 250, 300, 350]:
    H_MMD_AU_select, H_MMD_AU_no_select, P_MMD_AU_select, P_MMD_AU_no_select, _, _ = TST_MMD_AU(
        "BLOB", N, 15, 1, 100, 100, 0.05, 0, 128, 0.001, n_bandwidth=(5, 0, 0, 0), is_cov=True)
    # print(np.sum(H_MMD_AU_select))
    # print(np.sum(H_MMD_AU_no_select))
    result.append(np.sum(H_MMD_AU_select)/100)

print(result)