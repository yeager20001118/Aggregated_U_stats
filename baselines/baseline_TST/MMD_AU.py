from torchvision.models import resnet18
from torch.autograd import Function
import scipy
import sys
import os
sys.path.append(os.path.abspath('..'))
from dataloader_TST import load_data
import torch.nn as nn
import time
import numpy as np
import torch
import numpy as np

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


KERNEL_SET = ["gaussian", "laplacian", "mahalanobis", "deep_gaussian", "deep_laplacian", "deep_mahalanobis"]
def generate_kernel_bandwidth(n_bandwidth, X, Y, reg, is_cov, way, encoder):
    assert set(way) <= {'Agg', 'Boost', 'Grid', 'Fuse'}
    k_b_pair = []
    Z = torch.cat([X, Y], dim=0)
    dist = torch.cdist(Z, Z, p=2)
    for i in range(len(n_bandwidth)):
        if i >= 3 and n_bandwidth[i] > 0:
            if encoder is None:
                raise ValueError("Encoder must be provided for deep kernels.")
            Z_rep = encoder(Z)
            dist = torch.cdist(Z_rep, Z_rep, p=2)
        kernel = KERNEL_SET[i]
        if way[i] == 'Boost':
            bandwidths = get_bandwidth_boost(
                n_bandwidth[i], torch.median(dist).item())
            if "mahalanobis" in kernel.lower():
                if "deep" in kernel.lower():
                    for b in bandwidths:
                        k_b_pair.append((kernel, (b, torch.eye(Z_rep.shape[1], device=Z_rep.device))))
                else:
                    for b in bandwidths:
                        k_b_pair.append((kernel, (b, torch.eye(Z.shape[1], device=Z.device))))
            else:
                for b in bandwidths:
                    k_b_pair.append((kernel, b))
        elif way[i] == 'Agg':
            m = dist.shape[0]
            indices = torch.triu_indices(m, m, offset=0)
            dist_v = dist[indices[0], indices[1]]
            bandwidths = get_bandwidth_agg(dist_v, n_bandwidth[i])
            if "mahalanobis" in kernel.lower():
                if "deep" in kernel.lower():
                    for b in bandwidths:
                        k_b_pair.append((kernel, (b, torch.eye(Z_rep.shape[1], device=Z_rep.device))))
                else:
                    for b in bandwidths:
                        k_b_pair.append((kernel, (b, torch.eye(Z.shape[1], device=Z.device))))
            else:
                for b in bandwidths:
                    k_b_pair.append((kernel, b))
        elif way[i] == 'Grid':
            k_b_pair = get_bandwidth_grid(
                X, Y, n_bandwidth, k_b_pair, kernel, n_bandwidth[i], reg, is_cov, X.device, encoder)
        elif way[i] == "Fuse":
            median = torch.median(dist)
            dist = dist + (dist == 0) * median
            dd = torch.sort(dist)[0].view(-1)
            n = len(dd)
            idx_5 = int(torch.floor(torch.tensor(n * 0.05)).item())
            idx_95 = int(torch.floor(torch.tensor(n * 0.95)).item())
            lambda_min = dd[idx_5] / 2
            lambda_max = dd[idx_95] * 2
            bandwidths = torch.linspace(
                lambda_min, lambda_max, n_bandwidth[i]).to(dist.device)
            if "mahalanobis" in kernel.lower():
                if "deep" in kernel.lower():
                    for b in bandwidths:
                        k_b_pair.append((kernel, (b, torch.eye(Z_rep.shape[1], device=Z_rep.device))))
                else:
                    for b in bandwidths:
                        k_b_pair.append((kernel, (b, torch.eye(Z.shape[1], device=Z.device))))
            else:
                for b in bandwidths:
                    k_b_pair.append((kernel, b))

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
    power = (lambda_max / lambda_min) ** (1 / max(1,(n_bandwidth - 1)))
    # Generate geometric sequence of bandwidths
    bandwidths = torch.pow(power, torch.arange(
        n_bandwidth, device=device)) * lambda_min

    return bandwidths


def get_bandwidth_grid(X, Y, n_bandwidth, k_b_pair, kernel, number, reg, is_cov, device, encoder):
    if 'deep' in kernel.lower():
        X_rep = encoder(X)
        Y_rep = encoder(Y)
        Dxy = torch.cdist(X_rep, Y_rep, p=2)
    else:
        Dxy = torch.cdist(X, Y, p=2)
    Dxy[Dxy<0]=0
    list_bandwidths = Dxy.median() * (2.0 ** torch.linspace(0, 100, 500).to(device, torch.float32))
    list_bandwidths = list_bandwidths.sort()[0].reshape(-1,1)
    bandwidths = torch.zeros(number, device=device, dtype=torch.float32)

    for k in range(number):
        MAXMUM = 0
        for i in range(len(list_bandwidths)):            
            if 'mahalanobis' in kernel.lower():
                if "deep" in kernel.lower():
                    model_au = MMD_AU(X, Y, n_bandwidth, k_b_pair+[(kernel, (list_bandwidths[i], torch.eye(X_rep.shape[1], device=X_rep.device)))], reg, is_cov, encoder=encoder).to(device, torch.float32)
                else:
                    model_au = MMD_AU(X, Y, n_bandwidth, k_b_pair+[(kernel, (list_bandwidths[i], torch.eye(X.shape[1], device=X.device)))], reg, is_cov, encoder=encoder).to(device, torch.float32)
            else:
                model_au = MMD_AU(X, Y, n_bandwidth, k_b_pair+[(kernel,list_bandwidths[i])], reg, is_cov, encoder=encoder).to(device, torch.float32)
            U = model_au.compute_U_stats()
            Sigma = model_au.compute_Sigma()
            try:
                T = len(X)**2 * U.T @ torch.inverse(Sigma) @ U
                if T.item() > MAXMUM and list_bandwidths[i] not in bandwidths:
                    MAXMUM = T.item()
                    bandwidths[k] = list_bandwidths[i]
            except:
                pass
        if 'mahalanobis' in kernel.lower():
            if "deep" in kernel.lower():
                k_b_pair.append((kernel, (bandwidths[k], torch.eye(X_rep.shape[1], device=X_rep.device))))
            else:
                k_b_pair.append((kernel, (bandwidths[k], torch.eye(X.shape[1], device=X.device))))
        else:
            k_b_pair.append((kernel,bandwidths[k]))
    return k_b_pair

def get_bandwidth_boost(n_bandwidth, median):
    scales = get_scales(n_bandwidth)
    return scales * median


def get_scales(n_bandwidth, base=2):
    ratio = torch.sqrt(torch.Tensor([base]))
    if n_bandwidth % 2 == 1:  # odd n
        half = max(1,(n_bandwidth - 1)) // 2
        powers = torch.tensor(list(range(-half, half + 1)), dtype=torch.float)
    else:  # even n
        half = n_bandwidth // 2
        powers = torch.tensor(
            [-half + i + 0.5 for i in range(n_bandwidth)], dtype=torch.float)
    scales = torch.pow(ratio, powers)
    return scales

def Pdist2(x, y, M_matrix):
    """compute the paired distance between x and y."""
    n = x.shape[0]
    d = x.shape[1]
    assert x.shape[0]==y.shape[0]
    
    x_Mat = torch.matmul(x, M_matrix)
    y_Mat = torch.matmul(y, M_matrix)    
    
    x_Mat_x = torch.sum(torch.mul(x_Mat, x), 1).view(-1, 1)
    y_Mat_y = torch.sum(torch.mul(y_Mat, y), 1).view(1, -1)
    x_Mat_y = torch.sum(torch.mul(x_Mat.unsqueeze(1).expand(n,n,d), y), 2)
    y_Mat_x = torch.sum(torch.mul(y_Mat.unsqueeze(1).expand(n,n,d), x), 2).t()
    
    Pdist = x_Mat_x + y_Mat_y - x_Mat_y - y_Mat_x
    Pdist[Pdist<0]=0
    return Pdist

def kernel_matrix(kernel, bandwidth, dist=None):
    # Compute kernel matrix based on the specified kernel
    if "gaussian" in kernel.lower():
        # K_GAUSS(x,y) = exp(-||x-y||²/σ²)
        K = torch.exp(-torch.pow(dist, 2) / torch.pow(bandwidth, 2))
    elif "laplacian" in kernel.lower():
        # K_LAP(x,y) = exp(-||x-y||/σ)
        K = torch.exp(-dist / bandwidth)
    elif "mahalanobis" in kernel.lower():
        K = torch.exp(-dist / torch.pow(bandwidth, 2))
    else:
        raise ValueError(
            f"Unknown kernel: {kernel}. Use 'gaussian', 'laplacian' or 'mahalanobis'.")
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
    def __init__(self, X, Y, n_bandwidth, k_b_pair, reg=1e-5, is_cov=True, is_cov_tr=True, encoder=None):
        super(MMD_AU, self).__init__()
        self.X = X
        self.Y = Y
        self.k_b_pair = k_b_pair
        self.kernels = [pair[0] for pair in k_b_pair]
        self.bandwidths = nn.ParameterList()
        self.n_bandwidth = n_bandwidth
        self.M_matrix = nn.ParameterList()
        self.M_matrix_rep = nn.ParameterList()
        self.encoder = encoder
        self.is_cov = is_cov
        self.is_cov_tr = is_cov_tr
        self.reg = reg
        self.U_tr = None
        self.L_tr = None
        self.F_tr = None
        self.train_time = None
        
        for kernel, bandwidth in k_b_pair:
            if "mahalanobis" in kernel.lower():
                if "deep" in kernel.lower():                    
                    self.bandwidths.append(nn.Parameter(bandwidth[0].clone()))
                    self.M_matrix_rep.append(nn.Parameter(bandwidth[1].clone()))
                else:
                    self.bandwidths.append(nn.Parameter(bandwidth[0].clone()))
                    self.M_matrix.append(nn.Parameter(bandwidth[1].clone()))
            else:
                self.bandwidths.append(nn.Parameter(bandwidth.clone()))

    def forward():
        pass

    def compute_U_stats(self, X_test=None, Y_test=None, n_per=None):
        device = check_device()
        if X_test is not None and Y_test is not None:
            Z = torch.cat([X_test, Y_test], dim=0)
            n = len(X_test)
        else:
            Z = torch.cat([self.X, self.Y], dim=0)
            n = len(self.X)
        dist = torch.cdist(Z, Z, p=2)
        try:
            M_dist = torch.stack([Pdist2(Z, Z, M) for M in self.M_matrix], dim=0)
        except:
            pass
        try:
            rep_Z = self.encoder(Z)
            rep_dist = torch.cdist(rep_Z, rep_Z, p=2) if self.encoder is not None else None
        except:
            pass
        try:
            M_dist_rep = torch.stack([Pdist2(rep_Z, rep_Z, M) for M in self.M_matrix_rep], dim=0)
        except:
            pass
        n_b = len(self.bandwidths)
        
        U = []  # U.size() = (c, 1) column vector
        U_b = []  # U_b.size() = (c, n_per)

        if n_per is not None:
            B = torch.randint(0, 2, (n_per, n), dtype=torch.float).to(
                device) * 2.0 - 1.0
            B = torch.einsum('bi,bj->bij', B, B)
        for i in range(n_b):
            # kernel_matrix(kernel, bandwidth, dist=None):
            if 'mahalanobis' == self.kernels[i].lower():
                K = kernel_matrix(self.kernels[i], self.bandwidths[i], dist=M_dist[i-sum(self.n_bandwidth[:2])])
            elif 'deep_mahalanobis' == self.kernels[i].lower():
                K = kernel_matrix(self.kernels[i], self.bandwidths[i], dist=M_dist_rep[i-sum(self.n_bandwidth[:5])])
            elif 'deep' in self.kernels[i].lower():
                K = kernel_matrix(self.kernels[i], self.bandwidths[i], dist=rep_dist)
            else:
                K = kernel_matrix(self.kernels[i], self.bandwidths[i], dist=dist)
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
        device = check_device()
        n_b = len(self.bandwidths)    
        if not self.is_cov:
            return torch.eye(n_b).to(device, torch.float32)
        
        if X_test is not None and Y_test is not None:
            Z = torch.cat([X_test, Y_test], dim=0)
            n = len(X_test)
        else:
            Z = torch.cat([self.X, self.Y], dim=0)
            n = len(self.X)
        
        indices = torch.randperm(Z.size(0), device=Z.device)
        Z_null = Z[indices]
        
        dist = torch.cdist(Z_null, Z_null, p=2)
        try:
            M_dist = torch.stack([Pdist2(Z_null, Z_null, M) for M in self.M_matrix], dim=0)
        except:
            pass
        try:
            rep_Z_null = self.encoder(Z_null)
            rep_dist = torch.cdist(rep_Z_null, rep_Z_null, p=2) if self.encoder is not None else None
        except:
            pass
        try:
            M_dist_rep = torch.stack([Pdist2(rep_Z_null, rep_Z_null, M) for M in self.M_matrix_rep], dim=0)
        except:
            pass
        
        Sigma = torch.zeros(n_b, n_b).to(device, torch.float32)
        C = self.get_C(2, Z_null.size(0))
        for i in range(n_b):
            for j in range(i, n_b):
                if 'mahalanobis' == self.kernels[i].lower():
                    K1 = kernel_matrix(self.kernels[i], self.bandwidths[i], dist=M_dist[i-sum(self.n_bandwidth[:2])])
                elif 'deep_mahalanobis' == self.kernels[i].lower():
                    K1 = kernel_matrix(self.kernels[i], self.bandwidths[i], dist=M_dist_rep[i-sum(self.n_bandwidth[:5])])
                elif 'deep' in self.kernels[i].lower():
                    K1 = kernel_matrix(self.kernels[i], self.bandwidths[i], dist=rep_dist)
                else:
                    K1 = kernel_matrix(self.kernels[i], self.bandwidths[i], dist=dist)
                _, h_matrix1 = mmd_u(K1, n)

                if 'mahalanobis' == self.kernels[j].lower():
                    K2 = kernel_matrix(self.kernels[j], self.bandwidths[j], dist=M_dist[j-sum(self.n_bandwidth[:2])])
                elif 'deep_mahalanobis' == self.kernels[j].lower():
                    K2 = kernel_matrix(self.kernels[j], self.bandwidths[j], dist=M_dist_rep[j-sum(self.n_bandwidth[:5])])
                elif 'deep' in self.kernels[j].lower():
                    K2 = kernel_matrix(self.kernels[j], self.bandwidths[j], dist=rep_dist)
                else:
                    K2 = kernel_matrix(self.kernels[j], self.bandwidths[j], dist=dist)
                _, h_matrix2 = mmd_u(K2, n)

                mask = torch.triu(torch.ones(n, n), diagonal=1).to(device)

                Sigma[i, j] = C * (h_matrix1 * h_matrix2 * mask).sum()
                Sigma[j, i] = Sigma[i, j]
        return Sigma + self.reg * torch.eye(n_b).to(device, torch.float32)

    def train_kernel(self, N_epoch, batch_size, learning_rate):
        print(self.parameters)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        n = self.X.size(0)
        batch_size = min(n, batch_size)
        batches=n//batch_size
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
                U = self.compute_U_stats(X_test=X[idx*batch_size:idx*batch_size+batch_size], Y_test=Y[idx*batch_size:idx*batch_size+batch_size])
                Sigma = self.compute_Sigma(X_test=X[idx*batch_size:idx*batch_size+batch_size], Y_test=Y[idx*batch_size:idx*batch_size+batch_size])
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
            if (epoch + 1) % 100 == 0:
                with torch.no_grad():
                    for i in range(len(self.M_matrix)):
                        eigvalues, eigvectors = torch.linalg.eig(self.M_matrix[i])
                        eigvalues = torch.max(eigvalues.real, torch.tensor(1e-5).to(check_device(), torch.float))
                        eigvectors = eigvectors.real
                        eigvectors = eigvectors.t().reshape(eigvectors.shape[0], -1, eigvectors.shape[1])
                        self.M_matrix[i] = eigvalues[0] * eigvectors[0].t() * eigvectors[0]
                        for j in range(1, len(eigvalues)):
                            self.M_matrix[i] += eigvalues[j] * eigvectors[j] * eigvectors[j].t()
                    for i in range(len(self.M_matrix_rep)):
                        eigvalues, eigvectors = torch.linalg.eig(self.M_matrix_rep[i])
                        eigvalues = torch.max(eigvalues.real, torch.tensor(1e-5).to(check_device(), torch.float))
                        eigvectors = eigvectors.real
                        eigvectors = eigvectors.t().reshape(eigvectors.shape[0], -1, eigvectors.shape[1])
                        self.M_matrix_rep[i] = eigvalues[0] * eigvectors[0].t() * eigvectors[0]
                        for j in range(1, len(eigvalues)):
                            self.M_matrix_rep[i] += eigvalues[j] * eigvectors[j] * eigvectors[j].t()
        
        
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

    def test_bootstrap(self, X_test, Y_test, n_te, n_per):
        U_te, U_b = self.compute_U_stats(
            X_test=X_test, Y_test=Y_test, n_per=n_per)
        if self.is_cov_tr:
            # L_tr = self.compute_Sigma()
            L_te = self.L_tr  # L_te.size() = (c, c) column vector
        else:
            L_te = torch.inverse(
                sqrtm(self.compute_Sigma(X_test=X_test, Y_test=Y_test)))
        
        half_te = L_te @ U_te
        half_te_b = L_te @ U_b
        T_obs_no_select = n_te**2 * torch.norm(half_te, p=2)**2
        T_b_no_select = n_te**2 * torch.norm(half_te_b, p=2, dim=0)**2
        p_value_no_select = torch.sum(T_b_no_select > T_obs_no_select) / n_per
        
        F_te = torch.sign(half_te)
        F_te_b = torch.sign(half_te_b) #[:, 0:2]
        F = F_te == self.F_tr
        F_b = F_te_b == self.F_tr
        T_obs_select = n_te**2 * torch.norm(F * half_te, p=2)**2
        T_b_select = n_te**2 * torch.norm(F_b * half_te_b, p=2, dim=0)**2
        p_value_select = torch.sum(T_b_select > T_obs_select) / n_per

        return p_value_select, p_value_no_select

def train_MMD_AU(name, N1, rs, check, N_epoch, batch_size, learning_rate, n_bandwidth, reg=1e-5, way=['Agg', 'Agg', 'Agg', 'Agg', 'Agg', 'Agg'], is_cov=True, encoder=None):
    device = check_device()
    np.random.seed(rs)
    torch.manual_seed(rs)
    X_train, Y_train = load_data(name, N1, rs, check)
    X_train, Y_train = MatConvert(X_train, device, torch.float32), MatConvert(Y_train, device, torch.float32)
    with torch.no_grad():
        k_b_pair = generate_kernel_bandwidth(
            n_bandwidth, X_train, Y_train, reg, is_cov, way, encoder)
    model_au = MMD_AU(X_train, Y_train, n_bandwidth, k_b_pair, reg, is_cov, True, encoder).to(device, torch.float32)
    start_time = time.time()
    model_au.train_kernel(N_epoch, batch_size, learning_rate)
    model_au.train_time = time.time() - start_time
    with torch.no_grad():
        model_au.U_tr = model_au.compute_U_stats()
        model_au.L_tr = torch.inverse(sqrtm(model_au.compute_Sigma()))
        model_au.F_tr = torch.sign(model_au.L_tr @ model_au.U_tr)
        model_au.X = None
        model_au.Y = None
        return model_au

def TST_MMD_AU(name, N1, rs, check, n_test, n_per, alpha, is_cov_tr, model_au):
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
    
    model_au.eval()
    model_au.is_cov_tr = is_cov_tr
    with torch.no_grad():
        H_MMD_AU_select = np.zeros(n_test)
        H_MMD_AU_no_select = np.zeros(n_test)
        P_MMD_AU_select = np.zeros(n_test)
        P_MMD_AU_no_select = np.zeros(n_test)
        N_test_all = 10 * N1
        X_test_all, Y_test_all = load_data(name, N_test_all, rs+283, check)
        test_time = 0
        for k in range(n_test):
            ind_test = np.random.choice(N_test_all, N1, replace=False)
            X_test = X_test_all[ind_test]
            Y_test = Y_test_all[ind_test]
            X_test, Y_test = MatConvert(X_test, device, torch.float32), MatConvert(Y_test, device, torch.float32)
            start_time = time.time()
            p_value_select, p_value_no_select = model_au.test_bootstrap(X_test, Y_test, N1, n_per)
            test_time += time.time() - start_time
            
            P_MMD_AU_select[k] = p_value_select
            P_MMD_AU_no_select[k] = p_value_no_select
            H_MMD_AU_select[k] = p_value_select < alpha
            H_MMD_AU_no_select[k] = p_value_no_select < alpha

        return H_MMD_AU_select, H_MMD_AU_no_select, P_MMD_AU_select, P_MMD_AU_no_select, model_au.train_time, test_time


# from torchvision.models import resnet18
# import torch.nn as nn
# import pickle
# np.random.seed(1102)
# torch.manual_seed(1102)
# torch.cuda.manual_seed_all(1102)  # 如果有GPU
# class TorchVisionEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.backbone = resnet18(pretrained = True)
#         # Remove the final fully connected layer.
#         self.backbone.fc = nn.Identity() 
#     def forward(self, x):
#         x = x.resize(len(x), 1, 32, 32)
#         x = x.repeat(1, 3, 1, 1)
#         return self.backbone(x)

# SEED = 683
# name = "MNIST"
# N1 = 1000
# check = 0

# # model_au = train_MMD_AU(name, N1, SEED, check, 0, 128, 0.0005, (1, 1, 1, 1, 1, 1), reg=1e-2, way=['Fuse', 'Fuse', 'Fuse', 'Fuse', 'Fuse', 'Fuse'], is_cov=True, encoder=None)
# # with open("/data/gpfs/projects/punim2335/AU_exp/exp_TST/model/MNIST_[1, 1, 1, 1, 1, 1]_True_20_189_1000_128_5e-05.pkl", "rb") as f:
# with open("/data/gpfs/projects/punim2335/AU_exp/exp_TST/model/MNIST_[1, 1, 1, 1, 1, 1]_True_70_683_1000_128_5e-05.pkl", "rb") as f:
#             model_au = pickle.load(f)

# result_s = []
# result_no_s = []

# for seed in range(SEED, SEED+10):
#     H_MMD_AU_select, H_MMD_AU_no_select, P_MMD_AU_select, P_MMD_AU_no_select, _, _ = TST_MMD_AU(
#         name, N1, seed, check, 100, 100, 0.05, False, model_au)
#     result_s.append(np.sum(H_MMD_AU_select))
#     result_no_s.append(np.sum(H_MMD_AU_no_select))
#     # break

# print(result_s, result_no_s)
# print("{:.3f}±{:.3f}".format(np.mean(result_s)/100, np.std(result_s)/np.sqrt(10)/100))
# print("{:.3f}±{:.3f}".format(np.mean(result_no_s)/100, np.std(result_no_s)/np.sqrt(10)/100))