from torch.autograd import Function
import scipy
import sys
import os
sys.path.append(os.path.abspath('..'))
from dataloader_IDT import load_data
import torch.nn as nn
import time
import numpy as np
import torch
import math
import tqdm
import torchvision

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
    assert set(way) <= {'Agg', 'Fuse'}
    k_b_pair = []
    max_sample = 500
    distXX = torch_distance(X, X, max_size=max_sample, matrix=False)
    distYY = torch_distance(Y, Y, max_size=max_sample, matrix=False)
    for i in range(len(n_bandwidth)):
        if i >= 3 and n_bandwidth[i] > 0:
            if encoder is None:
                raise ValueError("Encoder must be provided for deep kernels.")
            encoder = encoder.to(check_device())
            X_rep = encoder(X)
            if Y.size(1) > 1:
                Y_rep = encoder(Y)
            else:
                Y_rep = Y
            distXX = torch_distance(X_rep, X_rep, max_size=max_sample, matrix=False)
            distYY = torch_distance(Y_rep, Y_rep, max_size=max_sample, matrix=False)
        kernel = KERNEL_SET[i]
        if way[i] == 'Agg':
            bandwidths_XX = get_bandwidth_agg(distXX, n_bandwidth[i])
            bandwidths_YY = get_bandwidth_agg(distYY, n_bandwidth[i])
            bandwidths = zip(bandwidths_XX, bandwidths_YY)
        if way[i] == 'Fuse':
            bandwidths_XX = get_bandwidth_fuse(distXX, n_bandwidth[i])
            bandwidths_YY = get_bandwidth_fuse(distYY, n_bandwidth[i])
            bandwidths = zip(bandwidths_XX, bandwidths_YY)
        # bandwidths = [(x, y) for x in bandwidths_XX for y in bandwidths_YY]

        if "mahalanobis" in kernel.lower():
            if "deep" in kernel.lower():
                for (b_x, b_y) in bandwidths:
                    k_b_pair.append((kernel, ((b_x, torch.eye(X_rep.shape[1], device=X_rep.device)), (b_y, torch.eye(Y_rep.shape[1], device=Y_rep.device)))))
            else:
                for (b_x, b_y) in bandwidths:
                    k_b_pair.append((kernel, ((b_x, torch.eye(X.shape[1], device=X.device)), (b_y, torch.eye(Y.shape[1], device=Y.device)))))
        else:    
            for b in bandwidths:
                k_b_pair.append((kernel, b))
    return k_b_pair

def torch_distance(X, Y, norm=2, max_size=None, matrix=True, is_squared=False):
    if X.dim() == 4:
        X = X.view(X.size(0), -1)
    if Y.dim() == 4:
        Y = Y.view(Y.size(0), -1)

    if X.dim() == 1:
        X = X.view(-1, 1)
    if Y.dim() == 1:
        Y = Y.view(-1, 1)
    diff = X[None, :, :] - Y[:, None, :]
    if norm == 2:
        if is_squared:
            dist = torch.sum(diff**2, dim=-1)
        else:
            dist = torch.sqrt(torch.sum(diff**2, dim=-1))
    elif norm == 1:
        dist = torch.sum(torch.abs(diff), dim=-1)
    else:
        raise ValueError("Norm must be L1 or L2")
    if max_size:
        dist = dist[:max_size, :max_size]
    if matrix:
        return dist
    else:
        m = dist.shape[0]
        indices = torch.triu_indices(m, m, offset=0)
        return dist[indices[0], indices[1]]

def get_bandwidth_fuse(dist, n_bandwidth):
    median = torch.median(dist)
    dist = dist + (dist == 0) * median
    dd = torch.sort(dist)[0].view(-1)
    n = len(dd)
    idx_5 = int(torch.floor(torch.tensor(n * 0.05)).item())
    idx_95 = int(torch.floor(torch.tensor(n * 0.95)).item())
    lambda_min = dd[idx_5] / 2
    lambda_max = dd[idx_95] * 2
    bandwidths = torch.linspace(
        lambda_min, lambda_max, n_bandwidth).to(dist.device)
    return bandwidths

def get_bandwidth_agg(dist, n_bandwidth):
    device = dist.device
    median = torch.median(dist).to(device)
    # Compute power sequence
    bandwidths = [2 ** i * median for i in get_power_range(n_bandwidth)]
    return bandwidths

def get_power_range(n):
    if n % 2 == 0:
        # For even n, generate n numbers with 0.5 offset
        start = -(n-1)/2
        return [start + i for i in range(n)]
    else: 
        # For odd n, generate n numbers centered at 0
        start = -(n//2)
        return [start + i for i in range(n)]


# def kernel_matrix(kernel, bandwidth, X, Y=None, dist=None, model=None):
#     # If Y is not provided, use X
#     if Y is None:
#         Y = X
#     # Compute pairwise distances
#     if dist is None:
#         dist = torch.cdist(X, Y, p=2)  # Euclidean distance (L2 norm)
#     # Compute kernel matrix based on the specified kernel
#     if kernel.lower() == "gaussian":
#         # K_GAUSS(x,y) = exp(-||x-y||²/σ²)
#         K = torch.exp(-torch.pow(dist, 2) / torch.pow(bandwidth, 2))
#     elif kernel.lower() == "laplacian":
#         # K_LAP(x,y) = exp(-||x-y||/σ)
#         K = torch.exp(-dist / bandwidth)
#     elif kernel.lower()[:4] == "deep":
#         if model is None:
#             raise ValueError("Model must be provided for deep kernel.")
#         X_rep = model(X)
#         Y_rep = model(Y)
#         dist = torch.cdist(X_rep, Y_rep, p=2)
#         K = kernel_matrix(kernel[5:], bandwidth, X_rep, dist=dist)
#     else:
#         raise ValueError(
#             f"Unknown kernel: {kernel}. Use 'gaussian' or 'laplacian'.")
#     return K

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

def get_h_matrix(K, n):
    n_idx = np.floor(n/2).astype(int)
    # h_matrix = torch.zeros([n_idx, n_idx]).to(K.device)
    # for i in range(n_idx-1):
    #     for j in range(i+1, n_idx):
    #             h_matrix[i][j] = K[i][j] + K[i+n_idx][j+n_idx] - K[i][j+n_idx] - K[j][i+n_idx] 
    
    i_indices = torch.arange(n_idx, device=K.device)
    j_indices = torch.arange(n_idx, device=K.device)
    i_grid, j_grid = torch.meshgrid(i_indices, j_indices, indexing='ij')
    i_shifted, j_shifted = i_grid + n_idx, j_grid + n_idx
    
    term1 = K[i_grid, j_grid]
    term2 = K[i_shifted, j_shifted]
    term3 = K[i_grid, j_shifted]
    term4 = K[j_grid, i_shifted]
    
    h_matrix = term1 + term2 - term3 - term4
    mask = torch.triu(torch.ones(n_idx, n_idx, device=K.device), diagonal=1).bool()
    h_matrix = h_matrix * mask

    # K_ij = K.view(n, n, 1, 1)
    # K_kl = K.view(1, 1, n, n)
    # K_il = K.view(n, 1, 1, n)
    # K_kj = K.view(1, n, n, 1)
    
    # # Compute H using broadcasting
    # h_matrix = K_ij + K_kl - K_il - K_kj
    
    # # Create mask to exclude cases where any indices are equal
    # # First create index tensors
    # i_indices = torch.arange(n).view(n, 1, 1, 1).expand(n, n, n, n).to(K.device)
    # j_indices = torch.arange(n).view(1, n, 1, 1).expand(n, n, n, n).to(K.device)
    # k_indices = torch.arange(n).view(1, 1, n, 1).expand(n, n, n, n).to(K.device)
    # l_indices = torch.arange(n).view(1, 1, 1, n).expand(n, n, n, n).to(K.device)
    
    # # Create mask where all indices are different
    # mask = (i_indices < j_indices) & (j_indices < k_indices) & (k_indices < l_indices)
    
    # # Apply mask to H
    # h_matrix = h_matrix * mask.float()
    
    return h_matrix

def hsic_so(kernel, bandwidths, distXY):
    b_X, b_Y = bandwidths
    distXX, distYY = distXY
    n = distXX.size(0)
    n_idx = np.floor(n/2).astype(int)
    K_X = kernel_matrix(kernel, b_X, distXX)
    K_Y = kernel_matrix(kernel, b_Y, distYY)
    h_X = get_h_matrix(K_X, n) # h.size() = (n_idx, n_idx)
    h_Y = get_h_matrix(K_Y, n)
    hsic = torch.sum(h_X*h_Y) / math.comb(n_idx, 2)
    return hsic, (h_X, h_Y)

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
        self.n_bandwidth = n_bandwidth
        self.bandwidths = nn.ParameterList()
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

        for kernel, (bandwidthX, bandwidthY) in k_b_pair:
            if "mahalanobis" in kernel.lower():
                b_x, b_y = nn.Parameter(bandwidthX[0].clone()), nn.Parameter(bandwidthY[0].clone())
                # self.register_parameter(f'bandwidth_x_{len(self.bandwidths)}', b_x)
                # self.register_parameter(f'bandwidth_y_{len(self.bandwidths)}', b_y)
                # self.bandwidths.append((b_x, b_y))
                m_X, m_Y = nn.Parameter(bandwidthX[1].clone()), nn.Parameter(bandwidthY[1].clone())
                if "deep" in kernel.lower():
                    self.register_parameter(f'matrix_x_{len(self.M_matrix_rep)}', m_X)
                    self.register_parameter(f'matrix_y_{len(self.M_matrix_rep)}', m_Y)
                    self.M_matrix_rep.append([m_X, m_Y])
                else:
                    self.register_parameter(f'matrix_x_{len(self.M_matrix)}', m_X)
                    self.register_parameter(f'matrix_y_{len(self.M_matrix)}', m_Y)
                    self.M_matrix.append([m_X, m_Y])
            else:    
                b_x, b_y = nn.Parameter(bandwidthX.clone()), nn.Parameter(bandwidthY.clone())

            self.register_parameter(f'bandwidth_x_{len(self.bandwidths)}', b_x)
            self.register_parameter(f'bandwidth_y_{len(self.bandwidths)}', b_y)
            self.bandwidths.append((b_x, b_y))

    def forward():
        pass

    def compute_U_stats(self, X_test=None, Y_test=None, n_per=None):
        device = check_device()
        if X_test is not None and Y_test is not None:
            X = X_test
            Y = Y_test
        else:
            X = self.X
            Y = self.Y

        distXX = torch.cdist(X, X, p=2) 
        distYY = torch.cdist(Y, Y, p=2)
        n = len(X)
        n_b = len(self.bandwidths)
        n_idx = np.floor(n/2).astype(int)

        try:
            M_distXX = torch.stack([Pdist2(X, X, M[0]) for M in self.M_matrix], dim=0)
            M_distYY = torch.stack([Pdist2(Y, Y, M[1]) for M in self.M_matrix], dim=0)
        except:
            pass
            
        try:
            X_rep = self.encoder(X)
            rep_distXX = torch.cdist(X_rep, X_rep, p=2)
        except:
            pass

        try:
            Y_rep = self.encoder(Y)
            rep_distYY = torch.cdist(Y_rep, Y_rep, p=2)
        except:
            Y_rep = Y
            rep_distYY = distYY

        try:
            M_rep_distXX = torch.stack([Pdist2(X_rep, X_rep, M[0]) for M in self.M_matrix_rep], dim=0)
            M_rep_distYY = torch.stack([Pdist2(Y_rep, Y_rep, M[1]) for M in self.M_matrix_rep], dim=0)
        except:
            pass
        
        U = []  # U.size() = (c, 1) column vector
        U_b = []  # U_b.size() = (c, n_per)

        if n_per is not None:
            B = torch.randint(0, 2, (n_per, n_idx), dtype=torch.float).to(device) * 2.0 - 1.0 # TODO: or use (n_per, n)?
            B = torch.einsum('bi,bj->bij', B, B)
        for i in range(n_b):

            if "mahalanobis" in self.kernels[i].lower():
                hsic, (h_X, h_Y) = hsic_so(self.kernels[i], self.bandwidths[i], distXY=(M_distXX[i-sum(self.n_bandwidth[:2])], M_distYY[i-sum(self.n_bandwidth[:2])]))
            elif "deep_mahalanobis" in self.kernels[i].lower():
                hsic, (h_X, h_Y) = hsic_so(self.kernels[i], self.bandwidths[i], distXY=(M_rep_distXX[i-sum(self.n_bandwidth[:5])], M_rep_distYY[i-sum(self.n_bandwidth[:5])]))
            elif "deep" in self.kernels[i].lower():
                hsic, (h_X, h_Y) = hsic_so(self.kernels[i], self.bandwidths[i], distXY=(rep_distXX, rep_distYY))
            else:
                hsic, (h_X, h_Y) = hsic_so(self.kernels[i], self.bandwidths[i], distXY=(distXX, distYY))

            U.append(hsic)
            if n_per is not None:
                hsic_b = torch.sum(B*h_X*h_Y, dim=(1, 2)) / math.comb(n_idx, 2) 
                U_b.append(hsic_b)

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
            X = X_test
            Y = Y_test
        else:
            X = self.X
            Y = self.Y

        n = len(X)
        n_idx = np.floor(n/2).astype(int)

        indices = torch.randperm(n, device=X.device)
        X_null = X[indices]
        Y_null = Y[indices]
        distXX = torch.cdist(X_null, X_null, p=2)
        distYY = torch.cdist(Y_null, Y_null, p=2)

        try:
            M_distXX = torch.stack([Pdist2(X_null, X_null, M[0]) for M in self.M_matrix], dim=0)
            M_distYY = torch.stack([Pdist2(Y_null, Y_null, M[1]) for M in self.M_matrix], dim=0)
        except:
            pass

        try:
            X_rep = self.encoder(X_null)
            rep_distXX = torch.cdist(X_rep, X_rep, p=2)
        except:
            pass

        try:
            Y_rep = self.encoder(Y_null)
            rep_distYY = torch.cdist(Y_rep, Y_rep, p=2)
        except:
            Y_rep = Y_null
            rep_distYY = distYY

        try:
            M_rep_distXX = torch.stack([Pdist2(X_rep, X_rep, M[0]) for M in self.M_matrix_rep], dim=0)
            M_rep_distYY = torch.stack([Pdist2(Y_rep, Y_rep, M[1]) for M in self.M_matrix_rep], dim=0)
        except:
            pass

        Sigma = torch.zeros(n_b, n_b).to(device, torch.float32)
        C = self.get_C(n_idx)
        for i in range(n_b):
            for j in range(i, n_b):
                if "mahalanobis" in self.kernels[i].lower():
                    _, (h_X1, h_Y1) = hsic_so(self.kernels[i], self.bandwidths[i], distXY=(M_distXX[i-sum(self.n_bandwidth[:2])], M_distYY[i-sum(self.n_bandwidth[:2])]))
                elif "deep_mahalanobis" in self.kernels[i].lower():
                    _, (h_X1, h_Y1) = hsic_so(self.kernels[i], self.bandwidths[i], distXY=(M_rep_distXX[i-sum(self.n_bandwidth[:5])], M_rep_distYY[i-sum(self.n_bandwidth[:5])]))
                elif "deep" in self.kernels[i].lower():
                    _, (h_X1, h_Y1) = hsic_so(self.kernels[i], self.bandwidths[i], distXY=(rep_distXX, rep_distYY))
                else:
                    _, (h_X1, h_Y1) = hsic_so(self.kernels[i], self.bandwidths[i], distXY=(distXX, distYY))
                h_matrix1 = h_X1 * h_Y1

                if "mahalanobis" in self.kernels[j].lower():
                    _, (h_X2, h_Y2) = hsic_so(self.kernels[j], self.bandwidths[j], distXY=(M_distXX[j-sum(self.n_bandwidth[:2])], M_distYY[j-sum(self.n_bandwidth[:2])]))
                elif "deep_mahalanobis" in self.kernels[j].lower():
                    _, (h_X2, h_Y2) = hsic_so(self.kernels[j], self.bandwidths[j], distXY=(M_rep_distXX[j-sum(self.n_bandwidth[:5])], M_rep_distYY[j-sum(self.n_bandwidth[:5])]))
                elif "deep" in self.kernels[j].lower():
                    _, (h_X2, h_Y2) = hsic_so(self.kernels[j], self.bandwidths[j], distXY=(rep_distXX, rep_distYY))
                else:
                    _, (h_X2, h_Y2) = hsic_so(self.kernels[j], self.bandwidths[j], distXY=(distXX, distYY))
                h_matrix2 = h_X2 * h_Y2

                # mask = torch.triu(torch.ones(n_idx, n_idx), diagonal=1).to(device)
                # Sigma[i, j] = C * (h_matrix1 * h_matrix2 * mask).sum()

                Sigma[i, j] = C * (h_matrix1 * h_matrix2).sum()
                Sigma[j, i] = Sigma[i, j]
        return Sigma + self.reg * torch.eye(n_b).to(device, torch.float32)

    def train_kernel(self, N_epoch, batch_size, learning_rate):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param)
        device = check_device()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        n = self.X.size(0)
        batch_size = min(n, batch_size)
        batches=n//batch_size
        with torch.no_grad():
            U = self.compute_U_stats()
            Sigma = self.compute_Sigma()
            T = n**2 * U.T @ torch.inverse(Sigma) @ U
            print(f"Epoch 0: T = {T.item()}")

            # L = torch.inverse(sqrtm(Sigma))
            # half = L @ U
            # print("check_train", n**2 * torch.norm(half, p=2)**2)
            
        for epoch in range(N_epoch):
            indices = torch.randperm(n, device=device)
            # X and Y are pairs, with same index
            X = self.X[indices]
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
                        for j in range(2):
                            eigvalues, eigvectors = torch.linalg.eig(self.M_matrix[i][j])
                            eigvalues = torch.max(eigvalues.real, torch.tensor(1e-5).to(check_device(), torch.float))
                            eigvectors = eigvectors.real
                            eigvectors = eigvectors.t().reshape(eigvectors.shape[0], -1, eigvectors.shape[1])
                            self.M_matrix[i][j] = eigvalues[0] * eigvectors[0].t() * eigvectors[0]
                            for k in range(1, len(eigvalues)):
                                self.M_matrix[i][j] += eigvalues[k] * eigvectors[k] * eigvectors[k].t()
                    for i in range(len(self.M_matrix_rep)):
                        for j in range(2):
                            eigvalues, eigvectors = torch.linalg.eig(self.M_matrix_rep[i][j])
                            eigvalues = torch.max(eigvalues.real, torch.tensor(1e-5).to(check_device(), torch.float))
                            eigvectors = eigvectors.real
                            eigvectors = eigvectors.t().reshape(eigvectors.shape[0], -1, eigvectors.shape[1])
                            self.M_matrix_rep[i][j] = eigvalues[0] * eigvectors[0].t() * eigvectors[0]
                            for k in range(1, len(eigvalues)):
                                self.M_matrix_rep[i][j] += eigvalues[k] * eigvectors[k] * eigvectors[k].t()

    def get_C(self, n):
        return (
            (n**2)
            / (math.comb(n, 2))**2
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
            L_te = self.L_tr  # L_te.size() = (c, c) column vector
        else:
            L_te = torch.inverse(
                sqrtm(self.compute_Sigma(X_test=X_test, Y_test=Y_test)))

        # Sigma = self.compute_Sigma()
        # T = n_te**2 * U_te.T @ torch.inverse(Sigma) @ U_te
        # # print(f"T = {T.item()}")  
        # # print("check", n_te**2 * U_te.T @ torch.inverse(self.compute_Sigma(X_test=X_test, Y_test=Y_test)) @ U_te)
        # L = torch.inverse(sqrtm(Sigma))
        # # print(L, L_te)
        # # half_te = L @ U_te
        # # print("check_test", n_te**2 * torch.norm(half_te, p=2)**2)
        # L_te = L

        half_te = L_te @ U_te
        # print("check", n_te**2 * torch.norm(half_te, p=2)**2)
        half_te_b = L_te @ U_b
        T_obs_no_select = n_te**2 * torch.norm(half_te, p=2)**2
        T_b_no_select = n_te**2 * torch.norm(half_te_b, p=2, dim=0)**2
        p_value_no_select = torch.sum(T_b_no_select > T_obs_no_select) / n_per
        
        F_te = torch.sign(half_te)
        F_te_b = torch.sign(half_te_b) #[:, 0:2]
        F = F_te == self.F_tr
        global count, th, th_s
        # count += F.reshape(-1).cpu().numpy()
        F_b = F_te_b == self.F_tr
        T_obs_select = n_te**2 * torch.norm(F * half_te, p=2)**2
        T_b_select = n_te**2 * torch.norm(F_b * half_te_b, p=2, dim=0)**2
        # print(torch.sort(T_b_no_select))
        th.append(torch.sort(T_b_no_select)[0][94].item())
        th_s.append(torch.sort(T_b_select)[0][94].item())
        # th.append(T_obs_no_select.item())
        # th_s.append(T_obs_select.item())
        # print(T_obs_no_select, th, T_obs_select, th_s)
        p_value_select = torch.sum(T_b_select > T_obs_select) / n_per

        return p_value_select, p_value_no_select

def train_HSIC_AU(name, N1, rs, check, N_epoch, batch_size, learning_rate, n_bandwidth, reg=1e-5, way=['Agg', 'Agg', 'Agg', 'Agg', 'Agg', 'Agg'], is_cov=True, encoder=None):
    device = check_device()
    np.random.seed(rs)
    torch.manual_seed(rs)
    X_train, Y_train = load_data(name, N1, rs, check)
    X_train, Y_train = MatConvert(X_train, device, torch.float32), MatConvert(Y_train, device, torch.float32)
    # print(X_train.shape, Y_train.shape)
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
        # model_au.X = None
        # model_au.Y = None
        return model_au

def IDT_HSIC_AU(name, N1, rs, check, n_test, n_per, alpha, is_cov_tr, model_au):
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
        H_HSIC_AU_select = np.zeros(n_test)
        H_HSIC_AU_no_select = np.zeros(n_test)
        P_HSIC_AU_select = np.zeros(n_test)
        P_HSIC_AU_no_select = np.zeros(n_test)
        N_test_all = 10 * N1
        X_test_all, Y_test_all = load_data(name, N_test_all, rs + 283, check)
        test_time = 0
        for k in range(n_test):
            ind_test = np.random.choice(N_test_all, N1, replace=False)
            X_test = X_test_all[ind_test]
            Y_test = Y_test_all[ind_test]
            X_test, Y_test = MatConvert(X_test, device, torch.float32), MatConvert(Y_test, device, torch.float32)

            start_time = time.time()
            p_value_select, p_value_no_select = model_au.test_bootstrap(X_test, Y_test, N1, n_per)
            test_time += time.time() - start_time
        
            P_HSIC_AU_select[k] = p_value_select
            P_HSIC_AU_no_select[k] = p_value_no_select
            H_HSIC_AU_select[k] = p_value_select < alpha
            H_HSIC_AU_no_select[k] = p_value_no_select < alpha
            # break

        return H_HSIC_AU_select, H_HSIC_AU_no_select, P_HSIC_AU_select, P_HSIC_AU_no_select, model_au.train_time, test_time

class DefaultImageModel(nn.Module):

    def __init__(self, n_channels=3, weights='DEFAULT', image_size=32):
        super().__init__()
        self.image_size = image_size
        self.n_channels = n_channels

        # Load base ResNet with new weights parameter
        if weights == 'DEFAULT':
            weights = torchvision.models.ResNet18_Weights.DEFAULT
            self.resnet = torchvision.models.resnet18(weights=weights)
        else:
            self.resnet = torchvision.models.resnet18(weights=None)

        # Modify input layer if needed
        if n_channels != 3:
            self.resnet.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7,
                                            stride=2, padding=3, bias=False)

        # For very small images (like MNIST 28x28 or CIFAR 32x32)
        if image_size < 64:
            # Modify first conv layer to have smaller stride
            self.resnet.conv1 = nn.Conv2d(n_channels, 64, kernel_size=3,
                                            stride=1, padding=1, bias=False)
            # Remove maxpool layer
            self.resnet.maxpool = nn.Identity()

        # Store the original fc layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 100)

    def forward(self, x):
        x = x.view(-1, self.n_channels, self.image_size, self.image_size)
        output = self.resnet(x)
        return output
    
n_channels = 1
image_size = 28
SEED = 208
torch.manual_seed(SEED)
model = DefaultImageModel(n_channels=n_channels, image_size=image_size, weights='DEFAULT')
for param in model.parameters():
    param.requires_grad = False

name = "MNIST"
N1 = 100
check = 1

th_total = []
th_s_total = []
result_s = []
result_no_s = []
# count_total = np.array([])

def insert_row(empty_arr, new_row):
    if empty_arr.size == 0:
        return new_row
    else:
        return np.vstack((empty_arr, new_row))

model_au = train_HSIC_AU(name, N1, SEED, check, 0, 128, 0.0005, (2, 2, 0, 3, 3, 0), reg=1e-8, way=['Fuse', 'Fuse', 'Agg', 'Fuse', 'Fuse', 'Agg'], is_cov=True, encoder=model)
for seed in range(SEED, SEED+10):
    # model_au = train_HSIC_AU(name, N1, SEED, check, 0, 128, 0.0005, (3, 3, 0, 0, 0, 0), reg=1e-8, way=['Fuse', 'Fuse', 'Agg', 'Agg', 'Agg', 'Agg'], is_cov=True, encoder=None)
    # model_au = train_HSIC_AU(name, N1, seed, check, 50, 128, 0.0005, (6, 0, 0, 0, 0, 0), reg=1e-8, way=['Fuse', 'Fuse', 'Agg', 'Agg', 'Agg', 'Agg'], is_cov=True, encoder=None)
    # count = np.zeros(6)
    th = []
    th_s = []
    H_HSIC_AU_select, H_HSIC_AU_no_select, P_HSIC_AU_select, P_HSIC_AU_no_select, _, _ = IDT_HSIC_AU(
        name, N1, seed, check, 100, 100, 0.05, False, model_au)
    # print(count/100)
    # count_total = insert_row(count_total, (count/100))
    result_s.append(np.sum(H_HSIC_AU_select))
    result_no_s.append(np.sum(H_HSIC_AU_no_select))
    th_total.append(np.mean(th))
    th_s_total.append(np.mean(th_s))
    # break

# mean = np.mean(count_total, axis=0)
# std = np.std(count_total, axis=0)/np.sqrt(10)
# for i in range(6):
#     print("{:.3f}±{:.3f}".format(mean[i], std[i]))
print("{:.3f}±{:.3f}".format(np.mean(result_s)/100, np.std(result_s)/np.sqrt(10)/100))
print("{:.3f}±{:.3f}".format(np.mean(result_no_s)/100, np.std(result_no_s)/np.sqrt(10)/100))
print(np.mean(th_total))
print(np.mean(th_s_total))

# tensor(107.8316, device='cuda:0') [77.551513671875] tensor(83.6041, device='cuda:0') [58.239383697509766]
