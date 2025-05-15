import numpy as np
import torch
from sklearn.utils import check_random_state
from sklearn.preprocessing import MinMaxScaler
import pickle
import torchvision.transforms as transforms
from torchvision import datasets
import sys
import os
import random
sys.path.append(os.path.abspath('../..'))

def load_data(name, N, rs, check=1, scale=True):
    """Load independence testing datasets.

    Args:
        name (string): name of the datasets
        N (int): size of sample
        rs (int): random seed
        check (int, optional): 1 for H_1, 0 for H_0. Defaults to 1.
        scale (bool, optional): whether to scale data in same range. Defaults to True.
    """
    if name == "HIGGS":
        X, Y = sample_HIGGS(N, rs, check, scale)
    ### MNIST + Cifar10 + Imagenet
    ### modified version denotes the Y are images with noise
    elif name =="MNIST":
        X, Y = load_data_MNIST(N, rs, check)
    # elif name =="MNIST_modified":
    #     X, Y = load_data_MNIST_modified(N, rs, check)
    elif name =="CIFAR10":
        X, Y = load_data_CIFAR10(N, rs, check)
    # elif name =="CIFAR10_modified":
    #     X, Y = load_data_CIFAR10_modified(N, rs, check)
    elif name =="ImageNet":
        X, Y = load_data_ImageNet(N, rs, check)
    # elif name =="ImageNet_modified":
    #     X, Y = load_data_ImageNet_modified(N, rs, check)
    else:
        print('No Dataset: ', name)

    return X, Y
        
    
        
def MMscaler(X, Y):
    scaler = MinMaxScaler()
    scaler.fit(np.concatenate((X, Y), axis=0))
    return scaler.transform(X), scaler.transform(Y)

def MatConvert(x, device, dtype):
    """convert the numpy to a torch tensor."""
    x = torch.from_numpy(x).to(device, dtype)
    return x
        
        
def sample_HIGGS(N, rs, check, scale, proportion=0.90):
    """#Feat. 4  #Class 2  #Inst. [5170877,5829123]"""
    np.random.seed(seed=rs)
    
    try:
        data = pickle.load(open('/data/gpfs/projects/punim2335/data/HIGGS_TST.pckl', 'rb'))
    except Exception as e:
        data = pickle.load(open('/data/gpfs/projects/punim2335/data/HIGGS_TST.pckl', 'rb'))
        
    dataX = data[0]
    N1_T = dataX.shape[0]
    dataY = dataX
    N2_T = dataY.shape[0]
    ind1 = np.random.choice(N1_T, N, replace=False)
    ind2 = np.random.choice(N2_T, N, replace=False)
    X = dataX[ind1, :4]
    Y = dataY[ind2, :4]
    
    if check == 1:
        # add dependency between X and Y
        mask = np.random.random(len(Y)) > proportion
        if np.any(mask):
            Y[mask] = add_perturbation(X[mask])
    del data
    
    if scale:
        X, Y = MMscaler(X, Y)
    return X, Y
    
def add_perturbation(X, epsilon=0.05):
    X = np.asarray(X)
    pert = np.empty_like(X)
    pert[:, 0] = np.sin(X[:, 0])
    pert[:, 1] = np.log1p(np.abs(X[:, 1]))  # log1p(x) = log(1+x)
    pert[:, 2] = np.cos(X[:, 2])
    pert[:, 3] = np.sqrt(np.abs(X[:, 3]))
    
    return X + epsilon * pert



def load_data_MNIST(N, rs, check=1):
    """
    Load MNIST dataset and generate data pairs for independence testing.
    
    Parameters:
        N (int): Number of samples to select.
        rs (int): Random seed for reproducibility.
        check (int, optional): 0 for H₀ (independent sampling of X and Y),
                               1 for H₁ (perturb Y so that only 25% are correct, 75% random).
        scale (bool, optional): Whether to apply additional scaling (MNIST images are already normalized to [0,1]).
        
    Returns:
        X (numpy.ndarray): Sampled image data with shape (N, 1, 28, 28).
        Y (numpy.ndarray): Corresponding labels with shape (N,).
    """
    # Set the random seed for reproducibility
    np.random.seed(seed=rs)
    random.seed(rs)
    
    # Use torchvision to load the MNIST dataset
    transform = transforms.ToTensor()  # Convert images to tensors and normalize to [0,1]
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Convert dataset to numpy arrays for easier manipulation
    X_all = []
    Y_all = []
    for img, label in mnist_dataset:
        # img is a tensor of shape [1, 28, 28]. Convert it to a numpy array.
        X_all.append(img.numpy())
        Y_all.append(label)
    X_all = np.array(X_all)  # Shape: (60000, 1, 28, 28)
    Y_all = np.array(Y_all)  # Shape: (60000,)
    
    if check == 0:
        # H₀: Independent sampling of images and labels
        indices_x = np.random.choice(len(X_all), N, replace=False)
        indices_y = np.random.choice(len(Y_all), N, replace=False)
        X = X_all[indices_x]
        Y = Y_all[indices_y]
    else:
        # H₁: Paired sampling, then perturb the labels
        indices = np.random.choice(len(X_all), N, replace=False)
        X = X_all[indices]
        Y = Y_all[indices].copy()  # Create a copy to avoid modifying the original labels
        
        # Perturb the labels: 75% chance to replace with a random incorrect label, 25% keep original
        for i in range(N):
            if np.random.rand() < 0.75:
                # Create a list of candidate labels (0-9) excluding the original label
                candidates = list(range(10))
                candidates.remove(Y[i])
                Y[i] = np.random.choice(candidates)
            # Otherwise (25%), the label remains unchanged
    
    # Reshape X from (N, 1, 28, 28) to (N, 784)
    X = X.reshape(N, -1)
    # Reshape Y from (N,) to (N, 1)
    Y = Y.reshape(-1, 1)
    
    return X.astype('float64'), Y.astype('float64')

def load_data_MNIST_modified(N, rs, check=1):
    """
    Load MNIST dataset and generate paired image data for independence testing.
    
    Parameters:
        N (int): Number of samples to select.
        rs (int): Random seed for reproducibility.
        check (int): 0 for H₀ (independent sampling of X and Y), 
                     1 for H₁ (generate Y by adding noise to X).
    
    Returns:
        X (numpy.ndarray): Sampled image data with shape (N, 1, 28, 28).
        Y (numpy.ndarray): For H₀, independent images (N, 1, 28, 28);
                           for H₁, images obtained by adding noise to X (N, 1, 28, 28).
    """
    # Set the random seed for reproducibility
    np.random.seed(seed=rs)
    random.seed(rs)
    # Load MNIST dataset using torchvision
    transform = transforms.ToTensor()  # Converts images to tensors and normalizes to [0, 1]
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Convert the dataset to a numpy array of images
    X_all = []
    for img, label in mnist_dataset:
        # Each img is a tensor of shape [1, 28, 28]. Convert it to a numpy array.
        X_all.append(img.numpy())
    X_all = np.array(X_all)  # Shape: (60000, 1, 28, 28)
    
    if check == 0:
        # H₀: Independent sampling of X and Y
        indices_x = np.random.choice(len(X_all), N, replace=False)
        indices_y = np.random.choice(len(X_all), N, replace=False)
        X = X_all[indices_x]
        Y = X_all[indices_y]
    else:
        # H₁: Paired sampling: sample X, then generate Y by adding noise to X
        indices = np.random.choice(len(X_all), N, replace=False)
        X = X_all[indices]
        Y = np.empty_like(X)
        # Generate base noise using a normal distribution (mean 0, moderate std dev)
        base_noise = np.random.normal(loc=0.0, scale=0.2, size=X[0].shape)
        # For each sample image, add noise to generate Y
        for i in range(N):
            Y[i] = add_noise(X[i],base_noise)
            
    # Flatten the images: reshape from (N, 1, 28, 28) to (N, 784)
    X = X.reshape(N, -1)
    Y = Y.reshape(N, -1)
    
    return X.astype('float64'), Y.astype('float64')

def load_data_CIFAR10(N, rs, check=1):
    """
    Load CIFAR10 dataset and generate data pairs for independence testing.
    
    Parameters:
        N (int): Number of samples to select.
        rs (int): Random seed for reproducibility.
        check (int, optional): 
            0 for H₀ (independent sampling of images and labels),
            1 for H₁ (paired sampling with label perturbation: 75% chance to change label).
    
    Returns:
        X (numpy.ndarray): Sampled image data with shape (N, 3, 32, 32).
        Y (numpy.ndarray): Corresponding labels with shape (N,).
    """
    # Set the random seed for reproducibility
    np.random.seed(rs)
    random.seed(rs)
    
    # Use torchvision to load the CIFAR10 dataset.
    # The transform converts images to tensors and normalizes pixel values to [0, 1].
    transform = transforms.ToTensor()
    cifar_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Convert dataset to numpy arrays for easier manipulation.
    X_all = []
    Y_all = []
    for img, label in cifar_dataset:
        # Each img is a tensor of shape [3, 32, 32]. Convert it to a numpy array.
        X_all.append(img.numpy())
        Y_all.append(label)
    X_all = np.array(X_all)  # Shape: (50000, 3, 32, 32)
    Y_all = np.array(Y_all)  # Shape: (50000,)
    
    if check == 0:
        # H₀: Independent sampling of images and labels.
        indices_x = np.random.choice(len(X_all), N, replace=False)
        indices_y = np.random.choice(len(Y_all), N, replace=False)
        X = X_all[indices_x]
        Y = Y_all[indices_y]
    else:
        # H₁: Paired sampling, then perturb the labels.
        indices = np.random.choice(len(X_all), N, replace=False)
        X = X_all[indices]
        Y = Y_all[indices].copy()  # Copy to avoid modifying the original labels.
        
        # Perturb the labels: 75% chance to replace with a random incorrect label,
        # 25% chance to keep the original label.
        for i in range(N):
            if np.random.rand() < 0.5:
                # Create a list of candidate labels (0-9) excluding the original label.
                candidates = list(range(10))
                candidates.remove(Y[i])
                Y[i] = np.random.choice(candidates)
    
    # Reshape X from (N, 3, 32, 32) to (N, 3072)
    X = X.reshape(N, -1)
    # Reshape Y from (N,) to (N, 1)
    Y = Y.reshape(-1, 1)
    return X.astype('float64'), Y.astype('float64')


def load_data_CIFAR10_modified(N, rs, check=1):
    """
    Load CIFAR10 dataset and generate paired image data for independence testing.
    
    Parameters:
        N (int): Number of samples to select.
        rs (int): Random seed for reproducibility.
        check (int): 
            0 for H₀ (independent sampling of images for X and Y),
            1 for H₁ (generate Y by adding noise to X).
    
    Returns:
        X (numpy.ndarray): Sampled image data with shape (N, 3, 32, 32).
        Y (numpy.ndarray): For H₀, independent images (N, 3, 32, 32);
                           for H₁, images obtained by adding noise to X (N, 3, 32, 32).
    """
    # Set the random seed for reproducibility
    np.random.seed(rs)
    random.seed(rs)
    # Load CIFAR10 dataset using torchvision.
    transform = transforms.ToTensor()
    cifar_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Convert the dataset to a numpy array of images.
    X_all = []
    for img, _ in cifar_dataset:
        # Each img is a tensor of shape [3, 32, 32]. Convert it to a numpy array.
        X_all.append(img.numpy())
    X_all = np.array(X_all)  # Shape: (50000, 3, 32, 32)
    
    if check == 0:
        # H₀: Independent sampling of images for X and Y.
        indices_x = np.random.choice(len(X_all), N, replace=False)
        indices_y = np.random.choice(len(X_all), N, replace=False)
        X = X_all[indices_x]
        Y = X_all[indices_y]
    else:
        # H₁: Paired sampling: sample X, then generate Y by adding noise to X.
        indices = np.random.choice(len(X_all), N, replace=False)
        X = X_all[indices]
        Y = np.empty_like(X)  # Preallocate an array for Y with the same shape as X.
        # Generate base noise using a normal distribution (mean 0, moderate std dev)
        base_noise = np.random.normal(loc=0.0, scale=0.2, size=X[0].shape)
        for i in range(N):
            Y[i] = add_noise(X[i],base_noise)
            
    # Flatten the images: reshape X and Y from (N, 3, 32, 32) to (N, 3072)
    X = X.reshape(N, -1)
    Y = Y.reshape(N, -1)
    
    return X.astype('float64'), Y.astype('float64')


def load_data_ImageNet(N, rs, check=1):
    """
    Load ImageNet dataset and generate data pairs for independence testing.
    
    Parameters:
        N (int): Number of samples to select.
        rs (int): Random seed for reproducibility.
        check (int, optional): 
            0 for H₀ (independent sampling of images and labels),
            1 for H₁ (paired sampling with label perturbation: 75% chance to change label).
    
    Returns:
        X (numpy.ndarray): Sampled image data with shape (N, 3, 224, 224).
        Y (numpy.ndarray): Corresponding labels with shape (N,).
    """
    # Set random seed for reproducibility
    np.random.seed(rs)
    random.seed(rs)
    
    # Define transformations: resize to 256, center-crop to 224, then convert to tensor
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()  # Converts to tensor and normalizes pixel values to [0, 1]
    ])
    
    # Load the ImageNet dataset (assuming the dataset is available at './data/imagenet')
    imagenet_dataset = datasets.ImageNet(root='./data/imagenet', split='train', transform=transform)
    
    # Convert dataset to numpy arrays for easier manipulation
    X_all = []
    Y_all = []
    for img, label in imagenet_dataset:
        # Each img is a tensor of shape [3, 224, 224]. Convert it to a numpy array.
        X_all.append(img.numpy())
        Y_all.append(label)
    X_all = np.array(X_all)  # Expected shape: (num_samples, 3, 224, 224)
    Y_all = np.array(Y_all)  # Expected shape: (num_samples,)
    
    if check == 0:
        # H₀: Independent sampling of images and labels.
        indices_x = np.random.choice(len(X_all), N, replace=False)
        indices_y = np.random.choice(len(Y_all), N, replace=False)
        X = X_all[indices_x]
        Y = Y_all[indices_y]
    else:
        # H₁: Paired sampling, then perturb the labels.
        indices = np.random.choice(len(X_all), N, replace=False)
        X = X_all[indices]
        Y = Y_all[indices].copy()  # Copy to avoid modifying the original labels.
        
        # Perturb the labels: for each sample, with 75% probability change the label.
        # Note: ImageNet typically has 1000 classes labeled from 0 to 999.
        for i in range(N):
            if np.random.rand() < 0.75:
                candidates = list(range(1000))  # Adjust according to the actual number of classes.
                candidates.remove(Y[i])
                Y[i] = np.random.choice(candidates)
                
    # Flatten the images: reshape from (N, 3, 224, 224) to (N, 150528)
    X = X.reshape(N, -1)
    # Reshape labels Y from (N,) to (N, 1)
    Y = Y.reshape(-1, 1)
    
    return X.astype('float64'), Y.astype('float64')


def load_data_ImageNet_modified(N, rs, check=1):
    """
    Load ImageNet dataset and generate paired image data for independence testing.
    
    Parameters:
        N (int): Number of samples to select.
        rs (int): Random seed for reproducibility.
        check (int):
            0 for H₀ (independent sampling of images for X and Y),
            1 for H₁ (generate Y by adding noise to X).
    
    Returns:
        X (numpy.ndarray): Sampled image data with shape (N, 3, 224, 224).
        Y (numpy.ndarray): For H₀, independent images (N, 3, 224, 224);
                           for H₁, images obtained by adding noise to X (N, 3, 224, 224).
    """
    # Set random seed for reproducibility
    np.random.seed(rs)
    random.seed(rs)
    
    # Define transformations: resize to 256, center-crop to 224, then convert to tensor.
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    
    # Load the ImageNet dataset (assuming the dataset is available at './data/imagenet')
    imagenet_dataset = datasets.ImageNet(root='./data/imagenet', split='train', transform=transform)
    
    # Convert the dataset to a numpy array of images.
    X_all = []
    for img, _ in imagenet_dataset:
        # Each img is a tensor of shape [3, 224, 224]. Convert it to a numpy array.
        X_all.append(img.numpy())
    X_all = np.array(X_all)  # Expected shape: (num_samples, 3, 224, 224)
    
    if check == 0:
        # H₀: Independent sampling of images for X and Y.
        indices_x = np.random.choice(len(X_all), N, replace=False)
        indices_y = np.random.choice(len(X_all), N, replace=False)
        X = X_all[indices_x]
        Y = X_all[indices_y]
    else:
        # H₁: Paired sampling: sample X, then generate Y by adding noise to X.
        indices = np.random.choice(len(X_all), N, replace=False)
        X = X_all[indices]
        Y = np.empty_like(X)  # Preallocate an array for Y with the same shape as X.
        # Generate base noise using a normal distribution (mean 0, moderate std dev)
        base_noise = np.random.normal(loc=0.0, scale=0.2, size=X[0].shape)
        for i in range(N):
            Y[i] = add_noise(X[i],base_noise)
    
    # Flatten the images: reshape from (N, 3, 224, 224) to (N, 150528)
    X = X.reshape(N, -1)
    Y = Y.reshape(N, -1)
    
    return X.astype('float64'), Y.astype('float64')

def add_noise(image, base_noise,epsilon=0.2):
    """
    Add noise to a single image to generate a new image.
    
    The noise is designed to be:
      1. Not fixed: A different random noise is generated for each image.
      2. Not too hard: The noise level is moderate.
      3. Not too simple: The noise includes both additive and multiplicative randomness.
    
    Parameters:
        image (numpy.ndarray): Input image with shape (1, 28, 28).
        epsilon (float): Scaling factor for the noise.
    
    Returns:
        noisy_image (numpy.ndarray): The image with added noise, clipped to the range [0, 1].
    """
    # Introduce additional variation by applying a random scaling factor between 0.8 and 1.2
    scaling_factor = np.random.uniform(0.8, 1.2, size=image.shape)
    noise = base_noise * scaling_factor
    # Add the noise to the original image with scaling factor epsilon
    noisy_image = image + epsilon * noise
    # Ensure the pixel values remain in the valid range [0, 1]
    noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image
