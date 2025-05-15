import numpy as np
import torch
from sklearn.utils import check_random_state
from sklearn.preprocessing import MinMaxScaler
import pickle
import torchvision.transforms as transforms
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import datasets
from collections import defaultdict
import json

def load_data(name, N, rs, check=1, scale=True):
    if name == 'BLOB':
        X, Y = sample_BLOB(N, rs, check, scale)
    elif name == 'HDGM':
        X, Y = sample_HDGM(N, rs, check, scale)
    elif name == 'HIGGS':
        X, Y = sample_HIGGS(N, rs, check, scale)
    elif name == 'MNIST':
        X, Y = sample_MNIST(N, rs, check, scale)   
    elif name == 'CIFAR10':
        X, Y = sample_CIFAR10(N, rs, check, scale)
    elif name == 'ImageNetV2':
        X, Y = sample_ImageNetV2(N, rs, check, scale)
    elif name == 'ImageNetR':
        X, Y = sample_ImageNetR(N, rs, check, scale)
    elif name == 'ImageNetSK':
        X, Y = sample_ImageNetSK(N, rs, check, scale)
    elif name == 'ImageNetA':
        X, Y = sample_ImageNetA(N, rs, check, scale)
    elif name == "ImageNetV2_Fea":
        X, Y = sample_ImageNetV2_Fea(N, rs, check, scale)
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

def sample_BLOB(N, rs, check, scale):
    """#Feat. 2  # Inst. inf"""
    rs = check_random_state(rs)
    rows = 3
    cols = 3
    if check == 0:
        """Generate Blob-S for testing type-I error"""
        sep = 1
        correlation = 0
        # generate within-blob variation
        mu = np.zeros(2)
        sigma = np.eye(2)
        X = rs.multivariate_normal(mu, sigma, size=N)
        corr_sigma = np.array([[1, correlation], [correlation, 1]])
        Y = rs.multivariate_normal(mu, corr_sigma, size=N)
        # assign to blobs
        X[:, 0] += rs.randint(rows, size=N) * sep
        X[:, 1] += rs.randint(cols, size=N) * sep
        Y[:, 0] += rs.randint(rows, size=N) * sep
        Y[:, 1] += rs.randint(cols, size=N) * sep
    else:
        """Generate Blob-D for testing type-II error (or test power)"""
        sigma_mx_2_standard = np.array([[0.03, 0], [0, 0.03]])
        sigma_mx_2 = np.zeros([9, 2, 2])
        for i in range(9):
            sigma_mx_2[i] = sigma_mx_2_standard
            if i < 4:
                sigma_mx_2[i][0, 1] = -0.02 - 0.002 * i
                sigma_mx_2[i][1, 0] = -0.02 - 0.002 * i
            if i == 4:
                sigma_mx_2[i][0, 1] = 0.00
                sigma_mx_2[i][1, 0] = 0.00
            if i > 4:
                sigma_mx_2[i][1, 0] = 0.02 + 0.002 * (i - 5)
                sigma_mx_2[i][0, 1] = 0.02 + 0.002 * (i - 5)

        mu = np.zeros(2)
        sigma = np.eye(2) * 0.03
        X = rs.multivariate_normal(mu, sigma, size=N)
        Y = rs.multivariate_normal(mu, np.eye(2), size=N)
        # assign to blobs
        X[:, 0] += rs.randint(rows, size=N)
        X[:, 1] += rs.randint(cols, size=N)
        Y_row = rs.randint(rows, size=N)
        Y_col = rs.randint(cols, size=N)
        locs = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
        for i in range(9):
            corr_sigma = sigma_mx_2[i]
            L = np.linalg.cholesky(corr_sigma)
            ind = np.expand_dims((Y_row == locs[i][0]) & (Y_col == locs[i][1]), 1)
            ind2 = np.concatenate((ind, ind), 1)
            Y = np.where(ind2, np.matmul(Y, L) + locs[i], Y)
    if scale:
        X, Y = MMscaler(X, Y)
    return X, Y

def sample_HDGM(N, rs, check, scale):
    """#Feat. 10  # Inst. inf"""
    d = 10 # data dim
    Num_clusters = 2  # number of modes
    n = int(N / Num_clusters)
    mu_mx = np.zeros([Num_clusters, d])
    mu_mx[1] = mu_mx[1] + 0.5
    sigma_mx_1 = np.identity(d)
    X = np.zeros([n * Num_clusters, d])
    Y = np.zeros([n * Num_clusters, d])
    # Generate HDGM-D
    for i in range(Num_clusters):
        np.random.seed(seed=rs + i + 283)
        X[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n)
    for i in range(Num_clusters):
        np.random.seed(seed=rs + i)
        if check == 0:
            Y[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n)
        else:
            sigma_mx_2 = [np.identity(d), np.identity(d)]
            sigma_mx_2[0][0, 1] = 0.5
            sigma_mx_2[0][1, 0] = 0.5
            sigma_mx_2[1][0, 1] = -0.5
            sigma_mx_2[1][1, 0] = -0.5
            Y[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_2[i], n)
    if scale:
        X, Y = MMscaler(X, Y)
    return X, Y

def sample_HIGGS(N, rs, check, scale):
    """#Feat. 4  #Class 2  #Inst. [5170877,5829123]"""
    np.random.seed(seed=rs)
    data = pickle.load(open('/data/gpfs/projects/punim2335/data/HIGGS_TST.pckl', 'rb'))
    if check == 0:
        dataX = data[0]
        dataY = data[0]
    else:
        dataX = data[0]
        dataY = data[1]
    del data

    N1_T = dataX.shape[0]
    N2_T = dataY.shape[0]
    ind1 = np.random.choice(N1_T, N, replace=False)
    ind2 = np.random.choice(N2_T, N, replace=False)
    X = dataX[ind1, :4]
    Y = dataY[ind2, :4]
    if scale:
        X, Y = MMscaler(X, Y)
    return X, Y

def sample_MNIST(N, rs, check, scale):
    """#Feat. 28*28  #Class 2  #Inst. [10000,10000]"""
    np.random.seed(seed=rs)
    torch.manual_seed(seed=rs)
    # True_MNIST
    img_size = 32
    dataloader_FULL_te = torch.utils.data.DataLoader(
    datasets.MNIST(
        "/data/gpfs/projects/punim2335/data/mnist",
        train=False,
        download=False,
        transform=transforms.Compose(
            [transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
                ]
        ),
    ),
    batch_size=10000,
    shuffle=True,
    )

    for i, (imgs, Labels) in enumerate(dataloader_FULL_te):
        dataX = np.array(imgs.view(len(imgs), -1))

    # Fake_MNIST
    Fake_MNIST = pickle.load(open('/data/gpfs/projects/punim2335/data/Fake_MNIST_data_EP100_N10000.pckl', 'rb'))
    dataY = torch.from_numpy(Fake_MNIST[0][:])
    dataY = np.array(dataY.view(len(dataY), -1))
    if check == 0:
        N1_T = dataX.shape[0]
        ind1 = np.random.choice(N1_T, N, replace=False)
        ind2 = np.random.choice(N1_T, N, replace=False)
        X = dataY[ind1, :]
        Y = dataY[ind2, :]
        # X = dataX[ind1, :]
        # Y = dataX[ind2, :]
    else:
        N1_T = dataX.shape[0]
        N2_T = dataY.shape[0]
        ind1 = np.random.choice(N1_T, N, replace=False)
        ind2 = np.random.choice(N2_T, N, replace=False)
        X = dataX[ind1, :]
        Y = dataY[ind2, :]

    if scale:
        X, Y = MMscaler(X, Y)
    
    # """transform to tensor"""
    # X = torch.from_numpy(X)
    # X = X.resize(len(X),1,img_size,img_size)
    return X, Y

def sample_CIFAR10(N, rs, check, scale):
    """#Feat. 32*32  #Class 2  #Inst. [10000,2021]"""
    np.random.seed(seed=rs)
    torch.manual_seed(seed=rs)
    channels = 3  # number of image channels

    img_size = 32  # size of each image dimension

    dataset_real_all = datasets.CIFAR10(root='/data/gpfs/projects/punim2335/data/cifar10', download=False, train=False,
                                        transform=transforms.Compose([
                                            transforms.Resize(img_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                            # transforms.Grayscale(num_output_channels=channels)
                                        ]))

    dataloader_real_all = DataLoader(dataset_real_all, batch_size=10000,
                                     shuffle=True)

    # Obtain CIFAR10 images
    for i, (imgs, labels) in enumerate(dataloader_real_all):
        data_real_all = imgs
        label_real_all = labels
    idx_real_all = np.arange(len(data_real_all))

    # Obtain CIFAR10.1 images
    # dataset_fake_all = np.load("/data/gpfs/projects/punim2335/data/ddpm_generated_images.npy").transpose(0,3,1,2)
    dataset_fake_all = np.load('/data/gpfs/projects/punim2335/data/cifar10.1_v4_data.npy').transpose(0,3,1,2)
    # dataset_fake_all = np.load('/data/gpfs/projects/punim2335/data/cifar10_X_adversarial.npy')

    # transpose to (samples, channels, height, width)
    idx_M = np.random.choice(len(dataset_fake_all),
                            len(dataset_fake_all), replace=False)
    dataset_fake_all = dataset_fake_all[idx_M]

    # preoprocessing fake dataset through the same transformation pipeline as real dataset
    fake_transfom = transforms.Compose([transforms.Resize(img_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        # transforms.Grayscale(num_output_channels=channels)
                                        ])
    trans = transforms.ToPILImage()

    data_fake_all = torch.zeros(
        [len(dataset_fake_all), channels, img_size, img_size])
    data_fake_tensor = torch.from_numpy(dataset_fake_all)

    for i in range(len(dataset_fake_all)):
        img_f = trans(data_fake_tensor[i])
        data_fake_all[i] = fake_transfom(img_f)
    idx_fake_all = np.arange(len(dataset_fake_all))
    # print(data_fake_all.shape)

    if check == 0:
        np.random.seed(seed=rs + 283)
        Ind = np.random.choice(len(data_fake_all), N, replace=False)
        s1 = data_fake_all[Ind]
        np.random.seed(seed=rs)
        Ind_v4 = np.random.choice(len(data_fake_all), N, replace=False)
        s2 = data_fake_all[Ind_v4]
    else:
        np.random.seed(seed=rs + 283)
        Ind = np.random.choice(len(data_real_all), N, replace=False)
        s1 = data_real_all[Ind]
        np.random.seed(seed=rs)
        Ind_v4 = np.random.choice(len(data_fake_all), N, replace=False)
        s2 = data_fake_all[Ind_v4]

    s1 = np.array(s1.view(len(s1), -1))
    s2 = np.array(s2.view(len(s2), -1))
    # print(s1.shape, s2.shape)

    if scale:
        s1, s2 = MMscaler(s1, s2)

    return s1, s2

transform = transforms.Compose([
    transforms.Resize(256),  # Resize the shortest side to 256 pixels
    transforms.CenterCrop(224),  # Crop the center to get a 224x224 image
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize using ImageNet mean and std
    transforms.Resize(32),  # Resize the shortest side to 256 pixels
    transforms.Grayscale(num_output_channels=3)
])

class ImageNetDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]

        if self.transform:
            try:
                image = self.transform(image)
            except:
                return self.__getitem__((idx + 1) % len(self.dataset))
                
        return image, label

def sample_ImageNetA(N, rs, check, scale):
    """#Feat. 4  #Class 2  #Inst. [5170877,5829123]"""
    np.random.seed(seed=rs)
    torch.manual_seed(rs)
    torch.cuda.manual_seed(rs)
    
    if check == 1:
        dataset_X = datasets.ImageFolder(root='/data/gpfs/datasets/Imagenet/ILSVRC/Data/CLS-LOC/val', transform=transform)
        X_dataloader = DataLoader(dataset_X, batch_size=N, shuffle=True, num_workers=0)
        for X_, labels in X_dataloader:
            X, labels = X_.detach(), labels.detach()
            break
        X = X.view(X.size(0), -1).cpu().numpy()
        
        dataset_Y = load_dataset("barkermrl/imagenet-a", split='train', cache_dir='/data/gpfs/projects/punim2335/data')
        dataset_Y = ImageNetDataset(dataset_Y, transform=transform)
        Y_dataloader = DataLoader(dataset_Y, batch_size=N, shuffle=True, num_workers=0)
        for Y_, labels in Y_dataloader:
            Y, labels = Y_.detach(), labels.detach()
            break
        Y = Y.view(Y.size(0), -1).cpu().numpy()
        
    if check == 0:
        dataset_Y = load_dataset("barkermrl/imagenet-a", split='train', cache_dir='/data/gpfs/projects/punim2335/data')
        dataset_Y = ImageNetDataset(dataset_Y, transform=transform)
        Y_dataloader = DataLoader(dataset_Y, batch_size=N, shuffle=True, num_workers=0)
        
        Flag = True
        for Y_, labels in Y_dataloader:
            if Flag:
                Y, labels = Y_.detach(), labels.detach()
                Flag = False
            else:
                X, labels = Y_.detach(), labels.detach()
                break
        X = X.view(X.size(0), -1).cpu().numpy()
        Y = Y.view(Y.size(0), -1).cpu().numpy()
    
    if scale:
        X, Y = MMscaler(X, Y)
    return X, Y

def sample_ImageNetV2(N, rs, check, scale):
    """#Feat. 4  #Class 2  #Inst. [5170877,5829123]"""
    np.random.seed(seed=rs)
    torch.manual_seed(rs)
    torch.cuda.manual_seed(rs)

    if check == 1:
        dataset_X = datasets.ImageFolder(root='/data/gpfs/datasets/Imagenet/ILSVRC/Data/CLS-LOC/val', transform=transform)
        X_dataloader = DataLoader(dataset_X, batch_size=N, shuffle=True, num_workers=0)
        for X_, labels in X_dataloader:
            X, labels = X_.detach(), labels.detach()
            break
        X = X.view(X.size(0), -1).cpu().numpy()
        
        dataset_Y = datasets.ImageFolder(root='/data/gpfs/projects/punim2335/data/imagenetv2', transform=transform)
        Y_dataloader = DataLoader(dataset_Y, batch_size=N, shuffle=True, num_workers=0)
        for Y_, labels in Y_dataloader:
            Y, labels = Y_.detach(), labels.detach()
            break
        Y = Y.view(Y.size(0), -1).cpu().numpy()
        
    if check == 0:
        dataset_Y = datasets.ImageFolder(root='/data/gpfs/datasets/Imagenet/ILSVRC/Data/CLS-LOC/val', transform=transform)
        X_dataloader = DataLoader(dataset_Y, batch_size=N, shuffle=True, num_workers=0)
        Y_dataloader = DataLoader(dataset_Y, batch_size=N, shuffle=True, num_workers=0)
        
        # Flag = True
        for Y_, labels in Y_dataloader:
            Y, labels = Y_.detach(), labels.detach()
            break
        for X_, labels in X_dataloader:
            X, labels = Y_.detach(), labels.detach()
            break
        X = X.view(X.size(0), -1).cpu().numpy()
        Y = Y.view(Y.size(0), -1).cpu().numpy()
    
    if scale:
        X, Y = MMscaler(X, Y)
    return X, Y

def sample_ImageNetSK(N, rs, check, scale):
    """#Feat. 4  #Class 2  #Inst. [5170877,5829123]"""
    np.random.seed(seed=rs)
    torch.manual_seed(rs)
    torch.cuda.manual_seed(rs)
    
    if check == 1:
        dataset_X = datasets.ImageFolder(root='/data/gpfs/datasets/Imagenet/ILSVRC/Data/CLS-LOC/val', transform=transform)
        X_dataloader = DataLoader(dataset_X, batch_size=N, shuffle=True, num_workers=0)
        for X_, labels in X_dataloader:
            X, labels = X_.detach(), labels.detach()
            break
        X = X.view(X.size(0), -1).cpu().numpy()
        
        dataset_Y = load_dataset("imagenet_sketch",  split='train', cache_dir='/data/gpfs/projects/punim2335/data')
        dataset_Y = ImageNetDataset(dataset_Y, transform=transform)
        Y_dataloader = DataLoader(dataset_Y, batch_size=N, shuffle=True, num_workers=0)
        for Y_, labels in Y_dataloader:
            Y, labels = Y_.detach(), labels.detach()
            break
        Y = Y.view(Y.size(0), -1).cpu().numpy()
        
    if check == 0:
        dataset_Y = load_dataset("imagenet_sketch",  split='train', cache_dir='/data/gpfs/projects/punim2335/data')
        dataset_Y = ImageNetDataset(dataset_Y, transform=transform)
        Y_dataloader = DataLoader(dataset_Y, batch_size=N, shuffle=True, num_workers=0)
        
        Flag = True
        for Y_, labels in Y_dataloader:
            if Flag:
                Y, labels = Y_.detach(), labels.detach()
                Flag = False
            else:
                X, labels = Y_.detach(), labels.detach()
                break
        X = X.view(X.size(0), -1).cpu().numpy()
        Y = Y.view(Y.size(0), -1).cpu().numpy()
    
    if scale:
        X, Y = MMscaler(X, Y)
    return X, Y

def sample_ImageNetR(N, rs, check, scale):
    """#Feat. 4  #Class 2  #Inst. [5170877,5829123]"""
    np.random.seed(seed=rs)
    torch.manual_seed(rs)
    torch.cuda.manual_seed(rs)
    
    if check == 1:
        dataset_X = datasets.ImageFolder(root='/data/gpfs/datasets/Imagenet/ILSVRC/Data/CLS-LOC/val', transform=transform)
        X_dataloader = DataLoader(dataset_X, batch_size=N, shuffle=True, num_workers=0)
        for X_, labels in X_dataloader:
            X, labels = X_.detach(), labels.detach()
            break
        X = X.view(X.size(0), -1).cpu().numpy()
        
        dataset_Y = load_dataset("axiong/imagenet-r",  split='test', cache_dir='/data/gpfs/projects/punim2335/data')
        with open('/data/gpfs/projects/punim2335/data/imagenet_class_index.json', 'r') as f:
            imagenet_class_index = json.load(f)
        wnid_label = defaultdict(int)
        for label in list(imagenet_class_index.keys()):
            wnid_label[imagenet_class_index[label][0]] = int(label)
        def add_label(example):
            example["label"] = wnid_label.get(example["wnid"], "Unknown")
            return example
        dataset_Y = dataset_Y.map(add_label)
        dataset_Y = dataset_Y.sort('label')
        dataset_Y = ImageNetDataset(dataset_Y, transform=transform)
        Y_dataloader = DataLoader(dataset_Y, batch_size=N, shuffle=True, num_workers=0)
        for Y_, labels in Y_dataloader:
            Y, labels = Y_.detach(), labels.detach()
            break
        Y = Y.view(Y.size(0), -1).cpu().numpy()
        
    if check == 0:
        dataset_Y = load_dataset("axiong/imagenet-r",  split='test', cache_dir='/data/gpfs/projects/punim2335/data')
        with open('/data/gpfs/projects/punim2335/data/imagenet_class_index.json', 'r') as f:
            imagenet_class_index = json.load(f)
        wnid_label = defaultdict(int)
        for label in list(imagenet_class_index.keys()):
            wnid_label[imagenet_class_index[label][0]] = int(label)
        def add_label(example):
            example["label"] = wnid_label.get(example["wnid"], "Unknown")
            return example
        dataset_Y = dataset_Y.map(add_label)
        dataset_Y = dataset_Y.sort('label')
        dataset_Y = ImageNetDataset(dataset_Y, transform=transform)
        Y_dataloader = DataLoader(dataset_Y, batch_size=N, shuffle=True, num_workers=0)
        
        Flag = True
        for Y_, labels in Y_dataloader:
            if Flag:
                Y, labels = Y_.detach(), labels.detach()
                Flag = False
            else:
                X, labels = Y_.detach(), labels.detach()
                break
        X = X.view(X.size(0), -1).cpu().numpy()
        Y = Y.view(Y.size(0), -1).cpu().numpy()
    
    if scale:
        X, Y = MMscaler(X, Y)
    return X, Y

def sample_ImageNetV2_Fea(N, rs, check, scale):
    """#Feat. 4  #Class 2  #Inst. [5170877,5829123]"""
    np.random.seed(seed=rs)
    torch.manual_seed(rs)
    torch.cuda.manual_seed(rs)
    
    X = torch.load('/data/gpfs/projects/punim2335/data/imagenet_Fea.pt')
    Y = torch.load('/data/gpfs/projects/punim2335/data/imagenetv2_Fea.pt')
    if check == 0:
        X = Y
    N1_T = X.shape[0]
    N2_T = Y.shape[0]
    ind1 = np.random.choice(N1_T, N, replace=False)
    ind2 = np.random.choice(N2_T, N, replace=False)
    X = X[ind1]
    Y = Y[ind2]
    X = X.view(X.size(0), -1).cpu().numpy()
    Y = Y.view(Y.size(0), -1).cpu().numpy()
    if scale:
        X, Y = MMscaler(X, Y)
    return X, Y

# load_data("ImageNetV2_Fea", 100, 1, check=1, scale=True)