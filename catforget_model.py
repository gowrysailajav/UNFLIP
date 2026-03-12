import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner, ShardPartitioner
from torchvision.transforms import ToTensor, Normalize, Compose
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model import train, test
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from torch.utils.data import Subset
from torch.utils.data import TensorDataset
import random
import pandas as pd
import os
from PIL import Image
import requests, zipfile, io
from torch import nn, optim
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.utils import save_image
from typing import List, Iterable, Dict, Optional
import os
import pickle

# -----------------
# Hyperparameters
# -----------------
batch_size = 16
image_size = 28
nz = 100       # Latent vector size
ngf = 64       # Generator feature maps
ndf = 64       # Discriminator feature maps
num_epochs = 30
lr = 0.0002
beta1 = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def get_kmnist(trainset_size: int = 60000, testset_size: int = 10000, data_path: str = './data'):
    """Loads and returns KMNIST dataset with subset sizes."""
    print("✅ LOADING KMNIST "*5)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = datasets.KMNIST(root=data_path, train=True, download=True, transform=transform)
    testset  = datasets.KMNIST(root=data_path, train=False, download=True, transform=transform)

    trainset = Subset(trainset, random.sample(range(len(trainset)), min(trainset_size, len(trainset))))
    testset  = Subset(testset, random.sample(range(len(testset)), min(testset_size, len(testset))))
    # # ✅ Sanity checks
    print(f"✅ Train dataset size: {len(trainset)}")
    print(f"✅ Test dataset size: {len(testset)}")
    return trainset, testset


def get_signmnist(trainset_size: int = 27455, testset_size: int = 7172, data_path: str = '/home/sailaja/OneDrive/Code_Repository/paper2'):
    """Loads Sign Language MNIST dataset from CSV, filtered to only include classes 0–9."""
    print("✅ LOADING signMINIST "*5)
    def load_signmnist_file(filepath):
        df = pd.read_csv(filepath)
        # Filter rows where label is between 0 and 9 (inclusive)
        df = df[df.iloc[:, 0].between(0, 9)]
        labels = torch.tensor(df.iloc[:, 0].values, dtype=torch.long)
        images = torch.tensor(df.iloc[:, 1:].values, dtype=torch.float32).reshape(-1, 1, 28, 28) / 255.0
        return TensorDataset(images, labels)

    trainset = load_signmnist_file(f'{data_path}/sign_mnist_train.csv')
    testset = load_signmnist_file(f'{data_path}/sign_mnist_test.csv')
    
    trainset = Subset(trainset, random.sample(range(len(trainset)), min(trainset_size, len(trainset))))
    testset  = Subset(testset, random.sample(range(len(testset)), min(testset_size, len(testset))))
    # # ✅ Sanity checks
    print(f"✅ Train dataset size: {len(trainset)}")
    print(f"✅ Test dataset size: {len(testset)}")
    return trainset, testset


def get_random_mnist_like(trainset_size: int = 60000, testset_size: int = 10000, data_path: str = './data'):
    """Creates random data with the same shape and label range as MNIST."""
    print("✅ LOADING random "*5)
    train_data = torch.randn(trainset_size, 1, 28, 28)
    train_labels = torch.randint(0, 10, (trainset_size,))
    test_data = torch.randn(testset_size, 1, 28, 28)
    test_labels = torch.randint(0, 10, (testset_size,))
    return TensorDataset(train_data, train_labels), TensorDataset(test_data, test_labels)


def get_emnist(trainset_size=None, testset_size=None, root="./data/EMNIST"):
    print("✅ LOADING EMNIST " * 5)

    transform = transforms.Compose([
        transforms.Resize((28, 28)),   # Normalize to 28x28
        transforms.Normalize((0.5,), (0.5,))  # Scale to [-1, 1]
    ])

    # Load CSV datasets
    train_dataset = AHCDCsvDataset(
        images_file=f"{root}/csvTrainImages.csv",
        labels_file=f"{root}/csvTrainLabel.csv",
        transform=transform
    )
    test_dataset = AHCDCsvDataset(
        images_file=f"{root}/csvTestImages.csv",
        labels_file=f"{root}/csvTestLabel.csv",
        transform=transform
    )

    # Subsample if requested
    if trainset_size is not None:
        train_dataset, _ = random_split(train_dataset, [trainset_size, len(train_dataset) - trainset_size])
    if testset_size is not None:
        test_dataset, _ = random_split(test_dataset, [testset_size, len(test_dataset) - testset_size])

    # ✅ Sanity checks
    print(f"✅ Train dataset size: {len(train_dataset)}")
    print(f"✅ Test dataset size: {len(test_dataset)}")
    return train_dataset, test_dataset


def get_mnist(trainset_size: int = 60000, testset_size: int = 10000, data_path: str = './data'):
    """Loads and returns MNIST dataset with subset sizes."""
    print("✅ LOADING MNIST "*5)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    testset  = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)

    # Subsample
    trainset = Subset(trainset, random.sample(range(len(trainset)), min(trainset_size, len(trainset))))
    testset  = Subset(testset, random.sample(range(len(testset)), min(testset_size, len(testset))))
    # # ✅ Sanity checks
    print(f"✅ Train dataset size: {len(trainset)}")
    print(f"✅ Test dataset size: {len(testset)}")
    return trainset, testset


def get_fashion_mnist(trainset_size: int = 60000, testset_size: int = 10000, data_path: str = './data'):
    """Loads and returns FashionMNIST dataset with subset sizes."""
    print("✅ LOADING FMNIST "*5)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = datasets.FashionMNIST(root=data_path, train=True, download=True, transform=transform)
    testset  = datasets.FashionMNIST(root=data_path, train=False, download=True, transform=transform)

    trainset = Subset(trainset, random.sample(range(len(trainset)), min(trainset_size, len(trainset))))
    testset  = Subset(testset, random.sample(range(len(testset)), min(testset_size, len(testset))))
    # # ✅ Sanity checks
    print(f"✅ Train dataset size: {len(trainset)}")
    print(f"✅ Test dataset size: {len(testset)}")
    return trainset, testset


# ---------------------------
# QMNIST (Extended MNIST)
# ---------------------------
def get_qmnist(trainset_size=None, testset_size=None, root="./data/QMNIST"):
    print("✅ LOADING QMNIST "*5)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.QMNIST(root=root, what='train', download=True, transform=transform)
    testset = torchvision.datasets.QMNIST(root=root, what='test', download=True, transform=transform)

    # Subsample if sizes are given
    if trainset_size is not None:
        trainset, _ = random_split(trainset, [trainset_size, len(trainset) - trainset_size])
    if testset_size is not None:
        testset, _ = random_split(testset, [testset_size, len(testset) - testset_size])
    # # ✅ Sanity checks
    print(f"✅ Train dataset size: {len(trainset)}")
    print(f"✅ Test dataset size: {len(testset)}")
    return trainset, testset

def get_cifar10(trainset_size: int = 50000, testset_size: int = 10000, data_path: str = './data'):
    """Loads and returns CIFAR10 dataset with subset sizes."""
    print("✅ LOADING CIFAR10 " * 5)

    # Standard CIFAR10 normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), 
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    trainset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    testset  = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)

    # Subsample train and test exactly like MNIST version
    trainset = Subset(trainset, random.sample(range(len(trainset)), min(trainset_size, len(trainset))))
    testset  = Subset(testset, random.sample(range(len(testset)), min(testset_size, len(testset))))

    print(f"✅ Train dataset size: {len(trainset)}")
    print(f"✅ Test dataset size: {len(testset)}")

    return trainset, testset

def get_svhn(trainset_size: int = 73257, testset_size: int = 26032, data_path: str = './data'):
    """Loads and returns SVHN dataset (32×32 RGB) with subset sizes."""
    print("✅ LOADING SVHN " * 5)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4377, 0.4438, 0.4728),
            std=(0.1980, 0.2010, 0.1970)
        )
    ])

    trainset = datasets.SVHN(root=data_path, split='train', download=True, transform=transform)
    testset  = datasets.SVHN(root=data_path, split='test', download=True, transform=transform)

    trainset = Subset(trainset, random.sample(range(len(trainset)), min(trainset_size, len(trainset))))
    testset  = Subset(testset, random.sample(range(len(testset)), min(testset_size, len(testset))))

    print(f"✅ SVHN Train size: {len(trainset)}")
    print(f"✅ SVHN Test size: {len(testset)}")

    return trainset, testset

def get_gtsrb32(trainset_size: int = 39209, testset_size: int = 12630, data_path: str = './data/GTSRB-32'):
    """Loads and returns GTSRB (32×32 RGB) with subset sizes."""
    print("✅ LOADING GTSRB-32 " * 5)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.3403, 0.3121, 0.3214), 
            std=(0.2724, 0.2608, 0.2669)
        )
    ])

    train_dir = os.path.join(data_path, "train")
    test_dir  = os.path.join(data_path, "test")

    trainset = datasets.ImageFolder(root=train_dir, transform=transform)
    testset  = datasets.ImageFolder(root=test_dir, transform=transform)

    trainset = Subset(trainset, random.sample(range(len(trainset)), min(trainset_size, len(trainset))))
    testset  = Subset(testset, random.sample(range(len(testset)), min(testset_size, len(testset))))

    print(f"✅ GTSRB-32 Train size: {len(trainset)}")
    print(f"✅ GTSRB-32 Test size: {len(testset)}")

    return trainset, testset

def get_tinyimagenet32(trainset_size: int = 100000, testset_size: int = 10000, data_path: str = './data/TinyImageNet32'):
    """Loads and returns Tiny-ImageNet downsampled to 32×32 RGB with subset sizes."""
    print("✅ LOADING Tiny-ImageNet32 " * 5)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4802, 0.4481, 0.3975),
            std =(0.2302, 0.2265, 0.2262)
        )
    ])

    train_dir = os.path.join(data_path, "train")
    val_dir   = os.path.join(data_path, "val")

    trainset = datasets.ImageFolder(root=train_dir, transform=transform)
    testset  = datasets.ImageFolder(root=val_dir, transform=transform)

    trainset = Subset(trainset, random.sample(range(len(trainset)), min(trainset_size, len(trainset))))
    testset  = Subset(testset, random.sample(range(len(testset)), min(testset_size, len(testset))))

    print(f"✅ Tiny-ImageNet32 Train size: {len(trainset)}")
    print(f"✅ Tiny-ImageNet32 Test size: {len(testset)}")

    return trainset, testset


# ---------------------------
# Arabic Handwritten Digits Dataset (AHCD)
# (https://www.kaggle.com/datasets/mloey1/ahcd1)
# ---------------------------


class AHCDCsvDataset(Dataset):
    def __init__(self, images_file, labels_file, transform=None):
        self.images = pd.read_csv(images_file, header=None).values
        self.labels = pd.read_csv(labels_file, header=None).values.flatten()
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Convert row to 28x28 image
        img = self.images[idx].reshape(32, 32)  # AHCD is usually 32×32
        img = img.astype(np.uint8)

        # Convert to PIL image before transform
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1, 32, 32)

        if self.transform:
            img = self.transform(img)

        label = int(self.labels[idx])
        return img, label


def get_ahcd(trainset_size=None, testset_size=None, root="./data/AHCD"):
    print("✅ LOADING AHCD " * 5)

    transform = transforms.Compose([
        transforms.Resize((28, 28)),   # Normalize to 28x28
        transforms.Normalize((0.5,), (0.5,))  # Scale to [-1, 1]
    ])

    # Load CSV datasets
    train_dataset = AHCDCsvDataset(
        images_file=f"{root}/csvTrainImages.csv",
        labels_file=f"{root}/csvTrainLabel.csv",
        transform=transform
    )
    test_dataset = AHCDCsvDataset(
        images_file=f"{root}/csvTestImages.csv",
        labels_file=f"{root}/csvTestLabel.csv",
        transform=transform
    )

    # Subsample if requested
    if trainset_size is not None:
        train_dataset, _ = random_split(train_dataset, [trainset_size, len(train_dataset) - trainset_size])
    if testset_size is not None:
        test_dataset, _ = random_split(test_dataset, [testset_size, len(test_dataset) - testset_size])

    # ✅ Sanity checks
    print(f"✅ Train dataset size: {len(train_dataset)}")
    print(f"✅ Test dataset size: {len(test_dataset)}")
    return train_dataset, test_dataset


def prepare_forget_dataset(dataset_name: str, trainset_size, testset_size, partition_type: str, batch_size: int, val_ratio: float = 0.1):
    print(f'dataset_name : {dataset_name}')
    # Load dataset
    
    if dataset_name.lower() == "mnist":
        trainset, tareget_testset = get_mnist(trainset_size=trainset_size, testset_size=testset_size)
    elif dataset_name.lower() in ["fmnist", "fashion_mnist"]:
        trainset, tareget_testset = get_fashion_mnist(trainset_size=trainset_size, testset_size=testset_size)
    elif dataset_name.lower() == "kmnist":
        trainset, tareget_testset = get_kmnist(trainset_size=trainset_size, testset_size=testset_size)
    elif dataset_name.lower() == "signmnist":
        trainset, tareget_testset = get_signmnist(trainset_size=trainset_size, testset_size=testset_size)
    elif dataset_name.lower() == "qmnist":
        trainset, tareget_testset = get_qmnist(trainset_size=trainset_size, testset_size=testset_size)
    elif dataset_name.lower() == "cifar10":
        trainset, tareget_testset = get_cifar10(trainset_size=trainset_size, testset_size=testset_size)
    elif dataset_name.lower() == "svhn":
        trainset, tareget_testset = get_svhn(trainset_size=trainset_size, testset_size=testset_size)
    elif dataset_name.lower() == "gtsrb32":
        trainset, tareget_testset = get_gtsrb32(trainset_size=trainset_size, testset_size=testset_size)
    elif dataset_name.lower() == "tinyimagenet32":
        trainset, tareget_testset = get_tinyimagenet32(trainset_size=trainset_size, testset_size=testset_size)                
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Split into train and val
    num_total = len(trainset)
    num_val = int(val_ratio * num_total)
    num_train = num_total - num_val

    for_train, for_val = random_split(trainset, [num_train, num_val], torch.Generator().manual_seed(2023))

    # Collate function
    def collate_fn(batch):
        images, labels = [], []
        for item in batch:
            image = item["image"] if isinstance(item, dict) else item[0]
            label = item["label"] if isinstance(item, dict) else item[1]

            # Convert numpy → PIL → Tensor
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            if not isinstance(image, torch.Tensor):
                image = transforms.ToTensor()(image)

            # Ensure shape [C, H, W], not batched already
            if image.dim() == 4:   # e.g., [B, C, H, W]
                image = image[0]   # take first if redundant batch
            images.append(image)
            labels.append(label)

        return torch.stack(images), torch.tensor(labels)

    # Dataloaders
    trainloader = DataLoader(for_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    target_testloader = DataLoader(tareget_testset, batch_size=batch_size, shuffle=False)
    return trainloader, target_testloader


def compute_bfr_from_accuracies(
    acc_ft_list: Iterable[float],
    acc_ft1_list: Iterable[float],
    num_classes: int
) -> Dict[str, float]:
    """
    Compute BFR when you already measured acc(ft, D̃t) and acc(ft+1, D̃t).

    Args:
      acc_ft_list: [acc(f1,D̃1), acc(f2,D̃2), ..., acc(f_{T-1},D̃_{T-1})]
      acc_ft1_list: [acc(f2,D̃1), acc(f3,D̃2), ..., acc(f_T,D̃_{T-1})]
      num_classes: n(Y_t)

    Returns:
      Same dict as above.
    """
    eps = 1e-12
    acc_ft_list = list(acc_ft_list)
    acc_ft1_list = list(acc_ft1_list)
    assert len(acc_ft_list) == len(acc_ft1_list) and len(acc_ft_list) >= 1

    raw_terms, terms = [], []
    for t, (acc_ft, acc_ft1) in enumerate(zip(acc_ft_list, acc_ft1_list), start=1):
        num = (acc_ft - acc_ft1)
        # den = max(acc_ft - 1.0/num_classes, eps)
        den = acc_ft - (1/num_classes)
        if abs(den) < eps:  
            den = eps
        term = num / den
        raw_terms.append({"t": t, "acc_ft": acc_ft, "acc_ft1": acc_ft1,
                          "num": num, "den": den, "term": term})
        terms.append(term)

    bfr = sum(terms) / len(terms)
    return {"BFR": bfr, "per_task": bfr, "raw_terms": raw_terms}




def cached_prepare_forget_dataset(dataset_name,
                                  trainset_size, testset_size,
                                  partition_type, batch_size, CACHE_PATH,
                                  val_ratio=0.1):
    """
    Cache only DATASETS (not DataLoaders)
    Recreate loaders after loading.
    """
    print(f'trainset_size: {trainset_size} | testset_size: {testset_size} ')
    # ------------------------------
    # 1. If cache exists, load it
    # ------------------------------
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached dataset from {CACHE_PATH}")

        with open(CACHE_PATH, "rb") as f:
            data = pickle.load(f)

        trainset = data["trainset"]
        target_testset = data["target_testset"]

    else:
        print(f"No cache. Creating dataset and saving to {CACHE_PATH}")

        # Use your existing function (DO NOT MODIFY)
        trainloader, target_testloader = prepare_forget_dataset(
            dataset_name, trainset_size, testset_size,
            partition_type, batch_size, val_ratio
        )

        # Extract datasets only (LOADERS CANNOT BE PICKLED)
        trainset = trainloader.dataset
        target_testset = target_testloader.dataset

        # Save datasets only
        data = {
            "trainset": trainset,
            "target_testset": target_testset
        }

        with open(CACHE_PATH, "wb") as f:
            pickle.dump(data, f)

        print("Dataset cached successfully.")

    # ✔ Using default collate_fn (safe & picklable)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    target_testloader = DataLoader(target_testset, batch_size=batch_size, shuffle=False)

    return trainloader, target_testloader


from collections import Counter
import numpy as np
import torch


def check_iid_distribution(dataloader, num_classes=None, tol=0.05):
    """
    tol = allowable deviation from uniform (5% default)
    """

    labels = []

    for _, y in dataloader:
        if torch.is_tensor(y):
            y = y.cpu().numpy()
        labels.extend(y.tolist())

    counts = Counter(labels)

    if num_classes is None:
        num_classes = len(counts)

    total = sum(counts.values())
    freqs = {k: v / total for k, v in counts.items()}

    ideal = 1.0 / num_classes
    deviations = {k: abs(freqs[k] - ideal) for k in freqs}

    max_dev = max(deviations.values())

    print("\nShadow dataset class distribution:")
    for c in sorted(freqs):
        print(f"Class {c}: {counts[c]} ({freqs[c]:.4f})")

    print(f"\nIdeal frequency: {ideal:.4f}")
    print(f"Max deviation: {max_dev:.4f}")

    is_iid = max_dev <= tol

    if is_iid:
        print("✅ Dataset is approximately IID")
    else:
        print("⚠️ Dataset NOT IID!")

    return is_iid, freqs

def create_shadow_model_parameters(trainloader, model, lr, momentum, epochs, device, SHADOW_CACHE_PATH):
     # --------------------------------------------------
    # Load cached shadow parameters if available
    # --------------------------------------------------
    if os.path.exists(SHADOW_CACHE_PATH):
        print(f"Loading cached shadow model params from {SHADOW_CACHE_PATH}")

        with open(SHADOW_CACHE_PATH, "rb") as f:
            shadow_model_params = pickle.load(f)

        return shadow_model_params

    # --------------------------------------------------
    # Otherwise train ONCE and cache
    # --------------------------------------------------
    print("No shadow cache found. Training shadow model once...")
    model.to(device)  
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    train_loss, y_true, y_pred, avg_grads = train(model, trainloader, optim, epochs, device)
    shadow_model_params = [val.detach().cpu().numpy().astype(np.float32) for _, val in model.state_dict().items()]
    # --------------------------------------------------
    # Save for future reuse
    # --------------------------------------------------
    with open(SHADOW_CACHE_PATH, "wb") as f:
        pickle.dump(shadow_model_params, f)

    print(f"Shadow parameters cached at {SHADOW_CACHE_PATH}")
    return shadow_model_params

     
def create_d2_parameters(state_dict, dataset_name, aux_dataset_name, 
                         trainset_size, testset_size, partition_type, batch_size, 
                         model, lr, momentum, epochs, device, CACHE_PATH):
    # Load previous parameters
    model.load_state_dict(state_dict, strict=False)
    model.to(device)  

    # Prepare dataset loaders
    trainloader, valloader, testloader, target_testloader = cached_prepare_forget_dataset(
        dataset_name, aux_dataset_name,
        trainset_size, testset_size,
        partition_type, batch_size,
        model, lr, momentum, epochs, device, CACHE_PATH
    )
    
    
    print(f'lr = {lr} | momentum = {momentum} | epochs = {epochs}')
    
    # Optimizer
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # -------------------------
    # Evaluate BEFORE training
    # -------------------------
    acc_step1_taskA = evaluate_model(model, testloader, device)
    acc_step1_taskB = evaluate_model(model, target_testloader, device)

    # -------------------------
    # Train on Task B
    # -------------------------
    train_loss, y_true, y_pred = train(model, trainloader, optim, epochs, device)

    # -------------------------
    # Evaluate AFTER training
    # -------------------------
    acc_step2_taskA = evaluate_model(model, testloader, device)
    acc_step2_taskB = evaluate_model(model, target_testloader, device)

    # calculate BFR values
    acc_ft_list = [acc_step1_taskA]
    acc_ft1_list = [acc_step2_taskA]
    num_classes = 10 
    results = compute_bfr_from_accuracies(acc_ft_list, acc_ft1_list, num_classes)
    print(results)
    
    # Convert parameters for return
    d2_parameters = [val.detach().cpu().numpy().astype(np.float32) for _, val in model.state_dict().items()]


    # Classification metrics (on training predictions)
    # accuracy = accuracy_score(y_true, y_pred)
    # precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    # recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    # f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)


    # Collect metrics
    metrics = {
        "train_loss": train_loss,
        # "accuracy_train": accuracy,
        # "precision_train": precision,
        # "recall_train": recall,
        # "f1_score_train": f1,
        "acc_step1_taskA": acc_step1_taskA,
        "acc_step1_taskB": acc_step1_taskB,
        "acc_step2_taskA": acc_step2_taskA,
        "acc_step2_taskB": acc_step2_taskB,
        "BFR": results,
    }

    # print(f"METRICS: {metrics}")
    return d2_parameters, metrics, target_testloader



def evaluate_model(model, dataloader, device):
    """Evaluate model accuracy on given dataloader (e.g., MNIST)."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:   # MNIST returns (image, label)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)         # logits: [batch_size, 10]
            preds = outputs.argmax(dim=1)   # predicted class
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


