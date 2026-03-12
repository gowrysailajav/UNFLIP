import torch
from torch.utils.data import DataLoader, random_split, Subset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner, ShardPartitioner
from torchvision import transforms
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
import random
from collections import Counter
from torchvision import datasets, transforms
from flwr_datasets import FederatedDataset
import math
from datasets import Dataset as HFDataset 
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms


# ---------------------------
# Manual partitioners
# ---------------------------

def iid_partitions(dataset, num_partitions, seed=42):
    rng = torch.Generator().manual_seed(seed)
    base = len(dataset) // num_partitions
    sizes = [base] * num_partitions
    for i in range(len(dataset) - base * num_partitions):
        sizes[i] += 1
    subsets = list(random_split(dataset, sizes, generator=rng))
    return subsets

def dirichlet_partitions(dataset, num_partitions, alpha=0.5, seed=42, get_label_fn=None):
    """Label-aware non-iid split using Dirichlet distribution."""
    if get_label_fn is None:
        # Works for torchvision datasets and TensorDataset
        def get_label_fn(idx):
            item = dataset[idx]
            # item = (img, label) or dict-like; handle both
            return int(item[1].item() if isinstance(item[1], torch.Tensor) else item[1])

    rng = np.random.default_rng(seed)
    labels = np.array([get_label_fn(i) for i in range(len(dataset))])
    classes = np.unique(labels)

    idx_per_client = [[] for _ in range(num_partitions)]
    for c in classes:
        c_idx = np.where(labels == c)[0]
        rng.shuffle(c_idx)
        # proportions for this class across clients
        props = rng.dirichlet(alpha=np.ones(num_partitions) * alpha)
        # turn proportions into integer splits
        splits = (props * len(c_idx)).astype(int)
        # fix rounding to match exact count
        while splits.sum() < len(c_idx):
            splits[rng.integers(0, num_partitions)] += 1
        while splits.sum() > len(c_idx):
            j = rng.integers(0, num_partitions)
            if splits[j] > 0:
                splits[j] -= 1
        # assign
        start = 0
        for k, s in enumerate(splits):
            if s > 0:
                idx_per_client[k].extend(c_idx[start:start+s])
                start += s

    subsets = [Subset(dataset, idxs) for idxs in idx_per_client]
    return subsets

def shard_partitions(dataset, num_partitions, shard_size=200, seed=42, get_label_fn=None):
    """Classic shard-based split: sort by label, split into shards, assign shards to clients."""
    print(f'shard_size = {shard_size}')
    if get_label_fn is None:
        def get_label_fn(idx):
            item = dataset[idx]
            return int(item[1].item() if isinstance(item[1], torch.Tensor) else item[1])

    # sort indices by label
    idxs = list(range(len(dataset)))
    idxs.sort(key=lambda i: get_label_fn(i))

    # build shards
    shards = [idxs[i:i+shard_size] for i in range(0, len(idxs), shard_size)]
    rng = random.Random(seed)
    rng.shuffle(shards)

    # assign shards round-robin
    per_client = [[] for _ in range(num_partitions)]
    for i, sh in enumerate(shards):
        per_client[i % num_partitions].extend(sh)

    subsets = [Subset(dataset, idxs) for idxs in per_client]
    return subsets


def make_partitions(dataset, partition_type, num_partitions, alpha=0.5, shard_size=200, seed=42):
    if partition_type == "iid":
        return iid_partitions(dataset, num_partitions, seed=seed)
    elif partition_type == "dirichlet":
        return dirichlet_partitions(dataset, num_partitions, alpha=alpha, seed=seed)
    elif partition_type == "shard":
        return shard_partitions(dataset, num_partitions, shard_size=shard_size, seed=seed)
    else:
        raise ValueError(f"Unsupported partition type: {partition_type}")


def get_fashion_mnist(data_path: str = "./data"):
    """Loads and returns Fashion-MNIST dataset (28×28 grayscale)."""
    print("✅ LOADING FASHION-MNIST " * 3)

    # Standard normalization for Fashion-MNIST
    fmnist_mean = (0.2860,)
    fmnist_std = (0.3530,)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(fmnist_mean, fmnist_std),
    ])

    trainset = datasets.FashionMNIST(
        root=data_path,
        train=True,
        download=True,
        transform=transform
    )

    testset = datasets.FashionMNIST(
        root=data_path,
        train=False,
        download=True,
        transform=transform
    )

    print(f"✅ Fashion-MNIST Train: {len(trainset)}, Test: {len(testset)}")
    return trainset, testset



def get_mnist(data_path="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    trainset = datasets.MNIST(root=data_path, train=True, download=False, transform=transform)
    testset  = datasets.MNIST(root=data_path, train=False, download=False, transform=transform)
    
    return trainset, testset


def get_kmnist(data_path: str = './data'):
    """Loads and returns KMNIST dataset with subset sizes."""
    print("✅ LOADING KMNIST "*5)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = datasets.KMNIST(root=data_path, train=True, download=True, transform=transform)
    testset  = datasets.KMNIST(root=data_path, train=False, download=True, transform=transform)

    # # ✅ Sanity checks
    print(f"✅ Train dataset size: {len(trainset)}")
    print(f"✅ Test dataset size: {len(testset)}")
    return trainset, testset


# ---------------------------
# QMNIST (Extended MNIST)
# ---------------------------
def get_qmnist(root="./data"):
    print("✅ LOADING QMNIST "*5)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.QMNIST(root=root, what='train', download=True, transform=transform)
    testset = torchvision.datasets.QMNIST(root=root, what='test10k', download=True, transform=transform)

    # ✅ Sanity checks
    print(f"✅ Train dataset size: {len(trainset)}")
    print(f"✅ Test dataset size: {len(testset)}")
    return trainset, testset


def get_cifar10(data_path: str = './data'):
    """Loads and returns CIFAR-10 dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
    
    return trainset, testset


def get_svhn(data_path: str = './data'):
    """Loads and returns SVHN dataset (32×32 RGB)."""
    print("✅ LOADING SVHN " * 3)

    svhn_mean = (0.4377, 0.4438, 0.4728)
    svhn_std  = (0.1980, 0.2010, 0.1970)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(svhn_mean, svhn_std),
    ])

    trainset = datasets.SVHN(root=data_path, split='train', download=True, transform=transform)
    testset  = datasets.SVHN(root=data_path, split='test',  download=True, transform=transform)

    print(f"✅ SVHN Train: {len(trainset)}, Test: {len(testset)}")
    return trainset, testset


# ---------- Main prepare_dataset ----------

def prepare_dataset(
    mal_clients: int,
    attack_type: str,
    dataset_name: str,
    partition_type: str,
    num_partitions: int,
    batch_size: int,
    val_ratio: float,
    attack_fraction: float,
    attack_mode: str = "untargeted",
):
    dirichlet_alpha = 0.5
    shard_size = 200
    seed = 42

    # --- Dataset loading ---
    print(f"Loading dataset: {dataset_name} with partition type: {partition_type}")
    
    if dataset_name == "fashion_mnist":
        trainset, testset = get_fashion_mnist()
        train_partitions = make_partitions(trainset, partition_type, num_partitions,
                                           alpha=dirichlet_alpha, shard_size=shard_size, seed=seed)
        test_partition = testset
        print(f"FMNIST dataset loaded with {len(trainset)} train samples and {len(testset)} test samples")
        
    elif dataset_name == "mnist":
        trainset, testset = get_mnist()
        train_partitions = make_partitions(trainset, partition_type, num_partitions,
                                           alpha=dirichlet_alpha, shard_size=shard_size, seed=seed)
        test_partition = testset
        print(f"MNIST dataset loaded with {len(trainset)} train samples and {len(testset)} test samples")

    elif dataset_name == "kmnist":
        trainset, testset = get_kmnist()
        train_partitions = make_partitions(trainset, partition_type, num_partitions,
                                           alpha=dirichlet_alpha, shard_size=shard_size, seed=seed)
        test_partition = testset
        print(f"KMNIST dataset loaded with {len(trainset)} train samples and {len(testset)} test samples")


    elif dataset_name == "qmnist":
        trainset, testset = get_qmnist()
        train_partitions = make_partitions(trainset, partition_type, num_partitions,
                                           alpha=dirichlet_alpha, shard_size=shard_size, seed=seed)
        test_partition = testset
        print(f"QMNIST dataset loaded with {len(trainset)} train samples and {len(testset)} test samples")

    elif dataset_name == "cifar10":
        trainset, testset = get_cifar10()
        train_partitions = make_partitions(trainset, partition_type, num_partitions,
                                           alpha=dirichlet_alpha, shard_size=shard_size, seed=seed)
        test_partition = testset
        print(f"CIFAR10 dataset loaded with {len(trainset)} train samples and {len(testset)} test samples")
       
    elif dataset_name == "svhn":
        trainset, testset = get_svhn()
        train_partitions = make_partitions(trainset, partition_type, num_partitions,
                                           alpha=dirichlet_alpha, shard_size=shard_size, seed=seed)
        test_partition = testset
        print(f"SVHN dataset loaded with {len(trainset)} train samples and {len(testset)} test samples")
         
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    print("Dataset partitions created successfully.")

    def apply_label_flipping_svhn(
        dataset,
        attack_fraction=0.5,
        num_classes=10,
        attack_mode="untargeted",
    ):
        """
        Apply in-place label flipping for SVHN.
        attack_mode:
            - "untargeted": cyclic flip
            - "targeted": 3 -> 5
        """

        if isinstance(dataset, torch.utils.data.Subset):
            base_ds = dataset.dataset
            indices = dataset.indices
            labels = base_ds.labels
        else:
            base_ds = dataset
            indices = list(range(len(dataset)))
            labels = dataset.labels

        labels_list = labels.tolist() if hasattr(labels, "tolist") else list(labels)

        n_total = len(indices)
        n_to_flip = int(n_total * attack_fraction)
        flip_indices = random.sample(indices, n_to_flip)

        print(
            f"SVHN | {attack_type} label flipping | "
            f"{n_to_flip}/{n_total} samples"
        )

        for idx in flip_indices:
            old_label = int(labels_list[idx])

            if attack_mode == "targeted":
                # print(f'attack type targeted activated')
                if old_label == 3:
                    labels_list[idx] = 5
            else:  # untargeted
                # print(f'attack type untargeted activated')
                labels_list[idx] = (old_label + 1) % num_classes

        base_ds.labels = labels_list
        return dataset




    def apply_label_flipping_cifar10(
        dataset,
        attack_fraction=0.5,
        num_classes=10,
        attack_mode="untargeted",
    ):
        """
        Apply in-place label flipping for CIFAR-10.
        attack_mode:
            - "untargeted": cyclic flip
            - "targeted": 3 -> 5
        """

        if isinstance(dataset, torch.utils.data.Subset):
            base_ds = dataset.dataset
            indices = dataset.indices
            targets = base_ds.targets
        else:
            base_ds = dataset
            indices = list(range(len(dataset)))
            targets = dataset.targets

        targets_list = targets.tolist() if torch.is_tensor(targets) else list(targets)

        n_total = len(indices)
        n_to_flip = int(n_total * attack_fraction)
        flip_indices = random.sample(indices, n_to_flip)

        print(
            f"CIFAR10 | {attack_type} label flipping | "
            f"{n_to_flip}/{n_total} samples"
        )

        for idx in flip_indices:
            old_label = targets_list[idx]

            if attack_mode == "targeted":
                # print(f'attack type targeted activated')
                if old_label == 3:
                    targets_list[idx] = 5
            else:
                # print(f'attack type untargeted activated')
                targets_list[idx] = (old_label + 1) % num_classes

        base_ds.targets = targets_list
        return dataset




    def apply_label_flipping_kmnist(
        dataset,
        attack_fraction=0.5,
        num_classes=10,
        attack_mode="untargeted",
    ):
        """
        Apply in-place label flipping for KMNIST.
        attack_mode:
            - "untargeted": cyclic flip
            - "targeted": 3 -> 5
        """

        if isinstance(dataset, torch.utils.data.Subset):
            base_ds = dataset.dataset
            indices = dataset.indices
            targets = base_ds.targets
        else:
            base_ds = dataset
            indices = list(range(len(dataset)))
            targets = dataset.targets

        targets_list = targets.tolist() if torch.is_tensor(targets) else list(targets)

        n_total = len(indices)
        n_to_flip = int(n_total * attack_fraction)
        flip_indices = random.sample(indices, n_to_flip)

        print(
            f"{attack_type} label flipping | "
            f"{n_to_flip}/{n_total} samples"
        )

        for idx in flip_indices:
            old_label = targets_list[idx]

            if attack_mode == "targeted":
                # print(f'attack type targeted activated')
                if old_label == 3:
                    targets_list[idx] = 5
            else:
                # print(f'attack type untargeted activated')
                targets_list[idx] = (old_label + 1) % num_classes

        base_ds.targets = torch.tensor(targets_list, dtype=torch.long)
        return dataset


    
    
    def apply_label_flipping_direct(ds, attack_fraction=0.5):
        """Flip labels on the dataset directly (in-place replacement)."""
        n_samples = len(ds)
        n_to_flip = int(n_samples * attack_fraction)

        # Randomly choose which indices to flip
        flip_indices = random.sample(range(n_samples), n_to_flip)

        # Perform the flipping using map
        def flip_fn(example, idx=None):
            if idx in flip_indices:
                example["label"] = (example["label"] + 1) % 10
            return example

        # Important: use with_indices=True
        ds = ds.map(flip_fn, with_indices=True)

        return ds





    def apply_data_poisoning(dataset, frac):
        transform = transforms.ToTensor()
        n = int(len(dataset) * frac)

        # HuggingFace dataset
        if isinstance(dataset, HFDataset):
            for idx in random.sample(range(len(dataset)), n):
                example = dataset[idx]
                image = example["image"]

                # convert to tensor
                if isinstance(image, torch.Tensor):
                    image = image.float() / 255.0 if image.max() > 1 else image
                elif isinstance(image, np.ndarray):
                    image = transform(image)
                elif isinstance(image, Image.Image):
                    image = transform(image)

                # add noise
                poisoned = torch.clamp(image + torch.randn_like(image) * 0.1, 0, 1)

                # assign back
                dataset[idx]["image"] = poisoned
            return

        # Torchvision Dataset or Subset
        if hasattr(dataset, "indices"):  
            subset_indices = dataset.indices
            data = dataset.dataset.data
        else:
            subset_indices = list(range(len(dataset)))
            data = dataset.data  

        n_to_poison = int(len(subset_indices) * frac)
        poison_indices = random.sample(subset_indices, n_to_poison)

        for idx in poison_indices:
            image = data[idx]

            # convert to tensor
            if isinstance(image, torch.Tensor):
                image = image.float() / 255.0 if image.max() > 1 else image
            elif isinstance(image, np.ndarray):
                image = transform(image)
            elif isinstance(image, Image.Image):
                image = transform(image)
            else:
                raise TypeError(f"Unsupported image type: {type(image)}")

            poisoned_image = torch.clamp(image + torch.randn_like(image) * 0.1, 0, 1)

            # save back
            if isinstance(data, torch.Tensor) and data.dtype == torch.uint8:
                data[idx] = (poisoned_image * 255).byte()
            else:
                data[idx] = poisoned_image
        print(f"Data poisoning applied to approx {int(len(dataset)*frac)} samples in this partition")
            


    def apply_backdoor_attack(dataset, frac, target_label=None):
        transform = transforms.ToTensor()
        n = int(len(dataset) * frac)

        if isinstance(dataset, torch.utils.data.Subset):
            indices = random.sample(dataset.indices, n)
            underlying_dataset = dataset.dataset

            for idx in indices:
                # Torchvision datasets (MNIST, KMNIST, etc.)
                if hasattr(underlying_dataset, "data") and hasattr(underlying_dataset, "targets"):
                    image = underlying_dataset.data[idx]

                    # Convert to float tensor [0,1]
                    if isinstance(image, torch.Tensor):
                        image = image.float() / 255.0
                    else:
                        image = transform(image)

                    # Inject trigger (3x3 white square top-left)
                    image = image.clone()
                    if image.ndim == 3:  # (C, H, W)
                        image[:, :3, :3] = 1.0
                    elif image.ndim == 2:  # grayscale
                        image[:3, :3] = 1.0

                    # Save back into dataset.data
                    underlying_dataset.data[idx] = (image * 255).byte()

                    # Optional: force label to backdoor target
                    if target_label is not None:
                        underlying_dataset.targets[idx] = target_label

                # HuggingFace style dict datasets
                elif isinstance(underlying_dataset[idx], dict) and "image" in underlying_dataset[idx]:
                    sample = underlying_dataset[idx]
                    image = sample["image"]

                    if isinstance(image, Image.Image):
                        image = transform(image)
                    elif isinstance(image, np.ndarray):
                        image = torch.tensor(image, dtype=torch.float32)

                    image = image.clone()
                    if image.ndim == 3:
                        image[:, :3, :3] = 1.0
                    elif image.ndim == 2:
                        image[:3, :3] = 1.0

                    underlying_dataset[idx]["image"] = transforms.ToPILImage()(image)

                    if target_label is not None:
                        underlying_dataset[idx]["label"] = target_label

                else:
                    raise ValueError(f"Unsupported dataset type for backdoor attack: {type(underlying_dataset)}")
        print(f"Backdoor attack applied to approx {int(len(dataset)*frac)} samples in this partition")

    def get_label(item):
        if isinstance(item, dict) and "label" in item:
            return item["label"]
        else:
            # fallback to tuple/list style assuming label is at index 1
            return item[1]

    def collate_fn(batch):
        images, labels = [], []
        for item in batch:
            # Case 1: dict-style dataset (HuggingFace, FederatedDataset, etc.)
            if isinstance(item, dict):
                image, label = item["image"], item["label"]
            # Case 2: tuple-style dataset (torchvision)
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                image, label = item
            else:
                raise TypeError(f"Unsupported dataset item type: {type(item)}")

            # Normalize image types
            if isinstance(image, list):
                image = image[0]
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            if not isinstance(image, torch.Tensor):
                image = ToTensor()(image)

            images.append(image)
            labels.append(label)

        return torch.stack(images), torch.tensor(labels, dtype=torch.long)


    # --- Apply attacks ---
    print(f"Applying attack type '{attack_type}' on {mal_clients} malicious clients with attack fraction {attack_fraction}")
    for cid in range(mal_clients):
        if attack_type == "label_flipping":
            if dataset_name in ["mnist", "fashion_mnist", "kmnist"]:
                y_before = [get_label(train_partitions[cid][i]) for i in range(min(40, len(train_partitions[cid])))]
                train_partitions[cid] = apply_label_flipping_kmnist(train_partitions[cid], attack_fraction, attack_mode=attack_mode)
                y_after = [get_label(train_partitions[cid][i]) for i in range(min(40, len(train_partitions[cid])))]
                print("Labels before:", y_before)
                print("Labels after :", y_after)
                print(f"Applied Label Flipping to Client {cid}")
            elif dataset_name in ["cifar10"]:
                y_before = [get_label(train_partitions[cid][i]) for i in range(min(40, len(train_partitions[cid])))]
                train_partitions[cid] = apply_label_flipping_cifar10(train_partitions[cid], attack_fraction, attack_mode=attack_mode)
                y_after = [get_label(train_partitions[cid][i]) for i in range(min(40, len(train_partitions[cid])))]
                print("Labels before:", y_before)
                print("Labels after :", y_after)
                print(f"Applied Label Flipping to Client {cid}")
            elif dataset_name in ["svhn"]:
                y_before = [get_label(train_partitions[cid][i]) for i in range(min(40, len(train_partitions[cid])))]
                train_partitions[cid] = apply_label_flipping_svhn(train_partitions[cid], attack_fraction, attack_mode=attack_mode)
                y_after = [get_label(train_partitions[cid][i]) for i in range(min(40, len(train_partitions[cid])))]
                print("Labels before:", y_before)
                print("Labels after :", y_after)
                print(f"Applied Label Flipping to Client {cid}")
        elif attack_type == "data_poisoning":
            apply_data_poisoning(train_partitions[cid], attack_fraction)
            print(f"Applied Data Poisoning to Client {cid}")
        elif attack_type == "backdoor":
            apply_backdoor_attack(train_partitions[cid], attack_fraction)
            print(f"Applied Backdoor Attack to Client {cid}")
        elif attack_type == "none":
            pass

    print("Attacks applied to all malicious clients.")
    # --- Train/val/test loaders ---
    trainloaders, valloaders = [], []
    # for partition in train_partitions:
    for i, partition in enumerate(train_partitions):
        n_total = len(partition)
        n_val = int(val_ratio * n_total)
        n_train = n_total - n_val
        train_subset, val_subset = random_split(partition, [n_train, n_val], generator=torch.Generator().manual_seed(2023))
        trainloaders.append(DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn))
        valloaders.append(DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn))
        # print(f"Created train and val loaders for partition {i} with {n_train} train and {n_val} val samples")
    testloader = DataLoader(test_partition, batch_size=128, shuffle=False, collate_fn=collate_fn)
    print(f"Test loader created with batch size {128} and total samples {len(test_partition)}")
    print("Data loaders prepared successfully.")
    return trainloaders, valloaders, testloader