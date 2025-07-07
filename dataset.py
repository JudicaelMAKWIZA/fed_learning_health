from torchvision import datasets, transforms
from torch.utils.data import Subset
import numpy as np

def load_datasets(num_clients=2):
    transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    data_per_client = len(full_dataset) // num_clients

    client_datasets = []
    for i in range(num_clients):
        indices = list(range(i * data_per_client, (i + 1) * data_per_client))
        client_datasets.append(Subset(full_dataset, indices))

    return client_datasets
