import os
import csv
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Net
from dataset import load_datasets

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#/*This function logs the results of each round to a CSV file.*/
def log_results_to_csv(round_num, client_id, accuracy, algo_name="FedAvg"):
    os.makedirs("results", exist_ok=True)
    file_path = f"results/resultats_{algo_name.lower()}.csv"
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["round", "client_id", "accuracy", "algorithm"])
        writer.writerow([round_num, client_id, accuracy, algo_name])

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader):
        self.model = model.to(DEVICE)
        self.trainloader = trainloader
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1):
            for images, labels in self.trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.model(images), labels)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
         # Logger automatique
        round_num = config.get("server_round", 0)
        client_id = config.get("cid", 0)  # pas toujours présent, on met 0 si inconnu
        algo_name = os.environ.get("FED_ALGO", "FedAvg")  # Lire l'algo depuis une variable d'environnement
        log_results_to_csv(round_num, client_id, accuracy, algo_name)
        
        return 0.0, len(self.trainloader.dataset), {"accuracy": accuracy}


if __name__ == "__main__":
    model = Net()
    client_datasets = load_datasets(num_clients=2)
    # Changer l'index ici pour simuler différents clients
    trainloader = DataLoader(client_datasets[1], batch_size=32, shuffle=True)
    fl.client.start_numpy_client(server_address="localhost:8080", client=FlowerClient(model, trainloader))
