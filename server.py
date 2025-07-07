import flwr as fl
from flwr.server.strategy import FedAvg

# Stratégie personnalisée
class CustomFedAvg(FedAvg):
    def configure_fit(self, server_round, parameters, client_manager):
        # Injecte le numéro du round dans les configurations client
        instructions = super().configure_fit(server_round, parameters, client_manager)
        new_instructions = []
        for client, ins in instructions:
            ins.config["server_round"] = server_round  # <- Injecte ici
            new_instructions.append((client, ins))
        return new_instructions

if __name__ == "__main__":
    strategy = CustomFedAvg()
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
