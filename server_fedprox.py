import flwr as fl
from flwr.server.strategy import FedProx
from flwr.common import FitIns

class CustomFedProx(FedProx):
    def __init__(self):
        super().__init__(proximal_mu=0.1)  # <-- attention au nom exact ici

    def configure_fit(self, server_round, parameters, client_manager):
        instructions = super().configure_fit(server_round, parameters, client_manager)
        new_instructions = []
        for client, fit_ins in instructions:
            config = dict(fit_ins.config)
            config["server_round"] = server_round
            new_fit_ins = FitIns(fit_ins.parameters, config)
            new_instructions.append((client, new_fit_ins))
        return new_instructions

if __name__ == "__main__":
    strategy = CustomFedProx()
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
