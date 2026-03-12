import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from pathlib import Path
import pickle
from datetime import datetime
import os
import matplotlib.pyplot as plt
from sys import version
import hydra
from omegaconf import DictConfig, OmegaConf
from dataset import prepare_dataset
from client import generate_client_fn
import flwr as fl
from server import get_evaluate_fn, evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
import dataset  
import pathlib
import numpy as np
import seaborn as sns


def read_counter():
    """Reads the counter value from a file, returning 0 if the file does not exist."""
    counter_path = 'counter.txt'
    try:
        with open(counter_path, 'r') as file:
            return int(file.read().strip())
    except FileNotFoundError:
        return 0

def update_counter(counter):
    """Updates the counter value by incrementing and saving it to a file."""
    counter_path = 'counter.txt'
    with open(counter_path, 'w') as file:
        file.write(str(counter + 1))

counter = read_counter()
update_counter(counter)



def read_pkl_files(folder_path):
    
    """Reads all pkl files in the given folder and returns a dictionary of results."""
    results_data = {}
    folder = pathlib.Path(folder_path)
    
    for pkl_file in folder.glob("*.pkl"):
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
            results_data[pkl_file.stem] = data  # Use filename as the model identifier
    # print(results_data)
    return results_data



def plot_allmetrics(results_data, save_path, mal_clients, attack_fraction, dataset_name, aux_dataset_name):

    all_metrics = {}
    for key, result in results_data.items():
        metrics = {}
        history = result.get("history", None) 
        metrics = {
            "losses_distributed": history.losses_distributed,
            "losses_centralized": history.losses_centralized,
            "metrics_distributed_fit": history.metrics_distributed_fit,
            "metrics_distributed": history.metrics_distributed,
            "metrics_centralized": history.metrics_centralized
        }
        all_metrics[key] = metrics
        
    # Plot losses_distributed
    plt.figure(figsize=(10, 6))
    for key, values in all_metrics.items():
        losses = values["losses_distributed"]
        rounds, loss_values = zip(*losses)  # Unpack rounds and loss values
        plt.plot(rounds, loss_values, marker='o', linestyle='-', label=key)

    plt.xlabel("Rounds")
    plt.ylabel("Loss")
    plt.title("Distributed Losses Across Different Methods")
    plt.legend(fontsize=10)
    plt.grid(True)
    filename = 'losses_distributed'
    plt.savefig(f"{save_path}/{filename}_{mal_clients}_{attack_fraction}_{dataset_name}.pdf")
    plt.close()
    # plt.show()

    # Plot losses_centralized
    plt.figure(figsize=(10, 6))
    for key, values in all_metrics.items():
        losses = values["losses_centralized"]
        rounds, loss_values = zip(*losses)  # Unpack rounds and loss values
        plt.plot(rounds, loss_values, marker='s', linestyle='-', label=key)

    plt.xlabel("Rounds")
    plt.ylabel("Loss")
    plt.title("Centralized Losses Across Different Methods")
    plt.legend(fontsize=10)
    plt.grid(True)
    # plt.show()
    filename = 'losses_centralized'
    plt.savefig(f"{save_path}/{filename}_{mal_clients}_{attack_fraction}_{dataset_name}.pdf")
    plt.close()
    
    metrics = ["accuracy", "precision", "recall", "f1_score"]

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()  # Flatten for easy indexing

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for key, values in all_metrics.items():
            if "metrics_distributed_fit" in values and metric in values["metrics_distributed_fit"]:
                data = values["metrics_distributed_fit"][metric]
                rounds, metric_values = zip(*data)  # Unpack rounds and metric values
                ax.plot(rounds, metric_values, marker='o', linestyle='-', label=key)

        ax.set_xlabel("Rounds")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"{metric.capitalize()} over Rounds")
        ax.legend(fontsize=10)
        ax.grid(True)

    plt.tight_layout()
    filename = 'metrics_distributed_fit'
    plt.savefig(f"{save_path}/{filename}_{mal_clients}_{attack_fraction}_{dataset_name}.pdf")
    plt.close()

    metrics = ["accuracy", "precision", "recall", "f1_score"]

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()  # Flatten for easy indexing

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for key, values in all_metrics.items():
            if "metrics_distributed" in values and metric in values["metrics_distributed_fit"]:
                data = values["metrics_distributed"][metric]
                rounds, metric_values = zip(*data)  # Unpack rounds and metric values
                ax.plot(rounds, metric_values, marker='o', linestyle='-', label=key)

        ax.set_xlabel("Rounds")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"{metric.capitalize()} over Rounds")
        ax.legend(fontsize=10)
        ax.grid(True)

    plt.tight_layout()
    filename = 'metrics_distributed'
    plt.savefig(f"{save_path}/{filename}_{mal_clients}_{attack_fraction}_{dataset_name}.pdf")
    plt.close()
    
    metrics = ["accuracy", "precision", "recall", "f1_score"]

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()  # Flatten for easy indexing

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for key, values in all_metrics.items():
            if "metrics_centralized" in values and metric in values["metrics_distributed_fit"]:
                data = values["metrics_centralized"][metric]
                rounds, metric_values = zip(*data)  # Unpack rounds and metric values
                ax.plot(rounds, metric_values, marker='o', linestyle='-', label=key)

        ax.set_xlabel("Rounds")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"{metric.capitalize()} over Rounds")
        ax.legend(fontsize=10)
        ax.grid(True)

    plt.tight_layout()
    filename = 'metrics_centralized'
    plt.savefig(f"{save_path}/{filename}_{mal_clients}_{attack_fraction}_{dataset_name}.pdf")
    plt.close()
                                    
    print(f"Plots saved in : {save_path}")
    

@hydra.main(config_path='conf', config_name='base', version_base=None)
def main(cfg: DictConfig):


    # 1. Parse config & get experiment output directory
    print(OmegaConf.to_yaml(cfg))

    # 2. Prepare your dataset
    trainloaders, validationloaders, testloader = prepare_dataset(
        mal_clients=cfg.mal_clients,
        attack_type=cfg.attack_type,
        dataset_name=cfg.dataset_name,
        partition_type=cfg.partition_type,
        num_partitions=cfg.num_clients,
        batch_size=cfg.batch_size,
        val_ratio=cfg.val_ratio,            
        attack_fraction=cfg.attack_fraction,
        attack_mode=cfg.attack_mode
    )
    
    # 3. Define your clients
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.model)

    # 4. Define strategy
    strategy = instantiate(cfg.strategy, 
                           evaluate_fn=get_evaluate_fn(cfg.model,testloader),
                           fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                           evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn
                           )
    
    # 5. Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={'num_cpus': 4, 'num_gpus': 0 },
        
    )
    
    # 6. Save your results
    hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)
    base_dir = hydra_output_dir.parents[1]  
    master_dir = base_dir
    master_dir.mkdir(parents=True, exist_ok=True)
    if cfg["strategy"]["_target_"] == "flwr.server.strategy.FedAvg":
        print("Using FedAvg strategy")
        strategy_name = 'fedavg'      
    elif cfg["strategy"]["_target_"] == "0_ntn.create_strategy":
        print("Using 0_ntn Strategy")
        strategy_name = '0_ntn'           


    dataset_name = cfg.dataset_name if isinstance(cfg.dataset_name, str) else "unknown_dataset"
    sub_dir = master_dir / "pklfiles"
    sub_dir.mkdir(parents=True, exist_ok=True)
    results_path = sub_dir / f'exp{counter}_{strategy_name}_{int(cfg.percent_malicious*100)}_{int(cfg.attack_fraction*100)}_{cfg.attack_interval}.pkl'
    
    results = {'history': history, 'anythingelse': "putit here"}
    with open(str(results_path), 'wb') as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Results saved in: {results_path}")

    # 7. Plot results
    save_path = master_dir / "plots"
    save_path.mkdir(parents=True, exist_ok=True)
    results_data = read_pkl_files(sub_dir)
    plot_allmetrics(results_data, save_path, cfg.percent_malicious, cfg.attack_fraction, cfg.dataset_name, cfg.aux_dataset_name)

    return
    
if __name__ == '__main__':

    main()
    