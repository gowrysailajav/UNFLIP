
from omegaconf import DictConfig
from model import test
import torch
from collections import OrderedDict
from hydra.utils import instantiate
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import flwr as fl
from flwr.common import Parameters


def get_on_fit_config(config: DictConfig):
    def fit_config_fn(server_round: int):
        
        fit_config = {
            'lr': config.get("lr", 0.005),
            'momentum': config.get("momentum", 0.9),  
            'local_epochs': config.get("local_epochs", 2)
        }
        
        return fit_config
    
    return fit_config_fn


def fit_metrics_aggregation_fn(metrics):
    if not isinstance(metrics, list):
        raise TypeError(f"Expected metrics to be a list, but got {type(metrics)}")

    metrics = [m[1] for m in metrics]  # Extract the actual metric dicts

    # Aggregate scalar metrics
    accuracies = [m["accuracy"] for m in metrics]
    precisions = [m["precision"] for m in metrics]
    recalls = [m["recall"] for m in metrics]
    f1_scores = [m["f1_score"] for m in metrics]
    losses = [m["loss"] for m in metrics]

    # Final aggregated result
    return {
        "loss": np.mean(losses),
        "accuracy": np.mean(accuracies),
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "f1_score": np.mean(f1_scores),
    }
 

def evaluate_metrics_aggregation_fn(metrics):
    metrics = [m[1] for m in metrics]  

    # Aggregate global scalar metrics
    accuracies = [m["accuracy"] for m in metrics]
    precisions = [m["precision"] for m in metrics]
    recalls = [m["recall"] for m in metrics]
    f1_scores = [m["f1_score"] for m in metrics]
    losses = [m["loss"] for m in metrics]

    # Final aggregated metrics dictionary
    return {
        "loss": np.mean(losses),
        "accuracy": np.mean(accuracies),
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "f1_score": np.mean(f1_scores),
    }


def get_evaluate_fn(model_cfg, testloader):
    def evaluate_fn(server_round: int, parameters, config):
        model = instantiate(model_cfg)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        if isinstance(parameters, list):
            parameters = fl.common.ndarrays_to_parameters(parameters)

        parameters_ndarray = fl.common.parameters_to_ndarrays(parameters)
        parameters_torch = [torch.tensor(p, dtype=torch.float32) for p in parameters_ndarray]

        params_dict = zip(model.state_dict().keys(), parameters_torch)
        state_dict = OrderedDict({k: v for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        # Run evaluation
        y_true, y_pred = [], []
        total_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                preds = torch.argmax(outputs, dim=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                total_loss += loss.item()

        # Metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        return total_loss / len(testloader), {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        }

    return evaluate_fn
 
