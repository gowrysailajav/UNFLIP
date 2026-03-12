
import torch
import flwr as fl
from collections import OrderedDict
from model import train, test
from typing import Dict
from flwr.common import NDArrays, Scalar
from hydra.utils import instantiate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

class FlowerClient(fl.client.NumPyClient):
    def __init__(self,
                  trainloader,
                  valloader,
                  model_cfg):
        
        super().__init__()
          
        self.trainloader = trainloader
        self.valloader = valloader
        
        self.model = instantiate(model_cfg)
 
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        # print(f'self.device = {self.device}')
        
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})      
        self.model.load_state_dict(state_dict, strict=True)
    
    
    def get_parameters(self, config: Dict[str, Scalar]): 
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
     
          
    def fit(self, parameters, config):
        # Load global parameters
        self.set_parameters(parameters)

        # Hyperparameters
        lr = config.get("lr", 0.0001)
        momentum = config.get("momentum", 0.9)
        epochs = config.get("local_epochs", 2)

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
        )

        # Local training (returns averaged gradients)
        train_loss, y_true, y_pred, avg_grads = train(
            self.model,
            self.trainloader,
            optimizer,
            epochs,
            self.device,
        )

        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Metrics dictionary (includes gradients)
        metrics = {
            "loss": train_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "gradients": avg_grads,   # <-- gradients sent to server
        }

        # Return parameters + metrics
        return (
            self.get_parameters(config),
            len(self.trainloader.dataset),
            metrics,
        )
    
    
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
    
        self.set_parameters(parameters)
        
        loss, accuracy, y_true, y_pred = test(self.model, self.valloader, self.device)
        
        # Ensure the evaluation function returns valid outputs
        if y_true is None or y_pred is None:
            return self.get_parameters(config), len(self.trainloader), {}

        # Compute evaluation metrics
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        metrics = {
            "loss": float(loss),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

        return float(loss), len(self.valloader), metrics
    

def generate_client_fn(trainloaders, valloaders, model_cfg):
    
    def client_fn(cid: str):
        
        
        return FlowerClient(trainloader=trainloaders[int(cid)],
                            valloader=valloaders[int(cid)],
                            model_cfg=model_cfg,
                            )
    
    return client_fn