from typing import Union
from omegaconf import DictConfig
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import json
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy import Strategy
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Callable, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.mixture import GaussianMixture
import torch
import flwr as fl
from torchinfo import summary
from catforget_model import cached_prepare_forget_dataset, create_shadow_model_parameters
from model import train, test
import pickle
import random
import time
import psutil
import subprocess
import os
import uuid
from sklearn.base import BaseEstimator
from torch.utils.data import random_split, DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.svm import OneClassSVM
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import random
import threading
import gc
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset  
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
from typing import List, Tuple
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import hdbscan
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D 

  
class CustomFedavg(Strategy):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        mal_clients: int = 0,
        percent_malicious: float = 0,
        attack_interval: int = 1,
        num_clients: int = 0,
        num_rounds: int = 0,
        trainset_size: int = 0,
        aux_percent: float = 0,
        testset_size: int = 0,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        evaluate_fn: Optional[Callable] = None,  # Add this line
        num_classes: Optional[int] = None,  # Added num_classes
        model=None, 
        dataset_name='mnist',
        aux_dataset_name='mnist',
        partition_type='iid',
        batch_size = 128,
        config=None,
        threshold: float = 0.5,
        evaluate_metrics_aggregation_fn: Optional[Callable] = None,
        fit_metrics_aggregation_fn: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn  # Store it for later use
        self.num_classes = num_classes  # Store num_classes if needed
        self.model = model
        self.results_to_save = {}
        self.on_fit_config_fn = on_fit_config_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.mal_clients = mal_clients
        self.dataset_name = dataset_name
        self.aux_dataset_name = aux_dataset_name
        self.partition_type = partition_type
        self.batch_size = batch_size
        self.config = config
        self.percent_malicious = percent_malicious
        self.attack_interval = attack_interval
        self.num_clients = num_clients
        self.threshold = threshold
        self.aux_percent = aux_percent
        self.trainset_size = trainset_size
        self.testset_size = testset_size
        self.CACHE_PATH = self.get_random_cache_path()
        self.SHADOW_CACHE_PATH = self.get_shadow_model_path()
        self.num_rounds = num_rounds
        self.total_to_select = self.min_available_clients 
        self.num_malicious_selected = int(self.percent_malicious * self.total_to_select)
        self.num_normal_selected = self.total_to_select - self.num_malicious_selected
        # print(f'total_clients : {self.total_to_select} \n num_mal_to_select : {self.num_malicious_selected} \n num_normal_to_select : {self.num_normal_selected} ')
        
             
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
    def get_random_cache_path(self):
        uid = uuid.uuid4().hex[:8]   # short unique ID
        return f"/home/sailaja/Documents/p7_ntn/z_cached_forget_dataset_{uid}.pkl"
        
    def get_shadow_model_path(self):
        uid = uuid.uuid4().hex[:8]   # short unique ID
        return f"/home/sailaja/Documents/p7_ntn/z_cached_shadow_model_{uid}.pkl"
  
    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters
    
    def default_eval_metrics_aggregation(
        self, server_round: int, results: List, failures: List
    ) -> Dict[str, float]:
        """Aggregate evaluation metrics (e.g., accuracy, loss) across clients."""
        
        if not results:
            return {}

        num_clients = len(results)

        # Extract metrics from EvaluateRes objects
        try:
            aggregated_metrics = {
                key: sum(res.metrics[key] for res in results) / num_clients
                for key in results[0].metrics
            }
        except Exception as e:
            print(f"Error during metric aggregation: {e}")
            return {}

        return aggregated_metrics

        

    def __repr__(self) -> str:
        return "FedCustom"

    def configure_fit(self, server_round: int, parameters, client_manager):
        self.current_global_weights = parameters_to_ndarrays(parameters)
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
            
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        
        if server_round % self.attack_interval == 0:
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients, criterion=HardIncludeMalCidCriterion(self.mal_clients, self.num_clients, self.percent_malicious, total_to_select=sample_size)
            )
        else:
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients, criterion=ExcludeMalCidCriterion(self.mal_clients)
            )
            
        return [(client, fit_ins) for client in clients]



    def sanitize_params(self, params):
        clean = []
        for i, p in enumerate(params):
            if isinstance(p, torch.Tensor):
                clean.append(p.detach().cpu().numpy().astype(np.float32))
            elif isinstance(p, np.ndarray):
                clean.append(p.astype(np.float32))
            elif isinstance(p, (list, tuple)):
                # If list of scalars → ok, else skip
                try:
                    arr = np.array(p, dtype=np.float32)
                    if arr.ndim > 1 and any(isinstance(x, (list, np.ndarray)) for x in p):
                        print(f"Skipping nested parameter at index {i}, type={type(p)}")
                        continue
                    clean.append(arr)
                except Exception as e:
                    print(f"Skipping invalid param at index {i}, type={type(p)}, err={e}")
                    continue
            elif isinstance(p, (int, float)):
                clean.append(np.array(p, dtype=np.float32))
            else:
                print(f"Skipping unsupported param at index {i}, type={type(p)}")
                continue
        return clean


    def monitor_memory(self, memory_samples, stop_event, cpu_samples=None, interval=0.1):
        process = psutil.Process(os.getpid())
        process.cpu_percent(interval=None)
        while not stop_event.is_set():
            mem = process.memory_info().rss / 1024 ** 2
            memory_samples.append(mem)
            if cpu_samples is not None:
                cpu = process.cpu_percent(interval=None)
                cpu_samples.append(cpu)
            time.sleep(interval)

    def compute_client_detection_metrics(
        self,
        total_clients: int,
        actual_malicious: int,
        actual_normal: int,
        selected_clients: int,
        rejected_clients: int,
        wrongly_selected_malicious: int,
    ) -> Dict[str, float]:
        """Compute evaluation metrics for client anomaly detection."""

        # True Positives: Malicious correctly rejected
        TP = actual_malicious - wrongly_selected_malicious

        # False Negatives: Malicious wrongly selected
        FN = wrongly_selected_malicious

        # False Positives: Normal wrongly rejected
        FP = rejected_clients - TP

        # True Negatives: Normal correctly selected
        TN = actual_normal - FP

        # Accuracy
        accuracy = (TP + TN) / total_clients if total_clients else 0

        # Normal client perspective
        normal_precision = TN / (TN + FN) if (TN + FN) else 0
        normal_recall = TN / (TN + FP) if (TN + FP) else 0
        normal_f1 = 2 * normal_precision * normal_recall / (normal_precision + normal_recall) if (normal_precision + normal_recall) else 0

        # Malicious client perspective
        malicious_precision = TP / (TP + FP) if (TP + FP) else 0
        malicious_recall = TP / (TP + FN) if (TP + FN) else 0
        malicious_f1 = 2 * malicious_precision * malicious_recall / (malicious_precision + malicious_recall) if (malicious_precision + malicious_recall) else 0

        # Other metrics
        specificity = TN / (TN + FP) if (TN + FP) else 0
        false_selection_rate = FN / actual_malicious if actual_malicious else 0

        return {
            "TP_malicious": TP,
            "FN_malicious": FN,
            "FP_normal": FP,
            "TN_normal": TN,
            "detection_accuracy": accuracy,
            "normal_precision": normal_precision,
            "normal_recall": normal_recall,
            "normal_f1": normal_f1,
            "malicious_precision": malicious_precision,
            "malicious_recall": malicious_recall,
            "malicious_f1": malicious_f1,
            "specificity": specificity,
            "false_selection_rate": false_selection_rate,
        }


    def load_parameters_into_model(self, model, flwr_parameters):
        """Load Flower parameters into a PyTorch model."""
        params_ndarrays = parameters_to_ndarrays(flwr_parameters)

        state_dict = model.state_dict()
        new_state_dict = {}

        for (key, _), value in zip(state_dict.items(), params_ndarrays):
            new_state_dict[key] = torch.tensor(value)

        model.load_state_dict(new_state_dict, strict=True)


    def evaluate_model_loss(
        self,
        model,
        target_testloader,
    ):
        """Evaluate loss of a client model on the target test set."""
        model.eval()

        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for x, y in target_testloader:
                x, y = x.to(self.device), y.to(self.device)
                logits = model(x)
                loss = criterion(logits, y)

                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)

        return total_loss / max(total_samples, 1)


    def detect_malicious_clients_gac(
        self,
        client_updates: np.ndarray,
        client_losses: np.ndarray,
        cids: list,
        server_round: int,
        percent_malicious: float,
        n_components: int = 2,
    ):
        """
        Identify malicious clients using PCA + clustering,
        and visualize PCA space with ground-truth coloring (cid-based),
        including a 3D PCA + loss plot.

        Returns:
            malicious_indices (List[int])
            cluster_labels (np.ndarray)
        """
        base_path = "/home/sailaja/Documents/p7_ntn/outputs/"
        tag = f"{int(percent_malicious * 100)}_{server_round}"

        # -----------------------------
        # Safety checks
        # -----------------------------
        client_updates = np.asarray(client_updates, dtype=np.float64)
        client_losses = np.asarray(client_losses, dtype=np.float64)
        cids = np.array([int(cid) for cid in cids])

        assert (
            client_updates.shape[0]
            == client_losses.shape[0]
            == len(cids)
        ), "Mismatch in number of clients"

        client_losses = np.nan_to_num(
            client_losses,
            nan=np.inf,
            posinf=np.inf,
            neginf=np.inf,
        )

        # Replace inf with large finite value (robust)
        finite_losses = client_losses[np.isfinite(client_losses)]
        max_finite = finite_losses.max() if finite_losses.size > 0 else 1.0
        client_losses = np.where(
            np.isfinite(client_losses),
            client_losses,
            max_finite * 1.5,
        )

        # -----------------------------
        # Step 1: PCA projection
        # -----------------------------
        pca = PCA(n_components=n_components, random_state=0)
        updates_2d = pca.fit_transform(client_updates)   # shape [N, 2]

        # -----------------------------
        # Step 2: Robust loss scaling
        # -----------------------------
        loss_scaled = np.clip(
            client_losses,
            np.percentile(client_losses, 5),
            np.percentile(client_losses, 95),
        ) 

        loss_scaled = StandardScaler().fit_transform(
            loss_scaled.reshape(-1, 1)
        ).squeeze()  # shape [N]

        # Optional: weight loss importance (tune if needed)
        loss_weight = 1.0   # try 0.5, 1.0, 2.0
        loss_scaled *= loss_weight

        # -----------------------------
        # Step 3: Joint feature space
        # -----------------------------
        # Feature vector = [PCA1, PCA2, scaled_loss]
        features = np.column_stack([updates_2d, loss_scaled])
        
        # -----------------------------
        # Step 4: KMeans clustering
        # -----------------------------
        kmeans = KMeans(n_clusters=2, n_init=20, random_state=0)
        cluster_labels = kmeans.fit_predict(features)
        centers = kmeans.cluster_centers_
        from numpy.linalg import norm

        radii = []
        for k in [0, 1]:
            idx = np.where(cluster_labels == k)[0]
            centers_2d = centers[:, :2]   # drop loss dimension

            distances = np.linalg.norm(
                updates_2d[idx] - centers_2d[k],
                axis=1
            )

            radii.append(distances.max())


        # centroid distance in PCA space
        centroid_distance = norm(centers_2d[0] - centers_2d[1])
        overlap_margin = (radii[0] + radii[1]) - centroid_distance

        print(f"Centroid distance (PCA): {centroid_distance:.4f}")
        print(f"R0 + R1: {radii[0] + radii[1]:.4f}")

        cluster_0_idx = np.where(cluster_labels == 0)[0]
        cluster_1_idx = np.where(cluster_labels == 1)[0]

        # -----------------------------
        # Distance between clusters
        # -----------------------------
        cluster_distance = np.linalg.norm(centers[0] - centers[1])

        print(f"📏 Distance between cluster centroids: {cluster_distance:.4f}")

        # -----------------------------
        # Shared visualization setup
        # -----------------------------
        # colors = ["red" if cid < 40 else "green" for cid in cids]

        # # Robust loss scaling for size
        # loss_clip = np.clip(
        #     client_losses,
        #     np.percentile(client_losses, 5),
        #     np.percentile(client_losses, 95),
        # )
        # sizes = 40 + 120 * (loss_clip - loss_clip.min()) / (
        #     loss_clip.max() - loss_clip.min() + 1e-12
        # )

        # plt.figure(figsize=(8, 7))

        # # scatter points
        # plt.scatter(
        #     updates_2d[:, 0],
        #     updates_2d[:, 1],
        #     c=colors,
        #     s=sizes,
        #     alpha=0.8,
        #     edgecolors="k",
        #     linewidths=0.4,
        # )

        # # draw circles
        # for k, color in zip([0, 1], ["blue", "orange"]):
        #     circle = Circle(
        #         centers[k],
        #         radii[k],
        #         fill=False,
        #         linewidth=2,
        #         linestyle="--",
        #         color=color,
        #         label=f"Cluster {k} radius"
        #     )
        #     plt.gca().add_patch(circle)

        # # centroids
        # plt.scatter(
        #     centers[:, 0],
        #     centers[:, 1],
        #     c="black",
        #     s=180,
        #     marker="X",
        #     label="Centroids",
        # )

        # plt.xlabel("PCA Component 1", fontsize=13)
        # plt.ylabel("PCA Component 2", fontsize=13)
        # plt.title("KMeans Clusters", fontsize=14)
        # plt.legend()
        # plt.grid(alpha=0.3)
        # plt.tight_layout()

        # plt.savefig(f"{base_path}/kmeans_cluster_circles_{tag}.pdf", dpi=300)
        # plt.close()

        if overlap_margin > 0:
            print(f"⚠️ Clusters OVERLAP by {overlap_margin:.4f}")
        else:
            print(f"✅ Clusters SEPARATED by {-overlap_margin:.4f}")

        def safe_mean(arr):
            return float(np.mean(arr)) if arr.size > 0 else np.inf

        loss_0 = safe_mean(client_losses[cluster_0_idx])
        loss_1 = safe_mean(client_losses[cluster_1_idx])

        if loss_0 <= loss_1:
            benign_indices = cluster_0_idx
            malicious_indices = cluster_1_idx
        else:
            benign_indices = cluster_1_idx
            malicious_indices = cluster_0_idx

        benign_loss = safe_mean(client_losses[benign_indices])
        malicious_loss = safe_mean(client_losses[malicious_indices])
        
        mean_loss = 0.5 * (benign_loss + malicious_loss)
        loss_gap = abs(benign_loss - malicious_loss)

        LOSS_GAP_EPS = 0.05
        LOW_LOSS_TH  = 0.5
        print(f'mean_loss : {mean_loss} | LOW_LOSS_TH : {LOW_LOSS_TH} | LOSS_GAP_EPS : {LOSS_GAP_EPS} | loss_gap : {loss_gap} ')

        if loss_gap <= LOSS_GAP_EPS:
            if mean_loss > LOW_LOSS_TH:
                regime = "ALL_BENIGN"
            else:
                regime = "ALL_MALICIOUS"
        else:
            regime = "MIXED"

        print(f"🧠 Regime detected: {regime}")
        print(
            f"[PCA+CLUSTER+LOSS] "
            f"Benign avg loss: {benign_loss:.4f} | "
            f"Malicious avg loss: {malicious_loss:.4f} | "
            f"Rejected: {len(malicious_indices)}"
        )
        
        # =========================================================
        # TRUST SCORE COMPUTATION (PCA + LOSS GEOMETRY)
        # =========================================================

        # Benign centroid in PCA space
        # benign_center = centers[0] if loss_0 <= loss_1 else centers[1] # 2PCS + LOSS
        benign_center = (centers[0] if loss_0 <= loss_1 else centers[1])[:2] # PCA-only centroid
        benign_radius = radii[0] if loss_0 <= loss_1 else radii[1]

        # Precompute once (avoid repeated norms)
        diffs = updates_2d - benign_center
        dists = np.linalg.norm(diffs, axis=1)

        # Gaussian distance trust (clients near benign centroid get high trust)
        dist_trust = np.exp(-(dists ** 2) / (2 * (benign_radius ** 2 + 1e-12)))

        # Loss trust (already scaled)
        loss_trust = np.exp(-loss_scaled)

        # Final trust score
        trust_scores = dist_trust * loss_trust

        # Normalize to simplex
        trust_scores = np.clip(trust_scores, 0, None)
        trust_scores /= trust_scores.sum() + 1e-12
        
        # -----------------------------
        # 3D PCA + Loss plot (NEW)
        # -----------------------------
        # fig = plt.figure(figsize=(9, 7))
        # ax = fig.add_subplot(111, projection="3d")

        # ax.scatter(
        #     updates_2d[:, 0],
        #     updates_2d[:, 1],
        #     client_losses,
        #     c=colors,
        #     s=sizes,
        #     alpha=0.75,
        #     edgecolors="k",
        #     linewidths=0.3,
        # )

        # ax.set_xlabel("PCA Component 1", fontsize=12)
        # ax.set_ylabel("PCA Component 2", fontsize=12)
        # ax.set_zlabel("Client Loss", fontsize=12)
        # ax.set_title("KMeans Clusters", fontsize=14)

        # plt.tight_layout()
        # plt.savefig(f"{base_path}/pca_3d_loss_{tag}.pdf", dpi=300)
        # plt.close()

        return malicious_indices.tolist(), overlap_margin, regime, trust_scores

    def shadow_project_aggregate(
        self,
        selected_results,
        shadow_model_params,
        gamma=0.1,
    ):
        """
        Shadow + momentum projection aggregation with debug prints.
        """

        print("\n======== SHADOW PROJECT AGGREGATE START ========")

        # Previous global
        flat_prev = np.concatenate([w.flatten() for w in self.current_global_weights])

        # Shadow model
        flat_shadow = np.concatenate([w.flatten() for w in shadow_model_params])

        # ---------------------------------------------------
        # Collect client params
        # ---------------------------------------------------
        client_params = [parameters_to_ndarrays(r.parameters) for _, r in selected_results]
        num_clients = len(client_params)

        print(f"[SP] Clients received: {num_clients}")

        # flat_clients = np.vstack([
        #     np.concatenate([w.flatten() for w in p])
        #     for p in client_params
        # ])

        # ---------------------------------------------------
        # Client params (with shadow added)
        # ---------------------------------------------------
        flat_clients = np.vstack([
            np.concatenate([w.flatten() for w in p]) + flat_shadow
            for p in client_params
        ])

        # ---------------------------------------------------
        # Shadow direction
        # ---------------------------------------------------
        d_shadow = flat_shadow - flat_prev
        shadow_norm = np.linalg.norm(d_shadow) + 1e-12
        d_shadow /= shadow_norm

        print(f"[SP] Shadow direction norm: {shadow_norm:.4e}")

        # ---------------------------------------------------
        # Momentum
        # ---------------------------------------------------
        if hasattr(self, "prev_prev_global_weights"):
            flat_prevprev = np.concatenate([w.flatten() for w in self.prev_prev_global_weights])
            momentum = flat_prev - flat_prevprev
            mom_norm = np.linalg.norm(momentum) + 1e-12
            momentum /= mom_norm
            use_momentum = True
            print(f"[SP] Momentum norm: {mom_norm:.4e}")
        else:
            momentum = None
            use_momentum = False
            print("[SP] No momentum available (round 1)")

        # ---------------------------------------------------
        # Client deltas
        # ---------------------------------------------------
        deltas = flat_clients - flat_prev
        delta_norms = np.linalg.norm(deltas, axis=1)

        print(
            f"[SP] Client delta norms: "
            f"min={delta_norms.min():.4e}, "
            f"med={np.median(delta_norms):.4e}, "
            f"max={delta_norms.max():.4e}"
        )

        # ---------------------------------------------------
        # Vectorized projections
        # ---------------------------------------------------
        shadow_proj = np.maximum(0.0, deltas @ d_shadow)

        print(
            f"[SP] Shadow projection scalars: "
            f"min={shadow_proj.min():.4e}, "
            f"med={np.median(shadow_proj):.4e}, "
            f"max={shadow_proj.max():.4e}"
        )

        if use_momentum:
            momentum_proj = np.maximum(0.0, deltas @ momentum)

            print(
                f"[SP] Momentum projection scalars: "
                f"min={momentum_proj.min():.4e}, "
                f"med={np.median(momentum_proj):.4e}, "
                f"max={momentum_proj.max():.4e}"
            )

            cleaned_updates = (
                shadow_proj[:, None] * d_shadow +
                gamma * momentum_proj[:, None] * momentum
            )
        else:
            cleaned_updates = shadow_proj[:, None] * d_shadow

        # ---------------------------------------------------
        # Mean update + drift diagnostics
        # ---------------------------------------------------
        raw_mean_delta = deltas.mean(axis=0)
        mean_update = cleaned_updates.mean(axis=0)

        raw_norm = np.linalg.norm(raw_mean_delta)
        clean_norm = np.linalg.norm(mean_update)

        cos_raw_shadow = (raw_mean_delta @ d_shadow) / (raw_norm + 1e-12)
        cos_clean_shadow = (mean_update @ d_shadow) / (clean_norm + 1e-12)

        print(f"[SP-DRIFT] Raw delta norm      : {raw_norm:.4e}")
        print(f"[SP-DRIFT] Clean delta norm    : {clean_norm:.4e}")
        print(f"[SP-DRIFT] Shrink ratio        : {clean_norm/(raw_norm+1e-12):.4f}")
        print(f"[SP-DRIFT] Cos(raw, shadow)   : {cos_raw_shadow:.4f}")
        print(f"[SP-DRIFT] Cos(clean, shadow) : {cos_clean_shadow:.4f}")

        # ---------------------------------------------------
        # Apply update
        # ---------------------------------------------------
        flat_new = flat_prev + mean_update

        print("[SP] Global update applied")

        # ---------------------------------------------------
        # Reconstruct model
        # ---------------------------------------------------
        idx = 0
        reconstructed = []
        for w in self.current_global_weights:
            size = w.size
            reconstructed.append(flat_new[idx:idx+size].reshape(w.shape))
            idx += size

        print(f"[SP] Reconstructed {len(reconstructed)} layers")
        print(f"[SP] New model L2 norm: {np.linalg.norm(flat_new):.4f}")

        # Save momentum
        self.prev_prev_global_weights = copy.deepcopy(self.current_global_weights)

        print("======== SHADOW PROJECT AGGREGATE END ========\n")

        return ndarrays_to_parameters(reconstructed)

    def parameters_to_flat(self, params):
        nds = parameters_to_ndarrays(params)
        return np.concatenate([w.flatten() for w in nds])

    def flat_to_parameters(self, flat):
        idx = 0
        reconstructed = []
        for w in self.current_global_weights:
            size = w.size
            reconstructed.append(flat[idx:idx+size].reshape(w.shape))
            idx += size
        return ndarrays_to_parameters(reconstructed)

    def flatten_gradients(self, grad_dict):
        """
        grad_dict: Dict[str, np.ndarray]
        """
        flat_grads = []
        for _, g in grad_dict.items():
            flat_grads.append(g.reshape(-1))
        return np.concatenate(flat_grads)

    def flatten_flwr_parameters(self, flwr_parameters):
        """
        Convert Flower Parameters -> flattened numpy vector
        """
        ndarrays = parameters_to_ndarrays(flwr_parameters)
        flat = np.concatenate([p.flatten() for p in ndarrays])
        return flat


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        # metrics_aggregated = {}
        # start_time = time.time()
        # start = time.process_time()
        # memory_samples = []
        # cpu_samples = []
        # stop_event = threading.Event()
        # monitor_thread = threading.Thread(
        #     target=self.monitor_memory,
        #     args=(memory_samples, stop_event, cpu_samples)
        # )
        # monitor_thread.start()
        
        if not results:
            return None, {}

        if not self.accept_failures and failures:
            return None, {}
        
        # preparing root dataset
        trainloader, target_testloader = cached_prepare_forget_dataset(
            self.dataset_name,
            self.trainset_size, self.testset_size,
            self.partition_type, self.batch_size,
            self.CACHE_PATH
        )
        lr = self.config.get('lr', 0.005)  
        momentum = self.config.get('momentum', 0.9)
        epochs = 10
        # print(f'lr: {lr} | momentum: {momentum} | epochs: {epochs}')
        
        # creating shadow mmodel parameters for augmentation
        shadow_model_params = create_shadow_model_parameters(
            trainloader,
            self.model,
            lr, momentum, epochs, self.device,
            self.SHADOW_CACHE_PATH
        )
        
        # augmenting client weights with shadow model parameters
        client_weights = [parameters_to_ndarrays(r.parameters) for _, r in results]
        # augmented_client_weights = []
        # for cw in client_weights:
        #     merged = [
        #         c + s
        #         for c, s in zip(cw, shadow_model_params)
        #     ]
        #     augmented_client_weights.append(merged)

        # client_weights = augmented_client_weights
        
        # Flatten weights into single vectors
        flat_weights = np.array([np.concatenate([w.flatten() for w in weights]) for weights in client_weights])
        
        # 2. Get the Global Model Weights (the weights you sent out at start of round)
        # Assuming you store them in self.current_global_weights
        global_weights_flat = np.concatenate([w.flatten() for w in self.current_global_weights])
        
        # 1. Displacement Vectors (Directional movement)
        displacement_vectors = flat_weights - global_weights_flat
        
        # norms = np.linalg.norm(displacement_vectors, axis=1, keepdims=True) + 1e-12
        # normalized_disp = displacement_vectors / norms
        # median_direction = np.median(normalized_disp, axis=0).reshape(1, -1)
        
        
        # 2. Calculate Median Direction (Robust Consensus)
        median_direction = np.median(displacement_vectors, axis=0).reshape(1, -1)
        
        # 3. Calculate Cosine Similarity of each client's movement to the consensus
        from sklearn.metrics.pairwise import cosine_similarity
        movement_similarities = cosine_similarity(displacement_vectors, median_direction).flatten()
        
        # 4. Check for the pattern
        min_sim = np.min(movement_similarities)
        max_sim = np.max(movement_similarities)
        med_sim = np.median(movement_similarities)
        gap = med_sim - min_sim
        gap2 = max_sim - min_sim
        # === Dynamic Skewness Gate ===
        current_skew = gap / (gap2 - gap + 1e-6)
        
        # Initialize history if not present
        if not hasattr(self, "prev_skew"):
            self.prev_skew = current_skew
            bypass_clustering = True # Safety bypass for Round 1
        else:
            # Logic: If skewness jumps significantly (> 20% increase), 
            # it indicates a new divergence (attack).
            skew_jump = current_skew / (self.prev_skew + 1e-6)
            
            print(f'threshold = {self.threshold} | current_skew : {current_skew:.4f} | prev_skew : {self.prev_skew:.4f} | skew_jump : {skew_jump:.4f} ')
            if skew_jump > self.threshold or current_skew > 8.0:
            # if skew_jump > 1.0 or current_skew > 5.0:
            # if current_skew > 5.0:
                print(f"🚨 [Gate] Skewness Spike! {self.prev_skew:.2f} -> {current_skew:.2f} (Jump: {skew_jump:.2f})")
                bypass_clustering = False
            else:
                print(f"✅ [Gate] Stable Skewness. {self.prev_skew:.2f} -> {current_skew:.2f} (Jump: {skew_jump:.2f})")
                bypass_clustering = True
            
            # Update for next round
            self.prev_skew = current_skew
            
        print(f"Round {server_round} - min={min_sim:.4f}, med={med_sim:.4f}, max={max_sim:.4f}, Gap: {gap:.4f}, gap2={gap2:.4f}, gap1-gap2={gap2-gap:.4f}, current_skew={current_skew:.4f}, Bypass Clustering: {bypass_clustering}")
        
        if bypass_clustering:
            metrics_aggregated = {}
            start_time = time.time()
            start = time.process_time()
            memory_samples = []
            cpu_samples = []
            stop_event = threading.Event()
            monitor_thread = threading.Thread(
                target=self.monitor_memory,
                args=(memory_samples, stop_event, cpu_samples)
            )
            monitor_thread.start()
            print("⚠️ Bypassing clustering, treating all clients as benign.")
            parameters_aggregated = ndarrays_to_parameters(aggregate([
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]))
            filtered_results = results
           
        else:
            # trainloader, target_testloader = cached_prepare_forget_dataset(
            #         self.dataset_name,
            #         self.trainset_size, self.testset_size,
            #         self.partition_type, self.batch_size,
            #         self.CACHE_PATH
            #     )
            # === GAD for finding lambda ===
            metrics_aggregated = {}
            start_time = time.time()
            start = time.process_time()
            memory_samples = []
            cpu_samples = []
            stop_event = threading.Event()
            monitor_thread = threading.Thread(
                target=self.monitor_memory,
                args=(memory_samples, stop_event, cpu_samples)
            )
            monitor_thread.start()
            filtered_results = []
            filtered_malignant_results = []
            
            sm = []
            client_updates = []
            client_losses = []
            client_parameters = []   # ✅ NEW
            cids = []
            updates = []

            for client_proxy, fit_res in results:
                client_id = client_proxy.cid

                # ---- Gradient-based update ----
                if "gradients" not in fit_res.metrics:
                    print(f"[Round {server_round}] No gradients from {client_id}, skipping.")
                    continue

                try:
                    # ---- Gradients ----
                    grad_dict = fit_res.metrics["gradients"]
                    grad_vec = self.flatten_gradients(grad_dict)

                    # ---- Flattened model parameters (NEW) ----
                    param_vec = self.flatten_flwr_parameters(fit_res.parameters)

                    # ---- Evaluate loss on target_testloader ----
                    client_model = copy.deepcopy(self.model).to(self.device)
                    self.load_parameters_into_model(
                        model=client_model,
                        flwr_parameters=fit_res.parameters,
                    )

                    client_loss = self.evaluate_model_loss(
                        model=client_model,
                        target_testloader=target_testloader,
                    )

                    # ---- Collect ----
                    client_updates.append(grad_vec)
                    client_parameters.append(param_vec)   # ✅ NEW
                    client_losses.append(client_loss)
                    cids.append(client_id)

                except Exception as e:
                    print(f"[Round {server_round}] Error processing {client_id}: {e}")

            print("🔍 Performing PCA + Clustering for malicious client detection...")
            dissimilar_indices, overlap_margin, regime, trust_scores = self.detect_malicious_clients_gac(
                client_updates=client_updates,
                client_losses=client_losses,
                cids=cids,
                server_round=server_round,
                percent_malicious=self.percent_malicious,
                # k=2.5
            )

            dissimilar_cids = [cids[i] for i in dissimilar_indices]
            selected_clients = []
            wrongly_selected = []
            
            for client_proxy, fit_res in results:
                client_id = client_proxy.cid
                if client_id not in dissimilar_cids:
                    selected_clients.append(client_proxy.cid)
                    filtered_results.append((client_proxy, fit_res))
                else: 
                    filtered_malignant_results.append((client_proxy, fit_res))
            
            for cid in selected_clients:
                if int(cid) < self.mal_clients:
                    wrongly_selected.append(cid)
                    
            # If no clients passed rejection, abort round
            if not filtered_results:
                print("❌ All clients rejected — skipping round.")
                return None, {}  
            perfectly_selected = [x for x in selected_clients if x not in wrongly_selected]
        
            print(f'✅ Total Clients : {len(cids)} \n✅ Total Fit Selected Clients : {len(selected_clients)} \n✅ Selected Benign Clients : {len(perfectly_selected)} | {len(perfectly_selected)/len(selected_clients)} | {perfectly_selected} \n✅ Selected Malicious Clients : {len(wrongly_selected)} | {len(wrongly_selected)/len(selected_clients)} | {wrongly_selected} ')
            malicious_ratio = round(len(dissimilar_cids) / max(len(cids), 1), 1) #lambda 
            print(f'✅ actual malicious_ratio : {self.percent_malicious}\n✅ obtained malicious_ratio : {malicious_ratio}')

            # if regime in ["MIXED", "ALL_BENIGN"] and overlap_margin <= 0:
            if overlap_margin <= 0:
                print(f"✅ {regime} regime → aggregating benign + corrected malignant")

                # ------------------------------------------------
                # 1. FedAvg on benign
                # ------------------------------------------------
                print(f"📊 Benign cluster: {len(filtered_results)} clients → applying FedAvg")
                benign_params = ndarrays_to_parameters(
                    aggregate([
                        (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                        for _, fit_res in filtered_results
                    ])
                )

                flat_benign = self.parameters_to_flat(benign_params)

                # ------------------------------------------------
                # 2. Shadow-project ONLY malignant
                # ------------------------------------------------
                print(f"📊 Malignant cluster: {len(filtered_malignant_results)} clients → applying Shadow Projection")
                malignant_params = self.shadow_project_aggregate(
                    filtered_malignant_results,
                    shadow_model_params
                )

                flat_malignant = self.parameters_to_flat(malignant_params)

                # ------------------------------------------------
                # 3. Combine BOTH (simple mean)
                # ------------------------------------------------
                benign_ratio = round(1 - malicious_ratio, 1)   # benign trust (tuneable)
                # print(f"🔗 Combining benign {1-malicious_ratio} + malignant with beta = {malicious_ratio} blending")
                # mal_norm = np.linalg.norm(flat_malignant)
                # ben_norm = np.linalg.norm(flat_benign)

                # if mal_norm > ben_norm:
                #     scale = ben_norm / (mal_norm + 1e-12)
                #     flat_malignant *= scale
                #     print(f"[BLEND] Malignant rescaled by {scale:.4f}")
                
                if malicious_ratio > 0.5:
                    flat_final = malicious_ratio * flat_benign + (1 - malicious_ratio) * flat_malignant
                else:
                    flat_final = benign_ratio * flat_benign + malicious_ratio * flat_malignant
                parameters_aggregated = self.flat_to_parameters(flat_final)

                print(
                    f"✅ Benign: {len(filtered_results)} | "
                    f"Malignant corrected: {len(filtered_malignant_results)} | "
                    f"benign_ratio={benign_ratio}, malicious_ratio={malicious_ratio}"
                )
            else:
                print(f"⚠️ High cluster overlap detected → applying shadow projection to ALL clients (no clustering) ")
                parameters_aggregated = self.shadow_project_aggregate(
                    results,
                    shadow_model_params
                )
           
        # === Metrics aggregation ===
        
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in filtered_results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            print("⚠️ No fit_metrics_aggregation_fn provided")

        # === System metrics ===
        end_time = time.time()
        end = time.process_time()
        stop_event.set()
        monitor_thread.join()
        if memory_samples:
            avg_memory = sum(memory_samples) / len(memory_samples)
            peak_memory = max(memory_samples)
        else:
            avg_memory = peak_memory = 0

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        elapsed_time = end_time - start_time
        cpu_time = end - start
        time_taken = elapsed_time + cpu_time
        if cpu_samples:
            # print(cpu_samples)
            avg_cpu = sum(cpu_samples) / len(cpu_samples)
            peak_cpu = max(cpu_samples)
        else:
            avg_cpu = peak_cpu = 0
        gc.collect()

        metrics_aggregated["elapsed_time"] = elapsed_time
        metrics_aggregated["cpu_time"] = cpu_time
        metrics_aggregated["memory_used_mb"] = sum(memory_samples)
        metrics_aggregated["memory_samples"] = len(memory_samples)
        metrics_aggregated["avg_memory"] = avg_memory
        metrics_aggregated["peak_memory"] = peak_memory
        metrics_aggregated["cpu_samples"] = len(cpu_samples)
        metrics_aggregated["avg_cpu"] = avg_cpu
        metrics_aggregated["peak_cpu"] = peak_cpu
        # metrics_aggregated["all_cids"] = cids
        # metrics_aggregated["selected_cids"] = selected_clients
        # metrics_aggregated["not_selected_cids"] = dissimilar_cids
        # metrics_aggregated["Wrongly_selected_cids"] = wrongly_selected
        if not bypass_clustering:
            metrics_aggregated["sm"] = sm
            metrics_aggregated["total_cids"] = len(cids)
            metrics_aggregated["actual_normal_cids"] = self.num_normal_selected
            metrics_aggregated["actual_malicious_cids"] = self.num_malicious_selected
            metrics_aggregated["pred_normal_cids"] = len(selected_clients)
            metrics_aggregated["pred_malicious_cids"] = len(dissimilar_cids)
            metrics_aggregated["no_wrongly_selected_cids"] = len(wrongly_selected)
            metrics_aggregated["selected_malicious_clients_percent"] = len(wrongly_selected)/len(selected_clients)
            metrics_aggregated["selected_benign_clients_percent"] = len(perfectly_selected)/len(selected_clients)
            all_metrics = self.compute_client_detection_metrics(
                total_clients=len(cids),
                actual_malicious=self.num_malicious_selected,
                actual_normal=self.num_normal_selected,
                selected_clients=len(selected_clients),
                rejected_clients=len(dissimilar_cids),
                wrongly_selected_malicious=len(wrongly_selected)
            )
            metrics_aggregated["all_metrics"] = all_metrics

        return parameters_aggregated, metrics_aggregated


    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        
        if self.fraction_evaluate == 0.0:
            return []
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        if server_round % self.attack_interval == 0:
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients, criterion=IncludeMalCidCriterion(self.mal_clients)
            )
        else:
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients, criterion=ExcludeMalCidCriterion(self.mal_clients)
            )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""

        if not results:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
            
        elif server_round == 1:  # Only log this warning once
            print("❌ WARNING: No evaluate_metrics_aggregation_fn provided")
        return loss_aggregated, metrics_aggregated


    def evaluate(self, server_round, parameters):
        model = self.model 
        ndarrays = parameters_to_ndarrays(parameters)

        # Ensure length matches
        model_keys = list(model.state_dict().keys())
        if len(ndarrays) != len(model_keys):
            raise ValueError(f"Mismatch in parameter sizes: got {len(ndarrays)}, expected {len(model_keys)}")

        # Create a state_dict from parameters
        state_dict = {k: torch.tensor(v) for k, v in zip(model_keys, ndarrays)}

        # Load into model
        model.load_state_dict(state_dict, strict=True)

        if self.evaluate_fn is not None:
            config = {}
            loss, metrics = self.evaluate_fn(server_round, parameters, config)
        else:
            loss, metrics = None, {}

        my_results = {"loss": loss, **metrics}
        self.results_to_save[server_round] = my_results
        with open("results.json", "w") as json_file:
            json.dump(self.results_to_save, json_file, indent=4)

        return loss, metrics
        



class ExcludeMalCidCriterion:
    def __init__(self, mal_clients):
        self.mal_clients = mal_clients
    def select(self, client_proxy):
        return int(client_proxy.cid) > self.mal_clients
    
class IncludeMalCidCriterion:
    def __init__(self, mal_clients):
        self.mal_clients = mal_clients  # Use a set for fast lookup

    def select(self, client_proxy):
        """Ensure all malicious clients are included, and others must have cid >= 10."""
        cid = int(client_proxy.cid)
        return cid < self.mal_clients or cid >= self.mal_clients 
    
    
class HardIncludeMalCidCriterion:
    def __init__(self, mal_clients: int, num_clients: int, percent_malicious: float, total_to_select: int):
        self.mal_clients = mal_clients
        self.num_clients = num_clients
        self.percent_malicious = percent_malicious
        self.total_to_select = total_to_select

        self.num_malicious_selected = int(self.percent_malicious * self.total_to_select)
        self.num_normal_selected = self.total_to_select - self.num_malicious_selected
        print(f'🔧 actual malicious_ratio : {self.percent_malicious} \n🔧 total_to_select : {self.total_to_select} \n🔧 num_mal_to_select : {self.num_malicious_selected} \n🔧 num_normal_to_select : {self.num_normal_selected}')
        self.selected_malicious = set(random.sample(range(self.mal_clients), min(self.num_malicious_selected, self.mal_clients)))
        self.selected_normal = set(random.sample(range(self.mal_clients, self.num_clients), min(self.num_normal_selected, self.num_clients - self.mal_clients)))

    def select(self, client_proxy):
        cid = int(client_proxy.cid)
        return cid in self.selected_malicious or cid in self.selected_normal

    
def create_strategy(**kwargs):
    # print("Received kwargs:", kwargs)
    on_fit_config_fn = kwargs.get("on_fit_config_fn", None)
    evaluate_metrics_aggregation_fn = kwargs.get("evaluate_metrics_aggregation_fn", None)
    evaluate_fn = kwargs.get("evaluate_fn", None)
    fit_metrics_aggregation_fn = kwargs.get("fit_metrics_aggregation_fn", None)
    if "model" not in kwargs or kwargs["model"] is None:
        raise ValueError("Model must be provided to create the strategy.")
    
    return CustomFedavg(**kwargs)



      
        # GMM clustering
        # gmm = GaussianMixture(
        #     n_components=2,
        #     covariance_type="full",
        #     reg_covar=1e-6,
        #     random_state=0,
        # )
        # cluster_labels = gmm.fit_predict(updates_2d)
        
        # from sklearn.cluster import DBSCAN

        # dbscan = DBSCAN(
        #     eps=0.3,          # radius of density
        #     min_samples=5,    # minimum points to form dense region
        #     metric="euclidean"
        # )

        # cluster_labels = dbscan.fit_predict(updates_2d)
        
        

        # clusterer = hdbscan.HDBSCAN(
        #     min_cluster_size=5,
        #     min_samples=3,
        #     metric="euclidean"
        # )

        # cluster_labels = clusterer.fit_predict(updates_2d)
        # probabilities = clusterer.probabilities_

        # loss_norm = (client_losses - client_losses.mean()) / (
        #     client_losses.std() + 1e-12
        # )
        # loss_norm = loss_norm.reshape(-1, 1)

        # X = np.hstack([updates_2d, loss_norm])  # shape: [N, 4]
        
        # from sklearn.svm import OneClassSVM

        # ocsvm = OneClassSVM(
        #     kernel="rbf",
        #     nu=0.1,  # expected attack fraction
        #     gamma="scale"
        # )

        # cluster_labels = ocsvm.fit_predict(X)
        # malicious_indices = np.where(cluster_labels == -1)[0]


    # def detect_malicious_clients_gac(
    #     self,
    #     client_updates: np.ndarray,
    #     client_losses: np.ndarray,
    #     cids: list,
    #     server_round: int,
    #     percent_malicious: float,
    #     n_components: int = 2,
    # ):
    #     """
    #     Identify malicious clients using PCA + clustering,
    #     and visualize PCA space with ground-truth coloring (cid-based),
    #     including a 3D PCA + loss plot.

    #     Returns:
    #         malicious_indices (List[int])
    #         cluster_labels (np.ndarray)
    #     """
    #     base_path = "/home/sailaja/Documents/p7_ntn/outputs/"
    #     tag = f"{int(percent_malicious * 100)}_{server_round}"

    #     import numpy as np
    #     import matplotlib.pyplot as plt
    #     from sklearn.decomposition import PCA
    #     from sklearn.cluster import KMeans
    #     from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    #     # -----------------------------
    #     # Safety checks
    #     # -----------------------------
    #     client_updates = np.asarray(client_updates, dtype=np.float64)
    #     client_losses = np.asarray(client_losses, dtype=np.float64)
    #     cids = np.array([int(cid) for cid in cids])

    #     assert (
    #         client_updates.shape[0]
    #         == client_losses.shape[0]
    #         == len(cids)
    #     ), "Mismatch in number of clients"

    #     # client_losses = np.nan_to_num(
    #     #     client_losses, nan=np.inf, posinf=np.inf, neginf=np.inf
    #     # )

    #     # # -----------------------------
    #     # # Step 1: PCA projection
    #     # # -----------------------------
    #     # pca = PCA(n_components=n_components, random_state=0)
    #     # updates_2d = pca.fit_transform(client_updates)

    #     # # -----------------------------
    #     # # Step 2: Clustering
    #     # # -----------------------------
    #     # kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
    #     # cluster_labels = kmeans.fit_predict(updates_2d)
    #     # centers = kmeans.cluster_centers_
  
    #     # cluster_0_idx = np.where(cluster_labels == 0)[0]
    #     # cluster_1_idx = np.where(cluster_labels == 1)[0]
        
        
        
    #     # -----------------------------
    #     # Sanitize losses
    #     # -----------------------------
    #     client_losses = np.nan_to_num(
    #         client_losses,
    #         nan=np.inf,
    #         posinf=np.inf,
    #         neginf=np.inf,
    #     )

    #     # Replace inf with large finite value (robust)
    #     finite_losses = client_losses[np.isfinite(client_losses)]
    #     max_finite = finite_losses.max() if finite_losses.size > 0 else 1.0
    #     client_losses = np.where(
    #         np.isfinite(client_losses),
    #         client_losses,
    #         max_finite * 1.5,
    #     )

    #     # -----------------------------
    #     # Step 1: PCA projection
    #     # -----------------------------
    #     pca = PCA(n_components=n_components, random_state=0)
    #     updates_2d = pca.fit_transform(client_updates)   # shape [N, 2]

    #     # -----------------------------
    #     # Step 2: Robust loss scaling
    #     # -----------------------------
    #     loss_scaled = np.clip(
    #         client_losses,
    #         np.percentile(client_losses, 5),
    #         np.percentile(client_losses, 95),
    #     ) 

    #     loss_scaled = StandardScaler().fit_transform(
    #         loss_scaled.reshape(-1, 1)
    #     ).squeeze()  # shape [N]

    #     # Optional: weight loss importance (tune if needed)
    #     loss_weight = 1.0   # try 0.5, 1.0, 2.0
    #     loss_scaled *= loss_weight

    #     # -----------------------------
    #     # Step 3: Joint feature space
    #     # -----------------------------
    #     # Feature vector = [PCA1, PCA2, scaled_loss]
    #     features = np.column_stack([updates_2d, loss_scaled])

    #     # -----------------------------
    #     # Step 4: KMeans clustering
    #     # -----------------------------
    #     kmeans = KMeans(n_clusters=2, n_init=20, random_state=0)
    #     cluster_labels = kmeans.fit_predict(features)
    #     centers = kmeans.cluster_centers_
    #     from numpy.linalg import norm

    #     radii = []
    #     for k in [0, 1]:
    #         idx = np.where(cluster_labels == k)[0]
    #         centers_2d = centers[:, :2]   # drop loss dimension

    #         distances = np.linalg.norm(
    #             updates_2d[idx] - centers_2d[k],
    #             axis=1
    #         )

    #         radii.append(distances.max())


    #     # centroid distance in PCA space
    #     centroid_distance = norm(centers_2d[0] - centers_2d[1])
    #     overlap_margin = (radii[0] + radii[1]) - centroid_distance

    #     print(f"Centroid distance (PCA): {centroid_distance:.4f}")
    #     print(f"R0 + R1: {radii[0] + radii[1]:.4f}")

       

    #     cluster_0_idx = np.where(cluster_labels == 0)[0]
    #     cluster_1_idx = np.where(cluster_labels == 1)[0]

        
    #     # from sklearn.metrics import silhouette_score

    #     # sil_score = silhouette_score(updates_2d, cluster_labels)
    #     # print(f"Silhouette score: {sil_score:.4f}")

    #     # # -----------------------------
    #     # # Distance between clusters
    #     # # -----------------------------
    #     # cluster_distance = np.linalg.norm(centers[0] - centers[1])

    #     # print(f"📏 Distance between cluster centroids: {cluster_distance:.4f}")

    #     # -----------------------------
    #     # Shared visualization setup
    #     # -----------------------------
    #     # colors = ["red" if cid < 40 else "green" for cid in cids]

    #     # # Robust loss scaling for size
    #     # loss_clip = np.clip(
    #     #     client_losses,
    #     #     np.percentile(client_losses, 5),
    #     #     np.percentile(client_losses, 95),
    #     # )
    #     # sizes = 40 + 120 * (loss_clip - loss_clip.min()) / (
    #     #     loss_clip.max() - loss_clip.min() + 1e-12
    #     # )


    #     # import matplotlib.pyplot as plt
    #     # from matplotlib.patches import Circle

    #     # plt.figure(figsize=(8, 7))

    #     # # scatter points
    #     # plt.scatter(
    #     #     updates_2d[:, 0],
    #     #     updates_2d[:, 1],
    #     #     c=colors,
    #     #     s=sizes,
    #     #     alpha=0.8,
    #     #     edgecolors="k",
    #     #     linewidths=0.4,
    #     # )

    #     # # draw circles
    #     # for k, color in zip([0, 1], ["blue", "orange"]):
    #     #     circle = Circle(
    #     #         centers[k],
    #     #         radii[k],
    #     #         fill=False,
    #     #         linewidth=2,
    #     #         linestyle="--",
    #     #         color=color,
    #     #         label=f"Cluster {k} radius"
    #     #     )
    #     #     plt.gca().add_patch(circle)

    #     # # centroids
    #     # plt.scatter(
    #     #     centers[:, 0],
    #     #     centers[:, 1],
    #     #     c="black",
    #     #     s=180,
    #     #     marker="X",
    #     #     label="Centroids",
    #     # )

    #     # plt.xlabel("PCA Component 1", fontsize=13)
    #     # plt.ylabel("PCA Component 2", fontsize=13)
    #     # plt.title("KMeans Clusters with Enclosing Circles", fontsize=14)
    #     # plt.legend()
    #     # plt.grid(alpha=0.3)
    #     # plt.tight_layout()

    #     # plt.savefig(f"{base_path}/kmeans_cluster_circles_{tag}.pdf", dpi=300)
    #     # plt.close()

    #     # if overlap_margin > 0:
    #     #     print(f"⚠️ Clusters OVERLAP by {overlap_margin:.4f}")
    #     # else:
    #     #     print(f"✅ Clusters SEPARATED by {-overlap_margin:.4f}")

    #     def safe_mean(arr):
    #         return float(np.mean(arr)) if arr.size > 0 else np.inf

    #     loss_0 = safe_mean(client_losses[cluster_0_idx])
    #     loss_1 = safe_mean(client_losses[cluster_1_idx])

    #     if loss_0 <= loss_1:
    #         benign_indices = cluster_0_idx
    #         malicious_indices = cluster_1_idx
    #     else:
    #         benign_indices = cluster_1_idx
    #         malicious_indices = cluster_0_idx

    #     benign_loss = safe_mean(client_losses[benign_indices])
    #     malicious_loss = safe_mean(client_losses[malicious_indices])

    #     print(
    #         f"[PCA+CLUSTER+LOSS] "
    #         f"Benign avg loss: {benign_loss:.4f} | "
    #         f"Malicious avg loss: {malicious_loss:.4f} | "
    #         f"Rejected: {len(malicious_indices)}"
    #     )

       
        
        
    #     # -----------------------------
    #     # 2D PCA plot (existing)
    #     # -----------------------------
    #     # plt.figure(figsize=(8, 7))
    #     # plt.scatter(
    #     #     updates_2d[:, 0],
    #     #     updates_2d[:, 1],
    #     #     c=colors,
    #     #     s=sizes,
    #     #     alpha=0.8,
    #     #     edgecolors="k",
    #     #     linewidths=0.5,
    #     # )

    #     # plt.xlabel("PCA Component 1", fontsize=14)
    #     # plt.ylabel("PCA Component 2", fontsize=14)
    #     # plt.title("PCA of Client Updates (Color = Ground Truth)", fontsize=15)

    #     # red_patch = plt.Line2D([0], [0], marker="o", color="w",
    #     #                     markerfacecolor="red", label="Malicious (cid < 40)",
    #     #                     markersize=10)
    #     # green_patch = plt.Line2D([0], [0], marker="o", color="w",
    #     #                         markerfacecolor="green", label="Benign (cid ≥ 40)",
    #     #                         markersize=10)
    #     # plt.legend(handles=[red_patch, green_patch], fontsize=12)

    #     # plt.grid(True, alpha=0.3)
    #     # plt.tight_layout()
    #     # plt.savefig(f"{base_path}/pca_2d_{tag}.pdf", dpi=300)
    #     # plt.close()

    #     # -----------------------------
    #     # 3D PCA + Loss plot (NEW)
    #     # -----------------------------
    #     # fig = plt.figure(figsize=(9, 7))
    #     # ax = fig.add_subplot(111, projection="3d")

    #     # ax.scatter(
    #     #     updates_2d[:, 0],
    #     #     updates_2d[:, 1],
    #     #     client_losses,
    #     #     c=colors,
    #     #     s=sizes,
    #     #     alpha=0.75,
    #     #     edgecolors="k",
    #     #     linewidths=0.3,
    #     # )

    #     # ax.set_xlabel("PCA Component 1", fontsize=12)
    #     # ax.set_ylabel("PCA Component 2", fontsize=12)
    #     # ax.set_zlabel("Client Loss", fontsize=12)
    #     # ax.set_title("3D PCA + Loss (Color = Ground Truth)", fontsize=14)

    #     # plt.tight_layout()
    #     # plt.savefig(f"{base_path}/pca_3d_loss_{tag}.pdf", dpi=300)
    #     # plt.close()

    #     return malicious_indices.tolist(), overlap_margin
    
    
    

    # def plot_grouped_update_distribution(self, updates, client_ids, server_round, max_clients=20):
    #     """
    #     updates: list of np.ndarray (flattened updates)
    #     """
    #     # Limit number of clients for readability
    #     path = (
    #         f"/home/sailaja/OneDrive/Code_Repository/paper4_DNRFL/outputs/"
    #         f"plot_grouped_update_distribution_{server_round}.pdf"
    #     )
    #     updates = updates[:max_clients]
    #     client_ids = client_ids[:max_clients]

    #     data = [np.abs(u) for u in updates]

    #     plt.figure(figsize=(16, 6))
    #     plt.boxplot(
    #         data,
    #         showfliers=False,
    #         patch_artist=True
    #     )
    #     plt.xticks(range(1, len(client_ids) + 1), client_ids, rotation=45)
    #     plt.ylabel("Absolute Update Magnitude")
    #     plt.xlabel("Client ID")
    #     plt.title("Grouped Distribution of Client Update Magnitudes")
    #     plt.tight_layout()
    #     plt.savefig(path, bbox_inches='tight')
    #     plt.close()

    # def plot_update_concentration(self, updates, client_ids, server_round, k_ratio=0.01):
    #     path = (
    #         f"/home/sailaja/OneDrive/Code_Repository/paper4_DNRFL/outputs/"
    #         f"plot_update_concentration_{server_round}.pdf"
    #     )
    #     concentrations = []

    #     for u in updates:
    #         k = int(len(u) * k_ratio)
    #         topk_norm = np.linalg.norm(np.sort(np.abs(u))[-k:])
    #         total_norm = np.linalg.norm(u)
    #         concentrations.append(topk_norm / (total_norm + 1e-12))

    #     plt.figure(figsize=(14, 4))
    #     plt.bar(client_ids, concentrations)
    #     plt.xticks(rotation=45)
    #     plt.ylabel("Top-k Concentration Ratio")
    #     plt.title("Client-wise Update Concentration")
    #     plt.tight_layout()
    #     plt.savefig(path, bbox_inches='tight')
    #     plt.close()

    # def plot_signed_histograms(self, updates, client_ids, server_round, max_clients=6):
    #     path = (
    #         f"/home/sailaja/OneDrive/Code_Repository/paper4_DNRFL/outputs/"
    #         f"plot_signed_histograms_{server_round}.pdf"
    #     )
    #     plt.figure(figsize=(18, 8))

    #     for i, (u, cid) in enumerate(zip(updates[:max_clients], client_ids[:max_clients])):
    #         plt.subplot(2, 3, i + 1)
    #         plt.hist(u, bins=1000, color='blue', alpha=0.7)
    #         plt.title(f"Client {cid}")
    #         plt.yscale("log")

    #     plt.suptitle("Signed Update Distributions (Log Scale)")
    #     plt.tight_layout()
    #     plt.savefig(path, bbox_inches='tight')
    #     plt.close()




    # def plot_client_gradients_2d(
    #     self,
    #     client_updates,
    #     cids,
    #     server_round,
    #     mal_clients,
    #     percent_malicious,
    #     normalize: bool = True,
    # ):
    #     """
    #     Plots client gradient vectors in 2D using PCA.

    #     Args:
    #         client_updates (List[np.ndarray]): List of flattened gradient vectors.
    #         cids (List[str or int]): Corresponding client IDs.
    #         server_round (int): Current FL round.
    #         normalize (bool): Whether to L2-normalize gradients before PCA.
    #     """
    #     path = (
    #         f"/home/sailaja/Documents/p7_ntn/outputs/"
    #         f"plot_grad_vecs_{int(percent_malicious*100)}_{server_round}.pdf"
    #     )
        
    #     if len(client_updates) < 2:
    #         print("⚠️ Not enough client updates to plot.")
    #         return

        

    #     X = np.vstack(client_updates)
    #     cids_np = np.array(cids, dtype=int)

    #     # Optional normalization (recommended for direction-based analysis)
    #     if normalize:
    #         X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    #     # PCA projection
    #     pca = PCA(n_components=2, random_state=42)
    #     X_2d = pca.fit_transform(X)

    #     # Color rule: cid < 50 -> red, else green
    #     colors = ["red" if cid < mal_clients else "green" for cid in cids_np]

    #     plt.figure(figsize=(7, 6))
    #     plt.scatter(
    #         X_2d[:, 0],
    #         X_2d[:, 1],
    #         c=colors,
    #         alpha=0.7,
    #         edgecolors="k",
    #         s=60,
    #     )

    #     plt.xlabel("PCA Component 1")
    #     plt.ylabel("PCA Component 2")
    #     plt.title(f"Client Gradient Distribution (Round {server_round})")

    #     # Legend
    #     red_patch = mpatches.Patch(color="red", label=f"cid < {self.mal_clients}")
    #     green_patch = mpatches.Patch(color="green", label=f"cid ≥ {self.mal_clients}")
    #     plt.legend(handles=[red_patch, green_patch])

    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(path, bbox_inches='tight')
    #     plt.close()