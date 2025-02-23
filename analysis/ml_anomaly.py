import os
import time
import numpy as np
import math 
import sys
import glob

from datetime import timedelta

import socket 

import matplotlib.pyplot as plt
import sklearn
import scipy
from scipy.sparse import csr_matrix

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch_geometric.data import Data, DataListLoader
from torch_geometric.loader import DataLoader #.data.DataLoader has been deprecated
from torch_scatter import scatter_mean

from sklearn.metrics import roc_curve, auc

from models.models import EdgeNet, EdgeNetVAE, EdgeNet2, EdgeNetDeeper

import networkx
import energyflow as ef 
import random


class anomaly():
    def __init__(self, model_info, plot=True, plot_path='/global/homes/d/dimathan/gae_for_anomaly/plots_gae/plot_test') -> None:
        self.model_info = model_info
        self.path = model_info['path_SR'] # path to the data (pyg dataset)
        self.ddp = model_info['ddp']
        self.torch_device = model_info['torch_device']
        self.rank = 0
        self.plot_path = plot_path
        self.plot = plot 
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
            

        self.n_bkg = model_info['model_settings']['n_bkg']
        self.n_sig = model_info['model_settings']['n_sig']
        
        # parameters needed to just to set the path for the plots 
        self.n_train = model_info['n_train']
        self.n_part = model_info['n_part']
        self.epochs = self.model_info['model_settings']['epochs']
        self.learning_rate = self.model_info['model_settings']['learning_rate']
        self.input_dim = self.model_info['model_settings']['input_dim']
        

        self.plot_path = f'/global/homes/d/dimathan/gae_for_anomaly/plots_gae/plot_n{self.n_part}_e{self.epochs}_lr{self.learning_rate}_N{self.n_train//1000}k'
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)

        self.lossname = self.model_info['model_settings']['lossname']
        self.batch_size = self.model_info['model_settings']['batch_size']

        if self.lossname == 'MSE': self.criterion = torch.nn.MSELoss()
        else: sys.exit(f'Error: loss {self.lossname} not recognized.')


        self.data_loader = self.init_data()

    #---------------------------------------------------------------
    def init_data(self):

        dataset = torch.load(self.path)

        #random.Random(0).shuffle(dataset)
        #print(f'Loaded testing (SR) dataset with {len(dataset)} samples')
        dataset = dataset[:self.n_bkg + self.n_sig]
        dataset = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return dataset
    
    #---------------------------------------------------------------
    def run(self, model):
        model.eval()
        # A nodewise criterion
        criterion_node = torch.nn.MSELoss(reduction='none')
        criterion_edge = torch.nn.MSELoss(reduction='none')

        all_scores = []  # this will store the continuous anomaly score
        all_labels = []  # ground-truth anomaly labels (0 or 1)

        with torch.no_grad():
            for batch_jets0, batch_jets1 in self.data_loader:
                batch_jets0 = batch_jets0.to(self.torch_device)
                batch_jets1 = batch_jets1.to(self.torch_device)             
                if self.input_dim ==  4:
                    out1 = model(batch_jets0)
                    out2 = model(batch_jets1)

                    # 1) Nodewise loss => shape [N, F]
                    loss1_nodewise = criterion_node(out1, batch_jets0.match) 
                    loss2_nodewise = criterion_node(out2, batch_jets1.match)  

                    # 2) Average across features => shape [N]
                    loss1_per_node = loss1_nodewise.mean(dim=-1)
                    loss2_per_node = loss2_nodewise.mean(dim=-1)

                    # 3) Aggregate nodewise losses by graph ID => shape [G], where G = number of graphs in the batch
                    loss1_per_graph = scatter_mean(loss1_per_node, batch_jets0.batch, dim=0)
                    loss2_per_graph = scatter_mean(loss2_per_node, batch_jets1.batch, dim=0)
    
                    # 4) Compute the combined loss for each graph and append to event_losses
                    scores = (loss1_per_graph + loss2_per_graph)  # Shape: [G]

                elif self.input_dim in [1,3]:
                    if self.input_dim == 1:
                        target0 = batch_jets0.x[:,0].unsqueeze(1)
                        target1 = batch_jets1.x[:,0].unsqueeze(1)
                    elif self.input_dim == 3:
                        target0 = batch_jets0.x
                        target1 = batch_jets1.x

                    # Model now returns a tuple: (node_recon, edge_pred)
                    out1, edge_out1 = model(batch_jets0)
                    out2, edge_out2 = model(batch_jets1)

                    # 1) Compute node-wise loss (per node, per feature)
                    loss1_nodewise = criterion_node(out1, target0)  # shape: [N, F_node]
                    loss2_nodewise = criterion_node(out2, target1)  

                    # 2) Average across node feature dimensions => shape: [N]
                    loss1_per_node = loss1_nodewise.mean(dim=-1)
                    loss2_per_node = loss2_nodewise.mean(dim=-1)

                    # 3) Aggregate node losses by graph ID => shape: [G] (G = # graphs in batch)
                    loss1_per_graph = scatter_mean(loss1_per_node, batch_jets0.batch, dim=0)
                    loss2_per_graph = scatter_mean(loss2_per_node, batch_jets1.batch, dim=0)

                    # 4) Compute edge-wise loss
                    target_edge1 = batch_jets0.edge_attr
                    target_edge2 = batch_jets1.edge_attr
                    loss1_edgewise = criterion_edge(edge_out1, target_edge1)  # shape: [E, F_edge]
                    loss2_edgewise = criterion_edge(edge_out2, target_edge2)



                    # 5) Average across edge feature dimensions => shape: [E]
                    loss1_per_edge = loss1_edgewise.mean(dim=-1)
                    loss2_per_edge = loss2_edgewise.mean(dim=-1)

                    # 6) Determine graph membership for each edge.
                    #    We use the source node's batch id.
                    edge_batch1 = batch_jets0.batch[batch_jets0.edge_index[0]]  # shape: [E]
                    edge_batch2 = batch_jets1.batch[batch_jets1.edge_index[0]]

                    # 7) Aggregate edge losses by graph => shape: [G]
                    loss1_edge_per_graph = scatter_mean(loss1_per_edge, edge_batch1, dim=0)
                    loss2_edge_per_graph = scatter_mean(loss2_per_edge, edge_batch2, dim=0)

                    # 8) Combine node and edge losses per graph.
                    #     (Optionally add weighting factors, e.g. alpha*node_loss + beta*edge_loss)
                    scores = loss1_per_graph + loss2_per_graph + loss1_edge_per_graph + loss2_edge_per_graph


                all_scores.extend(scores.cpu().tolist())  # Convert to list and extend
                all_labels.extend(batch_jets0.y.cpu().tolist())  # or however your labels are stored
                
            # ----- Compute ROC and AUC -----
            fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
            auc_val = auc(fpr, tpr)

        
        print(f"Area Under Curve (AUC): {auc_val:.4f}")
        
        if self.plot:
            # ----- Plot the ROC curve -----
            plot_file = os.path.join(self.plot_path, "roc_curve.pdf")
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f'ROC (AUC = {auc_val:.3f})', color='b')
            plt.plot([0, 1], [0, 1], 'k--')  # diagonal line for "random" classification
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            # Add grid lines every 0.1
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            plt.xticks(np.arange(0, 1.1, 0.1))  # X-axis grid at 0.1 intervals
            plt.yticks(np.arange(0, 1.1, 0.1))  # Y-axis grid at 0.1 intervals
            plt.legend(loc="lower right")
            plt.tight_layout()

            # Save the plot
            plt.savefig(plot_file, dpi=300)
            plt.close()


            # ------------------------------------------------
            # 2) Plot loss distribution by label (normalized)
            # ------------------------------------------------
            normal_scores = [s for s, lbl in zip(all_scores, all_labels) if lbl == 0]
            anomalous_scores = [s for s, lbl in zip(all_scores, all_labels) if lbl == 1]


            # Compute the 99th percentile for both distributions
            p95_normal = np.percentile(normal_scores, 95)
            p95_anomalous = np.percentile(anomalous_scores, 95)

            # Determine the x-axis limit (max of the two 95th percentiles)
            x_max = max(p95_normal, p95_anomalous)

            # Define the bin edges based on [0, x_max]
            num_bins = 75  # Number of bins
            bin_edges = np.linspace(0, x_max, num_bins + 1)  # Create bins in the range [0, x_max]

            #all_scores_combined = normal_scores + anomalous_scores  # Combine all scores
            #bin_edges = np.histogram_bin_edges(all_scores_combined, bins=100)  # Compute bin edges

            dist_plot_file = os.path.join(self.plot_path, "loss_distribution_of_sig_vs_bkg.pdf")

            plt.figure(figsize=(6, 5))
            # density=True => each histogram integrates to 1, letting you compare shapes
            plt.hist(normal_scores, bins=bin_edges, label="Background (label=0)", color="green", histtype='step', density=True)
            plt.hist(anomalous_scores, bins=bin_edges, label="Signal (label=1)", color="red", histtype='step', density=True)

            plt.xlabel("Loss Score")
            plt.xlim(0, x_max)
            plt.ylabel("Density")
            plt.title("Loss Distribution by Label (Normalized)")
            plt.legend(loc="upper right")
            plt.grid()
            plt.tight_layout()
            plt.savefig(dist_plot_file, dpi=300)
            plt.close()
            print(f"Saved loss distribution plot to: {dist_plot_file}")
        return auc_val