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
    def __init__(self, model, model_info, plot=True, plot_path='/global/homes/d/dimathan/gae_for_anomaly/plots_gae/plot_test') -> None:
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
        self.lossname = self.model_info['model_settings']['lossname']
        self.batch_size = self.model_info['model_settings']['batch_size']

        if self.lossname == 'MSE': self.criterion = torch.nn.MSELoss()
        else: sys.exit(f'Error: loss {self.lossname} not recognized.')


        self.data_loader = self.init_data()
        self.model = model

    #---------------------------------------------------------------
    def init_data(self):

        dataset = torch.load(self.path)

        #random.Random(0).shuffle(dataset)
        #print(f'Loaded testing (SR) dataset with {len(dataset)} samples')
        dataset = dataset[:self.n_bkg + self.n_sig]
        dataset = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return dataset
    
    #---------------------------------------------------------------
    def run(self):
        self.model.eval()
        # A nodewise criterion
        criterion_node = torch.nn.MSELoss(reduction='none')

        all_scores = []  # this will store the continuous anomaly score
        all_labels = []  # ground-truth anomaly labels (0 or 1)

        with torch.no_grad():
            for batch_jets0, batch_jets1 in self.data_loader:
                batch_jets0 = batch_jets0.to(self.torch_device)
                batch_jets1 = batch_jets1.to(self.torch_device)             

                out1 = self.model(batch_jets0)
                out2 = self.model(batch_jets1)

                # 1) Nodewise loss => shape [N, F]
                loss1_nodewise = criterion_node(out1, batch_jets0.x)
                loss2_nodewise = criterion_node(out2, batch_jets1.x)

                # 2) Average across features => shape [N]
                loss1_per_node = loss1_nodewise.mean(dim=-1)
                loss2_per_node = loss2_nodewise.mean(dim=-1)

                # 3) Aggregate nodewise losses by graph ID => shape [G], where G = number of graphs in the batch
                loss1_per_graph = scatter_mean(loss1_per_node, batch_jets0.batch, dim=0)
                loss2_per_graph = scatter_mean(loss2_per_node, batch_jets1.batch, dim=0)

                # 4) If you want "both jets must exceed threshold" logic as a *score*,
                #    you can combine them somehow (e.g. take min)
                #scores = torch.min(loss1_per_graph, loss2_per_graph)

                # if we want the total loss as the score 
                scores = loss1_per_graph + loss2_per_graph
                
                # 5) Extend the scores and labels
                #    Typically, each graph in batch_jets0 has a single label in batch_jets0.y, shape [G].
                all_scores.extend(scores.cpu().tolist())
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
            p99_normal = np.percentile(normal_scores, 95)
            p99_anomalous = np.percentile(anomalous_scores, 95)

            # Determine the x-axis limit (max of the two 95th percentiles)
            x_max = max(p99_normal, p99_anomalous)

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