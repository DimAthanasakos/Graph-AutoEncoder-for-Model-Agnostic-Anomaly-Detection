
import json 
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
from sklearn.preprocessing import StandardScaler

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
from utils import load_jet_observables

import networkx
import energyflow as ef 
import random


class anomaly():
    def __init__(self, model_info, plot=False, plot_path='/global/homes/d/dimathan/gae_for_anomaly/plots_gae/plot_test', scaler = None) -> None:
        self.model_info = model_info
        self.path = ''
        if self.model_info['model'] not in  ['AE' ,'VAE']:
            self.path = model_info['path_SR'] # path to the data (pyg dataset)
        self.ddp = model_info['ddp']
        self.torch_device = model_info['torch_device']
        self.rank = 0
        self.plot_path = plot_path
        self.plot = plot 
        if self.ddp:
            if self.rank == 0:
                if not os.path.exists(self.plot_path):
                    os.makedirs(self.plot_path)
            dist.barrier()  # Ensure all processes wait until the directory is created
        else:
            if not os.path.exists(self.plot_path):
                os.makedirs(self.plot_path)
        self.scaler = scaler

        self.n_bkg = model_info['model_settings']['n_bkg']
        self.n_sig = model_info['model_settings']['n_sig']
        
        # parameters needed to just to set the path for the plots 
        self.n_train = model_info['n_train']
        self.n_part = model_info['n_part']
        self.epochs = self.model_info['model_settings']['epochs']
        self.learning_rate = self.model_info['model_settings']['learning_rate']
        self.input_dim = self.model_info['model_settings']['input_dim']
        

        self.plot_path = f'/global/homes/d/dimathan/gae_for_anomaly/plots_gae/0525/plot_n{self.n_part}_e{self.epochs}_lr{self.learning_rate}_N{self.n_train//1000}k'
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)

        self.unsupervised = model_info['model_settings']['unsupervised']
        self.lossname = self.model_info['model_settings']['lossname']
        self.batch_size = self.model_info['model_settings']['batch_size']

        if self.lossname == 'MSE': self.criterion = torch.nn.MSELoss()
        else: sys.exit(f'Error: loss {self.lossname} not recognized.')


        self.criterion_node = torch.nn.MSELoss(reduction='none')
        self.criterion_edge = torch.nn.MSELoss(reduction='none')

        self.data_loader = self.init_data()

    #---------------------------------------------------------------
    def init_data(self):
        print()
        print(f'ANOMALY')
        print(f'loading data from {self.path}')
        if self.model_info['model'] in ['AE', 'VAE']: 
            # Load observables from the sideband (or adjust use_SR as needed)
            obs, y = load_jet_observables(self.n_bkg + self.n_sig, use_SR=True)  # shape: (n_train, 2, 5)
            if self.model_info['model'] == 'VAE':
                obs, y = load_jet_observables(20000, use_SR=True)  # shape: (n_train, 2, 5)
                obs_bkg, y_bkg = obs[5000: 5000 + self.n_bkg], y[5000: 5000 + self.n_bkg]
                obs_sig, y_sig = obs[15000: 15000  + self.n_sig], y[15000: 15000 + self.n_sig]
                obs, y = np.concatenate([obs_bkg, obs_sig]), np.concatenate([y_bkg, y_sig])
            masses = obs[:, :, 0]
            sort_indices = np.argsort(-masses, axis=1)
            obs = np.take_along_axis(obs, sort_indices[:, :, np.newaxis], axis=1)
                

            # Flatten each event: 2 jets × 5 observables = 10 features per event.
            obs = obs[:, :, :self.input_dim]
            obs = obs.reshape(obs.shape[0], -1)
            # Normalize the data
            obs = self.scaler.transform(obs)

            # Create torch datasets and loaders.
            dataset = torch.utils.data.TensorDataset(torch.tensor(obs, dtype=torch.float), torch.tensor(y, dtype=torch.float))
            dataset = DataLoader(dataset, batch_size=128, shuffle=True)
        else: 
            dataset = torch.load(self.path, map_location='cpu', weights_only=False)
            if not self.unsupervised: # there is no overlap between dataset that we train on (SB) and the dataset we test on (SR)
                dataset_size = len(dataset) 
                dataset_bkg = dataset[:int(dataset_size * 0.5)]
                dataset_sig = dataset[int(dataset_size * 0.5):]

            elif self.unsupervised: # there is overlap between datasets 
                dataset_size = len(dataset) 
                for i in range(len(dataset)):
                    if dataset[i][0].y == 1:
                        first_signal_index = i
                        #print(f'first signal index: {first_signal_index}')
                        break

                dataset_bkg = dataset[:first_signal_index]
                dataset_sig = dataset[first_signal_index:]

            dataset_bkg = dataset_bkg[-self.n_bkg:]
            dataset_sig = dataset_sig[-self.n_sig:]

            print(f'Loaded testing (SR) dataset with {len(dataset_bkg)} bkg samples and {len(dataset_sig)} sig samples')
            print(f'-------------------')
            print()
            dataset = torch.utils.data.ConcatDataset([dataset_bkg, dataset_sig]) 

            dataset = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True) 

        return dataset
    
    #---------------------------------------------------------------
    def run(self, model):
        t_1 = time.time()
        if isinstance(model, tuple):
            model_0, model_1 = model
            model_0.eval()
            model_1.eval()
        else:
            model.eval()

        if self.model_info['model'] in ['VAE', 'EdgeNet_edge_VGAE']:
            kl_weight = self.model_info['model_settings']['kl_weight']
        else:  kl_weight = 0

        all_scores = []
        all_labels = []

        with torch.no_grad():
            for idx, batch in enumerate(self.data_loader):

                if self.model_info['model'] == 'AE': 
                    inputs = batch[0].to(self.torch_device)
                    input1 = inputs[:, :inputs.shape[1] // 2]
                    input2 = inputs[:, inputs.shape[1] // 2:]
                    recon1, _ = model(input1)
                    recon2, _ = model(input2)
                    loss1 = torch.mean((recon1 - input1) ** 2, dim=1)
                    loss2 = torch.mean((recon2 - input2) ** 2, dim=1)
                    loss_vals = loss1 + loss2
                    scores = loss_vals.cpu().tolist()
                    labels = batch[1].cpu().tolist()
                    if idx == -1: 
                        print(f'inputs[:5] = {inputs[:5]}')
                        print(f'recon1[:5] = {recon1[:5]}')
                        print(f'recon2[:5] = {recon2[:5]}')
                        print(f'scores[:5] = {scores[:5]}')
                        print(f'labels[:5] = {labels[:5]}')
                    #scores = inputs[:, 0].cpu().tolist()
                elif self.model_info['model'] == 'VAE': 
                    inputs = batch[0].to(self.torch_device)
                    labels = batch[1].cpu().tolist()
                    reconstruction, mu, logvar = model(inputs)
                    # Compute per-event MSE
                    loss_vals = torch.mean((reconstruction - inputs) ** 2, dim=1)
                    kl_score_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                    scores = (1/kl_score_per_sample).cpu().tolist() # Use raw KL divergence as score

                    #scores = (loss_vals + kl_weight * 1/kl_score_per_sample).cpu().tolist()
                    if not isinstance(scores, list):
                        scores = [scores]
                else: 
                    batch_jets0 = batch[0]
                    batch_jets1 = batch[1]
                    batch_jets0 = batch_jets0.to(self.torch_device)
                    batch_jets1 = batch_jets1.to(self.torch_device)

                    if self.input_dim == 4:
                        out1 = model(batch_jets0)
                        out2 = model(batch_jets1)
                        loss1_nodewise = self.criterion_node(out1, batch_jets0.match)
                        loss2_nodewise = self.criterion_node(out2, batch_jets1.match)
                        loss1_per_graph = scatter_mean(loss1_nodewise.mean(dim=-1), batch_jets0.batch, dim=0)
                        loss2_per_graph = scatter_mean(loss2_nodewise.mean(dim=-1), batch_jets1.batch, dim=0)
                        scores = loss1_per_graph + loss2_per_graph
                    
                    elif self.input_dim in [1, 3]:
                        if self.input_dim==1:
                            batch_jets0.x = batch_jets0.x[:, 0].unsqueeze(1)
                            batch_jets1.x = batch_jets1.x[:, 0].unsqueeze(1)
                            
                        target0 = batch_jets0.x
                        target1 = batch_jets1.x

                        if self.model_info['model'] == 'EdgeNet':
                            out1 = model(batch_jets0)
                            out2 = model(batch_jets1)
                            loss1_nodewise = self.criterion_node(out1, target0)
                            loss2_nodewise = self.criterion_node(out2, target1)
                            scores = scatter_mean(loss1_nodewise.mean(dim=-1), batch_jets0.batch, dim=0) + scatter_mean(loss2_nodewise.mean(dim=-1), batch_jets1.batch, dim=0)
                        
                        else:    
                            if self.model_info['model'] == 'EdgeNet_edge_VGAE':
                                out1, edge_out1, mu0, logvar0 = model(batch_jets0)
                                out2, edge_out2, mu1, logvar1 = model(batch_jets1)

                                #kl0 = -0.5 * torch.mean(torch.sum(1 + logvar0 - mu0.pow(2) - logvar0.exp(), dim=1))
                                kl0 = -0.5 * torch.sum(1 + logvar0 - mu0.pow(2) - logvar0.exp(), dim=1)
                                #kl1 = -0.5 * torch.mean(torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp(), dim=1))
                                kl1 = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp(), dim=1)

                            elif self.model_info['model'] in ['RelGAE']:
                                out1, edge_out1 = model(batch_jets0)
                                out2, edge_out2 = model(batch_jets1)


                            loss1_nodewise = self.criterion_node(out1, target0)
                            loss2_nodewise = self.criterion_node(out2, target1) 

                            loss1_per_node = loss1_nodewise.mean(dim=-1)
                            loss2_per_node = loss2_nodewise.mean(dim=-1)
                            loss1_per_graph = scatter_mean(loss1_per_node, batch_jets0.batch, dim=0)
                            loss2_per_graph = scatter_mean(loss2_per_node, batch_jets1.batch, dim=0)
                            
                            target_edge1 = batch_jets0.edge_attr
                            target_edge2 = batch_jets1.edge_attr
                            if self.model_info['model'] in ['EdgeNet_laman', 'Latent_GAE']:
                                target_edge1 = batch_jets0.fc_edge_attr
                                target_edge2 = batch_jets1.fc_edge_attr
                                
                            loss1_edgewise = self.criterion_edge(edge_out1, target_edge1)
                            loss2_edgewise = self.criterion_edge(edge_out2, target_edge2)
                            loss1_per_edge = loss1_edgewise.mean(dim=-1)
                            loss2_per_edge = loss2_edgewise.mean(dim=-1)

                            loss1_edge_per_graph = scatter_mean(loss1_per_edge, batch_jets0.batch[batch_jets0.edge_index[0]], dim=0)
                            loss2_edge_per_graph = scatter_mean(loss2_per_edge, batch_jets1.batch[batch_jets1.edge_index[0]], dim=0)

                            reconstruction_scores = loss1_per_graph + loss2_per_graph + loss1_edge_per_graph + loss2_edge_per_graph  
                            
                            if self.model_info['model'] == 'EdgeNet_edge_VGAE':
                                kl0_per_graph = scatter_mean(kl0, batch_jets0.batch, dim=0)
                                kl1_per_graph = scatter_mean(kl1, batch_jets1.batch, dim=0)

                                kl_loss = kl0_per_graph + kl1_per_graph
                                scores = reconstruction_scores + kl_weight * kl_loss
                                #scores = 1/kl_weight * kl_loss
                            else:
                                scores = reconstruction_scores

                all_scores.append(scores)
                all_labels.append(batch_jets0.y)


        all_scores_tensor = torch.cat(all_scores, dim=0)
        all_labels_tensor = torch.cat(all_labels, dim=0)
        all_scores = all_scores_tensor.cpu().tolist()
        all_labels = all_labels_tensor.cpu().tolist()
        
        # Compute ROC and AUC.
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        auc_val = auc(fpr, tpr)
        #print(f"Area Under Curve (AUC): {auc_val:.4f}")

        sic_y_values = np.full_like(tpr, fill_value=np.nan) # Initialize with NaN            
        # Calculate SIC where fpr > 0
        # Using np.errstate to suppress warnings for division by zero or invalid value,
        # as we handle these cases explicitly by checking the mask.
        with np.errstate(divide='ignore', invalid='ignore'):
            positive_fpr_mask = fpr > 0
            sic_y_values[positive_fpr_mask] = tpr[positive_fpr_mask] / np.sqrt(fpr[positive_fpr_mask])

        # Handle specific edge cases for SIC calculation:
        # Case: fpr = 0 and tpr = 0 (usually the start of ROC curve) -> SIC = 0
        sic_y_values[(fpr == 0) & (tpr == 0)] = 0
        # Case: fpr = 0 and tpr > 0 (perfect discrimination at that threshold) -> SIC = infinity
        sic_y_values[(fpr == 0) & (tpr > 0)] = np.inf

        sic_x_values = tpr # X-axis is Signal Efficiency (TPR)
        # calculate the max value of the SIC curve 
        # --- Calculate and print max *finite* SIC value ---
        finite_sic_mask = np.isfinite(sic_y_values)
        finite_sics = sic_y_values[finite_sic_mask]
        max_finite_sic = np.max(finite_sics) # nanmax not needed if already filtered by isfinite
        # Find the index corresponding to this max_finite_sic within the original sic_y_values
        # This requires careful indexing if multiple points have the same max finite value.
        # We'll take the first occurrence among finite values.
        original_indices_of_finite_sics = np.where(finite_sic_mask)[0]
        relative_max_idx = np.argmax(finite_sics)
        max_finite_sic_original_index = original_indices_of_finite_sics[relative_max_idx]
        
        max_finite_sic_tpr = sic_x_values[max_finite_sic_original_index]
        max_finite_sic_fpr = fpr[max_finite_sic_original_index]
        print(f"AUC: {auc_val:.4f}, Max *finite* SIC value: {max_finite_sic:.4f} at e_s: {max_finite_sic_tpr:.4f}, e_b: {max_finite_sic_fpr:.4f}")


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

            # --- Save all scores with labels to a single JSON file ---
            #plot_path = f'/global/homes/d/dimathan/gae_for_anomaly/txt_files/plot_n{self.n_part}_e{self.epochs}_N{self.n_train//1000}k'
            #if not os.path.exists(plot_path):
            #    os.makedirs(plot_path)

            all_scores_file = os.path.join(self.plot_path, "all_scores_with_labels.json")

            # Prepare data for JSON
            data_to_save = []
            for score in normal_scores:
                data_to_save.append({'score': score, 'label': 0})
            for score in anomalous_scores:
                data_to_save.append({'score': score, 'label': 1})

            with open(all_scores_file, 'w') as f:
                json.dump(data_to_save, f, indent=4) # Use json.dump to write the list of dictionaries
            print(f"Saved all scores with labels to: {all_scores_file}")


            # Compute the 99th percentile for both distributions
            p95_normal = np.percentile(normal_scores, 99)
            p95_anomalous = np.percentile(anomalous_scores, 99)

            # Determine the x-axis limit (max of the two 95th percentiles)
            x_max = max(p95_normal, p95_anomalous)

            # Define the bin edges based on [0, x_max]
            num_bins = 50  # Number of bins
            bin_edges = np.linspace(0, x_max, num_bins + 1)  # Create bins in the range [0, x_max]

            #all_scores_combined = normal_scores + anomalous_scores  # Combine all scores
            #bin_edges = np.histogram_bin_edges(all_scores_combined, bins=100)  # Compute bin edges

            dist_plot_file = os.path.join(self.plot_path, "loss_distribution_of_sig_vs_bkg.pdf")

            plt.figure(figsize=(6, 5))
            # density=True => each histogram integrates to 1, letting you compare shapes
            plt.hist(normal_scores, bins=bin_edges, label="Background (label=0)", color="green", histtype='step', density=True)
            plt.hist(anomalous_scores, bins=bin_edges, label="Signal (label=1)", color="red", histtype='step', density=True)

            plt.xlabel("Reconstruction Loss", fontsize=18)
            plt.xlim(0, x_max)
            plt.ylabel("Density", fontsize=14)
            #plt.title("Loss Distribution by Label (Normalized)")
            plt.legend(loc="best", fontsize=13, frameon=True)
            plt.tick_params(axis='both', which='major', labelsize=12)
            plt.grid()
            plt.tight_layout()
            plt.savefig(dist_plot_file, dpi=300)
            plt.close()
            print(f"Saved loss distribution plot to: {dist_plot_file}")


            # calculate the significance improvement curve and its max value 
            # SIC Y-axis: epsilon_S / sqrt(epsilon_B)  => tpr / sqrt(fpr)
            # SIC X-axis: epsilon_S => tpr
            sic_output_filename = os.path.join(self.plot_path, "SIC_curve.pdf")
            sic_y_values = np.full_like(tpr, fill_value=np.nan) # Initialize with NaN
            
            # Calculate SIC where fpr > 0
            # Using np.errstate to suppress warnings for division by zero or invalid value,
            # as we handle these cases explicitly by checking the mask.
            with np.errstate(divide='ignore', invalid='ignore'):
                positive_fpr_mask = fpr > 0
                sic_y_values[positive_fpr_mask] = tpr[positive_fpr_mask] / np.sqrt(fpr[positive_fpr_mask])

            # Handle specific edge cases for SIC calculation:
            # Case: fpr = 0 and tpr = 0 (usually the start of ROC curve) -> SIC = 0
            sic_y_values[(fpr == 0) & (tpr == 0)] = 0
            # Case: fpr = 0 and tpr > 0 (perfect discrimination at that threshold) -> SIC = infinity
            sic_y_values[(fpr == 0) & (tpr > 0)] = np.inf

            sic_x_values = tpr # X-axis is Signal Efficiency (TPR)

            plt.figure(figsize=(8, 6))

            # Filter out non-finite values for the main plot line to avoid display issues
            finite_mask = np.isfinite(sic_y_values)
            
            plt.plot(sic_x_values[finite_mask], sic_y_values[finite_mask], color='blue', lw=2, label=r'SIC ($\epsilon_S / \sqrt{\epsilon_B}$)')

            # Plot random classifier line for SIC: y = sqrt(x) where x is epsilon_S
            # This assumes that for a random classifier, epsilon_S = epsilon_B
            random_sic_x = np.linspace(0, 1, 100)
            random_sic_y = np.sqrt(random_sic_x)
            plt.plot(random_sic_x, random_sic_y, color='gray', lw=2, linestyle='--', label=r'Random ($\mathrm{SIC} = \sqrt{\epsilon_S}$ if $\epsilon_S=\epsilon_B$)')

            plt.xlim([0.0, 1.0])
            
            # Determine y-axis limits for SIC plot
            max_finite_sic_y = 0
            if np.any(finite_mask) and len(sic_y_values[finite_mask]) > 0:
                max_finite_sic_y = np.max(sic_y_values[finite_mask])
            
            if max_finite_sic_y > 0:
                plt.ylim([0.0, max_finite_sic_y * 1.15]) # Add 15% margin
            else:
                # If all finite values are 0 or negative (or no finite values), set a default sensible y_max
                # For SIC, values are non-negative. If max is 0, means no improvement.
                plt.ylim([0.0, 1.0]) # Default y_lim if no significant improvement

            plt.xlabel(r"Signal Efficiency ($\epsilon_S$)", fontsize=22)
            plt.ylabel(r"Significance ($\epsilon_S / \sqrt{\epsilon_B}$)", fontsize=24)
            plt.title("Significance Improvement Characteristic (SIC) Curve", fontsize=24)
            plt.legend(loc="upper right", fontsize=18, frameon=True)
            plt.tick_params(axis='both', which='major', labelsize=19)
            plt.grid(True)
            plt.tight_layout()

            plt.savefig(sic_output_filename)
            print(f"SIC curve saved to {sic_output_filename}")
            plt.close() # Close the figure

            # calculate the max value of the SIC curve
            max_sic = np.nanmax(sic_y_values)
            max_sic_index = np.nanargmax(sic_y_values)
            max_sic_x = sic_x_values[max_sic_index]

            print(f"Max SIC value: {max_sic:.4f} at Signal Efficiency (TPR): {max_sic_x:.4f}")
            # Save the max SIC value and the AUC to a text file
            max_sic_file = os.path.join(self.plot_path, "result.txt")
            with open(max_sic_file, 'w') as f:
                f.write(f"Max SIC value: {max_sic:.4f} at Signal Efficiency (TPR): {max_sic_x:.4f}\n")
                f.write(f"Area Under Curve (AUC): {auc_val:.4f}\n")
            print()
            print(f"Saved max SIC value and AUC to: {max_sic_file}")
            print(f'-----')

        return auc_val, max_finite_sic