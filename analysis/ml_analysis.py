import os
import sys
import yaml
import pickle
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import json 

import torch
import time 
import torch.nn as nn
from torch_geometric.data import Data, DataListLoader
import torch.distributed as dist

from torch_geometric.loader import DataLoader #.data.DataLoader has been deprecated

from torch_geometric.nn import EdgeConv, global_mean_pool, DataParallel
import random
sys.path.append('.')
from base import common_base
from models.models import EdgeNet, RelGAE
import gae_train, ml_anomaly



#torch.manual_seed(0)

################################################################
class MLAnalysis(common_base.CommonBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, config_file='', output_dir='', ddp=False, models = None, ext_plot=False, n_part=-1, input_dim=-1, n_runs = 1, graph_structures=[''], **kwargs):
        super(common_base.CommonBase, self).__init__(**kwargs)
        
        self.config_file = config_file
        self.output_dir = output_dir
        self.ddp = ddp  
        self.models = models
        self.ext_plot = ext_plot
        self.n_part = n_part
        self.input_dim = input_dim
        self.n_runs = n_runs
        self.graph_structures = graph_structures

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Initialize config file
        self.initialize_config()
        
        # Set torch device
        os.environ['TORCH'] = torch.__version__
        self.rank = int(os.getenv("LOCAL_RANK", "0"))
        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.rank == 0:
            print()
            print(f'pytorch version: {torch.__version__}')
            print('Using device:', self.torch_device)
            if self.torch_device.type == 'cuda':
                print(torch.cuda.get_device_name(0))
                print('Memory Usage:')
                print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
                print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
            print()
            print(self)
            print()


    #---------------------------------------------------------------
    # Initialize config file into class members
    #---------------------------------------------------------------
    def initialize_config(self):
    
        # Read config file
        with open(self.config_file, 'r') as stream:
          config = yaml.safe_load(stream)   
        
        self.n_train = config['n_train']
        self.n_val = config['n_val']
        self.n_total = self.n_train + self.n_val  
        self.val_frac  = 1. * self.n_val /  self.n_total

        # Initialize model-specific settings
        self.model_settings = {}
        for model in self.models:
            self.model_settings[model] = config[model]

    
    #---------------------------------------------------------------
    # Train models
    #---------------------------------------------------------------
    def train_models(self):

        self.AUC = defaultdict(list)
        self.val_loss = defaultdict(list)
        self.roc_curve_dict = self.recursive_defaultdict()

        for n_part in self.n_part:
            for model in self.models:
                if self.rank == 0:
                    print()
                    print(f'------------- Training model: {model} -------------')
                model_settings = self.model_settings[model]
                if model in ['EdgeNet', 'RelGAE', 'EdgeNet_edge_VGAE']: 
                    if self.graph_structures[0] == '': graph_structures = model_settings['graph_types'] 
                    else: graph_structures = self.graph_structures
                else: graph_structures = ['']
                
                model_info = {'model': model,
                            'model_settings': model_settings,
                            'n_part': n_part,
                            'n_total': self.n_total,
                            'n_train': self.n_train,
                            'n_val': self.n_val,
                            'input_dim': self.input_dim,
                            'graph_structures': graph_structures,
                            'torch_device': self.torch_device,
                            'output_dir': self.output_dir,
                            'ddp': self.ddp,
                            'ext_plot': self.ext_plot,}             


                #AUC = mdl.run_anomaly()
                if model in ['AE', 'VAE']: 
                    model_key = f'{model}'
                    model_info['model_key'] = model_key

                    all_train_losses = []
                    all_val_losses = []
                    all_aucs = []
                    all_aucs_max = []

                    for run in range(self.n_runs):
                        if self.rank==0: print(f"\n=== Run {run+1}/{self.n_runs} ===")
                        t_st = time.time()
                        analysis = gae_train.gae(model_info)
                        _, best_train_loss, best_val_loss, auc, auc_max, _ = analysis.train()
                        all_train_losses.append(best_train_loss)
                        all_val_losses.append(best_val_loss)
                        all_aucs.append(auc)
                        all_aucs_max.append(auc_max)
                        print(f'Run time: {time.time()-t_st:.2f} s\n')

                else:
                    model_key = f'{model}'
                    batch_size = model_info['model_settings']['batch_size']
                    n_total = model_info['n_total']
                    unsupervised = model_info['model_settings']['unsupervised'] if 'unsupervised' in model_info['model_settings']  else False

                    for graph_structure in graph_structures: 
                        regions = ['SB', 'SR']
                        for region in regions:
                            if graph_structure=='unique' and model in ['RelGAE', 'EdgeNet_edge_VGAE',]:
                                edge_addition = model_info['model_settings']['edge_addition']
                                graph_key = f'graphs_pyg_{region}__{graph_structure}_{edge_addition}_{n_part}{"_unsupervised" if unsupervised else ""}'
                            else:
                                graph_key = f'graphs_pyg_{region}__{graph_structure}_{n_part}{"_unsupervised" if unsupervised else ""}'
                            path = os.path.join(self.output_dir, f'{graph_key}.pt')
                            model_info[f'graph_key_{region}'] = graph_key
                            model_info[f'path_{region}'] = path
                            if self.rank==0:
                                print(f'graph_key_{region}: {graph_key}')
                                print(f'path_{region}: {path}')
                        all_train_losses = []
                        all_val_losses = []
                        all_aucs = []
                        all_aucs_max = []
                        all_max_sic = []

                        for run in range(self.n_runs):
                            if self.rank==0: print(f"\n=== Run {run+1}/{self.n_runs} ===")
                            t_st = time.time()
                            analysis = gae_train.gae(model_info)
                            _, best_train_loss, best_val_loss, auc, auc_max, max_sic = analysis.train()

                            print(f'rank:{self.rank} is here')
                            # --- Add cleanup HERE ---
                            print(f'dist.is_initialized: {dist.is_initialized()}') # Check state
                            if self.ddp:
                                print(f"Rank {self.rank}: Trying to destroy process group for run {run+1}") # Added print
                                dist.barrier()
                                print(f"Rank {self.rank}: Destroying process group for run {run+1}") # Added print
                                dist.destroy_process_group()
                                print(f"Rank {self.rank}: Process group destroyed for run {run+1}. is_initialized={dist.is_initialized()}") # Check state
                                time.sleep(2) # Add a 5-second pause

                            if self.rank==0:
                                all_train_losses.append(best_train_loss)
                                all_val_losses.append(best_val_loss)
                                all_aucs.append(auc)
                                all_aucs_max.append(auc_max)
                                all_max_sic.append(max_sic)
                                print(f'Run time: {time.time()-t_st:.2f} s\n')

                    # Compute averages and standard deviations across runs.
                    if self.rank==0:
                        avg_train, std_train = np.round(np.mean(all_train_losses), 4), np.round(np.std(all_train_losses), 4)
                        avg_val, std_val     = np.round(np.mean(all_val_losses), 4), np.round(np.std(all_val_losses), 4)
                        avg_auc, std_auc     = np.round(np.mean(all_aucs), 4), np.round(np.std(all_aucs), 4)
                        avg_auc_max, std_auc_max = np.round(np.mean(all_aucs_max), 4), np.round(np.std(all_aucs_max), 4)
                        avg_max_sic, std_max_sic = np.round(np.mean(all_max_sic), 4), np.round(np.std(all_max_sic), 4)
                        self.val_loss[n_part]=[avg_val, std_val]
                        self.AUC[n_part]=[avg_auc, std_auc]
                        summary = {
                                    "n_part": n_part,
                                    "Train Loss": [avg_train, std_train],
                                    "Val Loss": [avg_val, std_val],
                                    "Val_loss_list": all_val_losses,
                                    "AUC": [avg_auc, std_auc],
                                    "max_sic": [avg_max_sic, std_max_sic],
                                    "AUC_max": [avg_auc_max, std_auc_max],
                                    "AUC_list": all_aucs,
                                    "max_sic_list": all_max_sic,
                                    "AUC_max_list": all_aucs_max,
                                }
                        if self.rank==0:
                            print(json.dumps(summary))
                            
                            print(f"\n=== Summary over runs for n_part: {n_part} === ")
                            print(f"Train Loss: mean = {avg_train:.4f}, std = {std_train:.4f}")
                            print(f"Val Loss:   mean = {avg_val:.4f}, std = {std_val:.4f}")
                            print(f"AUC:        mean = {avg_auc:.4f}, std = {std_auc:.4f}")
                            print(f"max_sic:    mean = {avg_max_sic:.4f}, std = {std_max_sic:.4f}")
                            print(f'=.'*30, '\n \n')
        if self.rank==0:                
            print(f'val_loss: {self.val_loss}')
            print(f'AUC: {self.AUC}') 
            if len(self.n_part) > 1: self.plot_results()
        

    #---------------------------------------------------------------
    # plot results vs n_part
    #---------------------------------------------------------------
    def plot_results(self):
        """
        Create and save three plots:
        1. AUC vs n_part.
        2. Validation Loss vs n_part.
        3. A combined plot showing both AUC and Validation Loss vs n_part.
        
        The plots are saved in /global/homes/d/dimathan/gae_for_anomaly/plots_gae
        with filenames 'large_run_auc.pdf', 'large_run_loss.pdf', and 'large_run_combined.pdf'.
        """

        # Define the output directory.
        out_path = f'/global/homes/d/dimathan/gae_for_anomaly/plots_gae/extended_run_ntot{self.n_total}'
        if not os.path.exists(out_path): os.makedirs(out_path)
        print(f'Saving extended plots in {out_path}')

        # Sort the n_part keys (assumed to be numeric) so the x-axis is ordered.
        n_parts = sorted(self.AUC.keys())

        # For each n_part, extract the list of [avg, std] values, then compute overall averages.
        auc_means = []
        auc_stds = []
        for n in n_parts:
            # self.AUC[n] is a list of lists: each inner list is [avg_auc, std_auc] from one run.
            arr = np.array(self.AUC[n])  # shape: (num_runs, 2)
            mean_auc, std_auc = arr[0], arr[1]
            auc_means.append(mean_auc)
            auc_stds.append(std_auc)

        loss_means = []
        loss_stds = []
        for n in n_parts:
            # self.val_loss[n] is a list of lists: each inner list is [avg_val, std_val] from one run.
            arr = np.array(self.val_loss[n])  # shape: (num_runs, 2)
            mean_loss, std_loss = arr[0], arr[1]
            loss_means.append(mean_loss)
            loss_stds.append(std_loss)

        # Plot 1: AUC vs n_part
        plt.figure(figsize=(8, 6))
        plt.errorbar(n_parts, auc_means, yerr=auc_stds, fmt='o-', capsize=5, label='AUC')
        plt.xlabel('n_part')
        plt.ylabel('AUC')
        plt.title('AUC vs n_part')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, 'large_run_auc.pdf'))
        plt.close()

        # Plot 2: Validation Loss vs n_part
        plt.figure(figsize=(8, 6))
        plt.errorbar(n_parts, loss_means, yerr=loss_stds, fmt='s-', capsize=5, label='Validation Loss')
        plt.xlabel('n_part')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss vs n_part')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, 'large_run_loss.pdf'))
        plt.close()

        # Plot 3: Combined Plot (Dual y-axis): AUC and Validation Loss vs n_part
        fig, ax1 = plt.subplots(figsize=(8, 6))
        color1 = 'tab:blue'
        ax1.set_xlabel('n_part')
        ax1.set_ylabel('Validation Loss', color=color1)
        ax1.errorbar(n_parts, loss_means, yerr=loss_stds, fmt='s-', capsize=5, color=color1, label='Validation Loss')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True)
        
        ax2 = ax1.twinx()
        color2 = 'tab:green'
        ax2.set_ylabel('AUC', color=color2)
        ax2.errorbar(n_parts, auc_means, yerr=auc_stds, fmt='o-', capsize=5, color=color2, label='AUC')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        fig.tight_layout()
        plt.title('AUC and Validation Loss vs n_part')
        # Combine legends from both axes.
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.savefig(os.path.join(out_path, 'large_run_combined.pdf'))
        plt.close()


