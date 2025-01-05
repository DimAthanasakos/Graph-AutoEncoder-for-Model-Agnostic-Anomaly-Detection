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
import matplotlib.pyplot as plt

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch_geometric.data import Data, DataListLoader
from torch_geometric.loader import DataLoader #.data.DataLoader has been deprecated
from torch_scatter import scatter_mean

from models.models import EdgeNet, EdgeNetVAE, EdgeNet2, EdgeNetDeeper, GATAE
import ml_anomaly

import networkx
import energyflow as ef 
import random


class gae(): 
    def __init__(self, model_info, plot_path='/global/homes/d/dimathan/gae_for_anomaly/plots_gae/plot_test'):
        self.model_info = model_info
        self.path = model_info['path_SB'] # path to the data (pyg dataset)
        self.ddp = model_info['ddp']
        self.torch_device = model_info['torch_device']
        print(f'gae, device: {self.torch_device}')
        self.rank = 0
        self.plot_path = plot_path
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        if self.ddp: 
            self.rank = int(os.getenv("LOCAL_RANK"))
            # initialize the process group and set a timeout of 70 minutes, so that the process does not terminate
            # while rank=0 calculates the graph and the other ranks wait for the graph to be calculated
            dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(minutes=200)) 
            torch.cuda.set_device(self.rank)
            self.torch_device = torch.device('cuda', self.rank)
            if self.rank == 0:
                print()
                print('Running on multiple GPUs...')
                print()
                print('setting up DDP...')
                print("MASTER_ADDR:", os.getenv("MASTER_ADDR"))
                print("MASTER_PORT:", os.getenv("MASTER_PORT"))
                print("WORLD_SIZE:", os.getenv("WORLD_SIZE"))
                print("RANK:", os.getenv("RANK"))
                print("LOCAL_RANK:", os.getenv("LOCAL_RANK"))


        self.output_dir = model_info['output_dir']
        
        self.n_total = model_info['n_total']
        self.n_train = model_info['n_train']
        self.n_test = model_info['n_test']
        self.n_val = model_info['n_val'] 
        self.batch_size = self.model_info['model_settings']['batch_size']
        self.patience = self.model_info['model_settings']['patience']
        self.lossname = self.model_info['model_settings']['lossname']
        
        if self.lossname == 'MSE': self.criterion = torch.nn.MSELoss()
        else: sys.exit(f'Error: loss {self.lossname} not recognized.')


        # Use custon training parameters
        self.epochs = self.model_info['model_settings']['epochs']
        self.learning_rate = self.model_info['model_settings']['learning_rate']

        self.train_loader, self.val_loader, self.test_loader = self.init_data()

        self.model = self.init_model().to(self.torch_device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=self.patience, verbose=True)
        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank,)
            #self.batch_size = self.batch_size // torch.cuda.device_count()
            self.learning_rate = self.learning_rate * torch.cuda.device_count()

    #---------------------------------------------------------------
    def init_data(self):

        dataset = torch.load(self.path)

        #random.Random(0).shuffle(dataset)
        print(f'Loaded training (SB) dataset with {len(dataset)} samples')
        print()
        if self.rank==0 and self.n_total > len(dataset):
            print()
            print(f'==========================================')
            print(f'WARNING')
            print(f'Warning: n_total ({self.n_total}) is greater than the dataset length ({len(dataset)}).')
            print(f'==========================================')
            print()


        dataset = dataset[:self.n_total]
        print(f'Using {len(dataset)} samples for training, validation and testing')

        train_dataset  = dataset[:self.n_train]
        test_dataset   = dataset[self.n_train:self.n_train + self.n_test]
        valid_dataset  = dataset[self.n_train + self.n_test:]


        if self.ddp:
            self.train_sampler = DistributedSampler(train_dataset, shuffle=True)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler = self.train_sampler)
        else:
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader  = DataLoader(test_dataset,  batch_size=self.batch_size, shuffle=False)

        return train_loader, valid_loader, test_loader 


    #---------------------------------------------------------------
    def init_model(self):
        '''
        :return: pytorch architecture
        '''
        model_to_choose = self.model_info['model']
        if model_to_choose == 'EdgeNet':
            model = EdgeNet(input_dim=3, big_dim=32, hidden_dim=3, aggr='mean')
        elif model_to_choose == 'GATAE':
            model = GATAE(input_dim=3, hidden_dim=32, latent_dim=2, heads=2, dropout=0.0, add_skip=True)

        else: sys.exit(f'Error: model {model_to_choose} not recognized.') 

        # Print the model architecture if master process
        if self.rank == 0:
            print()
            print(model)
            print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')
            print()

        return model 
    
    #---------------------------------------------------------------
    def train(self):
        if self.rank == 0:
            print(f'Training...')
            print()
        best_val_loss = math.inf
        best_model_path = os.path.join(self.output_dir, 'model_best.pt')  # Path for saving the best model
        
        
        for epoch in range(1, self.epochs+1):
            if self.ddp: self.train_sampler.set_epoch(epoch)  

            t_start = time.time()
            loss_train = self._train_loop(ep=epoch)
            loss_val = self._test_loop(self.val_loader,)
            loss_test = self._test_loop(self.test_loader,)
            
            if epoch%4==0:
                AUC = ml_anomaly.anomaly(self.model, self.model_info, plot=False).run()
            # only print if its the main process
            if self.rank == 0:
                print(f'Epoch: {epoch:02d}, loss_train: {loss_train:.6f}, loss_val: {loss_val:.6f}, loss_test: {loss_test:.6f}, lr: {self.optimizer.param_groups[0]["lr"]:.6f}, Time: {time.time() - t_start:.1f} sec')
                print("--------------------------------")
                
            if loss_val < best_val_loss:
                best_val_loss = loss_val
                torch.save(self.model.state_dict(), best_model_path)
            
            self.scheduler.step(loss_val)


        # lets plot the loss distribution of the test set
        if self.rank == 0:            
            print(f'--------------------------------')
            print(f'Finished training')
            print()
            self._plot_loss()
        
        # Load the best model before returning
        self.model.load_state_dict(torch.load(best_model_path))  # Load best model's state_dict
        print("Loaded the best model based on validation loss.")
        loss_val = self._test_loop(self.val_loader,)
        loss_test = self._test_loop(self.test_loader,)
        print(f'Best model validation loss: {loss_val:.5f}')
        print(f'Best model test loss: {loss_test:.5f}')


        return self.model


    #---------------------------------------------------------------
    def _train_loop(self,ep=-1):
        self.model.train()
        loss_cum = 0
        count = 0
        for batch_idx, (batch_jets0, batch_jets1) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            length = len(batch_jets0)
            batch_jets0 = batch_jets0.to(self.torch_device)
            batch_jets1 = batch_jets1.to(self.torch_device)     

            out0 = self.model(batch_jets0)
            out1 = self.model(batch_jets1)
            #print(f'out0.shape: {out0.shape}')
            if batch_idx==0 and ep%1==0:
                n_print = min(2, length)
                values_to_match = torch.arange(0, n_print, device=batch_jets0.batch.device)  # Values from 0 to n_print-1
                mask0 = (batch_jets0.batch.unsqueeze(1) == values_to_match).any(dim=1)
                #mask0 = (batch_jets0.batch == 0) | (batch_jets0.batch == 1)
                in0 = batch_jets0.x[mask0].reshape(n_print, -1, 3)
                out0_reshaped = out0.reshape(-1, in0.shape[1], 3)

                l = torch.nn.MSELoss(reduction='none')(out0, batch_jets0.x)
                l = l.reshape(-1, in0.shape[1], 3)
                l_clamped = torch.clamp(l, min=-3, max=3)
                l_clamped = torch.round(l_clamped*1000)/1000

                # 'out0.x' has the node features after passing through the model.
                # 'batch_jets0.x' are the original input features for the same nodes.
                # We'll just print a few rows (.head equivalent) for clarity:
                print("=== First graph in the first batch of this epoch ===")
                print(f'in0[:2, :5]:' )
                print(in0[:2, :5])
                print()
                print("====================================================\n")
                print(f'out0_reshaped[:2,:5]:' )
                print(out0_reshaped[:2,:5])
                print()
                print("====================================================\n")
                print(f'l_clamped[:2, :5]:' )
                print(l_clamped[:2, :5])
                print()
                print("====================================================\n")

            loss0 = self.criterion(out0, batch_jets0.x)   # multiply by 100 to scale the loss
            loss1 = self.criterion(out1, batch_jets1.x)  
            loss = loss0 + loss1

            loss.backward()
            self.optimizer.step()
            loss_cum += loss.item() * length
            count += length
            # Cache management
            #torch.cuda.empty_cache()
        return loss_cum/count

    
    #---------------------------------------------------------------
    @torch.no_grad()
    def _test_loop(self, test_loader):
        self.model.eval()
        loss_cum = 0
        count = 0

        for batch_jets0, batch_jets1 in test_loader:
            length = len(batch_jets0)
            batch_jets0 = batch_jets0.to(self.torch_device)
            batch_jets1 = batch_jets1.to(self.torch_device)

            out0 = self.model(batch_jets0)
            out1 = self.model(batch_jets1)

            loss0 = self.criterion(out0, batch_jets0.x) * 100
            loss1 = self.criterion(out1, batch_jets1.x) * 100

            loss = loss0 + loss1

            loss_cum += loss.item() * length
            count += length
            # Cache management
            #torch.cuda.empty_cache()
            
        return loss_cum / count

    #---------------------------------------------------------------
    @torch.no_grad()
    def _plot_loss(self):
        self.model.eval()
        event_losses = []
        # A nodewise criterion
        criterion_node = torch.nn.MSELoss(reduction='none') # This 
        with torch.no_grad():
            for batch_jets0, batch_jets1 in self.test_loader:
                batch_jets0 = batch_jets0.to(self.torch_device)
                batch_jets1 = batch_jets1.to(self.torch_device)             

                out1 = self.model(batch_jets0)
                out2 = self.model(batch_jets1)

                # 1) Nodewise loss => shape [N, F]
                loss1_nodewise = criterion_node(out1, batch_jets0.x) * 100
                loss2_nodewise = criterion_node(out2, batch_jets1.x) * 100

                # 2) Average across features => shape [N]
                loss1_per_node = loss1_nodewise.mean(dim=-1)
                loss2_per_node = loss2_nodewise.mean(dim=-1)

                # 3) Aggregate nodewise losses by graph ID => shape [G], where G = number of graphs in the batch
                loss1_per_graph = scatter_mean(loss1_per_node, batch_jets0.batch, dim=0)
                loss2_per_graph = scatter_mean(loss2_per_node, batch_jets1.batch, dim=0)

                # 4) Compute the combined loss for each graph and append to event_losses
                scores = (loss1_per_graph + loss2_per_graph)  # Shape: [G]
                event_losses.extend(scores.cpu().tolist())  # Convert to list and extend

        plot_file = os.path.join(self.plot_path, 'val_loss_distribution.pdf')

        loss_tot  =  event_losses

        # Example: Compute quantiles for loss0_array and loss1_array
        quantiles_loss = {
            "50%": np.quantile(loss_tot, 0.5),  # 50% quantile (median)
            "70%": np.quantile(loss_tot, 0.7),  # 70% quantile
            "80%": np.quantile(loss_tot, 0.8),  # 80% quantile
            "90%": np.quantile(loss_tot, 0.9),  # 90% quantile
        }
        
        p99 = np.quantile(loss_tot, 0.99) # 99% quantile
        num_bins = 75
        bins = np.linspace(0, p99, num_bins) 
        print(f'--------------------------------')
        #print the average loss
        print(f"Average loss per event: {np.mean(loss_tot):.4f} ")
        print(f"Quantiles for the loss per jet: {quantiles_loss}")
        print()

        # Plot each distribution with a different color
        plt.figure(figsize=(8, 6))
        plt.hist(loss_tot, bins=bins, color='blue', histtype='step', label='Loss of test set (bkg)', density=True )
        #plt.hist(loss1_array, bins=100, color='red',  histtype='step', label='Loss 1')
        plt.xlabel('Loss')
        plt.ylabel('Counts')
        plt.title('Comparison of Two Loss Distributions')
        plt.grid()
        plt.legend()
        plt.tight_layout()

        plt.savefig(plot_file, dpi=300)
        plt.close()
        print(f"Saved loss distribution plot to: {plot_file}")
        print()

        return quantiles_loss
