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

from models.models import EdgeNet, EdgeNetVAE, EdgeNet2, EdgeNetDeeper, GATAE, EdgeNet_edge, StackedEdgeNet_edge, HybridEdgeNet
import ml_anomaly

import networkx
import energyflow as ef 
import random

import h5py 


class gae(): 
    def __init__(self, model_info, plot_path='/global/homes/d/dimathan/gae_for_anomaly/plots_gae/plot_test'):
        self.model_info = model_info
        self.path = model_info['path_SB'] # path to the data (pyg dataset)
        self.ddp = model_info['ddp']
        self.torch_device = model_info['torch_device']
        print(f'gae, device: {self.torch_device}')
        self.rank = 0
        self.plot_path = plot_path
        self.ext_plot = model_info['ext_plot']
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
        self.n_part = model_info['n_part']
        self.batch_size = self.model_info['model_settings']['batch_size']
        self.patience = self.model_info['model_settings']['patience']
        self.lossname = self.model_info['model_settings']['lossname']
        self.input_dim = self.model_info['model_settings']['input_dim']
        self.pair_input_dim = self.model_info['model_settings']['pair_input_dim']
        
        if self.lossname == 'MSE': self.criterion = torch.nn.MSELoss()
        else: sys.exit(f'Error: loss {self.lossname} not recognized.')


        # Use custon training parameters
        self.epochs = self.model_info['model_settings']['epochs']
        self.learning_rate = self.model_info['model_settings']['learning_rate']
        
        if self.epochs > 20 and self.n_train > 10000: # otherwise its just testing, no need to store the plots on a permanent folder
            self.plot_path = f'/global/homes/d/dimathan/gae_for_anomaly/plots_gae/plot_n{self.n_part}_e{self.epochs}_lr{self.learning_rate}_N{self.n_train//1000}k'
            if not os.path.exists(self.plot_path):
                os.makedirs(self.plot_path)

        self.train_loader, self.val_loader, self.test_loader = self.init_data()

        self.model = self.init_model().to(self.torch_device)
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)
        #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.33, patience=self.patience, verbose=True)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,  # Initial learning rate (e.g., 1e-3)
            weight_decay=0.01  # Regularization to reduce overfitting
        )

        # Cosine Annealing Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(self.train_loader) * self.epochs,  # Total number of iterations (epochs x steps per epoch)
            eta_min=1e-6  # Minimum learning rate at the end of annealing
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,  # Initial learning rate (use a smaller value, e.g., 1e-4 or 5e-5)
            weight_decay=0.01  # Regularization to reduce overfitting
        )

        # Scheduler
        num_training_steps = len(self.train_loader) * self.epochs  # Total number of training steps
        num_warmup_steps = int(0.05 * num_training_steps)  # Warmup for 20% of training steps

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,  # Peak learning rate during warmup
            total_steps=num_training_steps,
            anneal_strategy='linear',  # Linearly decay the learning rate after warmup
            pct_start=num_warmup_steps / num_training_steps,  # Proportion of warmup steps
            div_factor=5.0,  # Initial LR is 1/10th of max_lr
            final_div_factor=2.0 ) # Final LR is 1/5th of max_lr
        


        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank,)
            #self.batch_size = self.batch_size // torch.cuda.device_count()
            self.learning_rate = self.learning_rate * torch.cuda.device_count()

    #---------------------------------------------------------------
    def init_data(self):

        dataset = torch.load(self.path)

        #random.Random(0).shuffle(dataset)
        print(f'Loaded training (SB) dataset with {len(dataset)} samples\n')
        if self.n_total > len(dataset):
            print(f'\n==========================================')
            print(f'Warning: n_total ({self.n_total}) is greater than the dataset length ({len(dataset)}).')
            print(f'==========================================\n')


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
            model = EdgeNet(input_dim=self.input_dim, big_dim=32, latent_dim=2, dropout_rate=0.1)
        elif model_to_choose == 'EdgeNet_edge':
            model = EdgeNet_edge(input_dim=self.input_dim, edge_dim=self.pair_input_dim, big_dim=32, latent_dim=3, output_dim = self.input_dim, dropout_rate=0.0)
            #model = StackedEdgeNet_edge(input_dim=self.input_dim, edge_dim=3, hidden_dim=32, latent_dim=64, output_dim=3,
            #                    num_encoder_layers=3, num_decoder_layers=3, aggr='mean', dropout_rate=0.1)
        elif model_to_choose == 'HybridEdgeNet':
            model = HybridEdgeNet(input_dim=self.input_dim, edge_dim=3, hidden_dim=32, latent_dim=2, 
                          output_dim=3, num_encoder_layers=3, aggr='mean', dropout_rate=0.)
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
    def train(self,):
        if self.rank == 0:
            print(f'Training...')
            print()
        best_val_loss = math.inf
        best_model_path = os.path.join(self.output_dir, 'model_best.pt')  # Path for saving the best model
        
        self.auc_list = []
        self.train_loss_list = []
        self.val_loss_list = []

        anomaly_model = ml_anomaly.anomaly(self.model_info, plot=False)
        for epoch in range(1, self.epochs+1):
            if self.ddp: self.train_sampler.set_epoch(epoch)  

            t_start = time.time()
            loss_train, l_nodes, l_edges = self._train_loop(ep=epoch)
            loss_val = self._test_loop(self.val_loader,)
            loss_test = self._test_loop(self.test_loader,)
            
            if epoch%1==0 or self.ext_plot:
                auc = anomaly_model.run(model=self.model)
                #actual_train_loss = self._test_loop(self.train_loader,)
                #print(f'actual_train_loss={actual_train_loss:.5f}')
                self.auc_list.append(auc)
                #self.train_loss_list.append(actual_train_loss)
                self.val_loss_list.append(loss_val)


            print(f'Epoch: {epoch:02d}, loss_train: {loss_train:.5f}, loss_val: {loss_val:.5f}, loss_test: {loss_test:.5f}, lr: {self.optimizer.param_groups[0]["lr"]:.5f}, Time: {time.time() - t_start:.1f} sec')
            #print(f'Node loss: {l_nodes:.5f}, Edge loss: {l_edges:.5f}')
            print("--------------------------------")
                
            if loss_val < best_val_loss:
                best_val_loss = loss_val
                torch.save(self.model.state_dict(), best_model_path)
            

        # lets plot the loss distribution of the test set
        print(f'--------------------------------')
        print(f'Finished training\n')
        
        # Load the best model before evaluating final metrics.
        self.model.load_state_dict(torch.load(best_model_path))
        print("Loaded the best model based on validation loss.")

        best_train_loss = self._test_loop(self.train_loader)
        best_val_loss  = self._test_loop(self.val_loader)
        best_test_loss = self._test_loop(self.test_loader)
        anomaly_model = ml_anomaly.anomaly(self.model_info, plot=True)
        auc = anomaly_model.run(model=self.model)
        #auc = anomaly_model.run(model=self.model, plot=True)
        self._plot_loss()

        print(f'Best model train loss: {best_train_loss:.5f}')
        print(f'Best model validation loss: {best_val_loss:.5f}')
        print(f'Best model test loss: {best_test_loss:.5f}')
        print(f'AUC: {auc:.5f}')
        print(f'--------------------------------\n')
        # Return the best model and the metrics for this run.
        return self.model, best_train_loss, best_val_loss, best_test_loss, auc


    #---------------------------------------------------------------
    def _train_loop(self,ep=-1):
        self.model.train()
        loss_cum = 0
        count = 0
        loss_cum_nodes, loss_cum_edges = 0, 0
        for batch_idx, (batch_jets0, batch_jets1) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            length = len(batch_jets0)
            batch_jets0, batch_jets1 = batch_jets0.to(self.torch_device), batch_jets1.to(self.torch_device)

            # Unpack outputs: node reconstructions and predicted edge attributes.
            if self.input_dim == 4:
                out0 = self.model(batch_jets0)
                out1 = self.model(batch_jets1)

                loss_node0 = self.criterion(out0, batch_jets0.match)
                loss_node1 = self.criterion(out1, batch_jets1.match)
                loss = loss_node0 + loss_node1
                

            elif self.input_dim in [1,3]:
                if self.input_dim==1:
                    target0 = batch_jets0.x[:,0].unsqueeze(1)
                    target1 = batch_jets1.x[:,0].unsqueeze(1)
                elif self.input_dim==3:
                    target0 = batch_jets0.x
                    target1 = batch_jets1.x
                

                out0, edge_out0 = self.model(batch_jets0)
                out1, edge_out1 = self.model(batch_jets1)
        
                # Node reconstruction loss.
                loss_node0 = self.criterion(out0, target0)
                loss_node1 = self.criterion(out1, target1)

                # Edge reconstruction loss.
                # batch_jets0.edge_attr: shape [n_edges * batch_size * 2, F_edge]
                target_edge0 = batch_jets0.edge_attr 
                target_edge1 = batch_jets1.edge_attr 
                loss_edge0 = self.criterion(edge_out0, target_edge0)
                loss_edge1 = self.criterion(edge_out1, target_edge1)
                
                # Combine losses (you may weight them if necessary).
                loss = loss_node0 + loss_node1 + loss_edge0 + loss_edge1
                loss_nodes = loss_node0.item() + loss_node1.item()
                loss_edges = loss_edge0.item() + loss_edge1.item()
            

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            loss_cum += loss.item() * length
            loss_cum_nodes += loss_nodes * length
            loss_cum_edges += loss_edges * length
            count += length
            
        return loss_cum/count, loss_cum_nodes/count, loss_cum_edges/count

    
    #---------------------------------------------------------------
    @torch.no_grad()
    def _test_loop(self, test_loader):
        self.model.eval()
        loss_cum = 0
        count = 0

        for batch_jets0, batch_jets1 in test_loader:
            length = len(batch_jets0)
            batch_jets0, batch_jets1 = batch_jets0.to(self.torch_device) , batch_jets1.to(self.torch_device)
            # Unpack outputs: node reconstructions and predicted edge attributes.
            if self.input_dim == 4:
                out0 = self.model(batch_jets0)
                out1 = self.model(batch_jets1)

                loss_node0 = self.criterion(out0, batch_jets0.match)
                loss_node1 = self.criterion(out1, batch_jets1.match)
                loss = loss_node0 + loss_node1
                

            elif self.input_dim in [1,3]:
                if self.input_dim==1:
                    target0 = batch_jets0.x[:,0].unsqueeze(1)
                    target1 = batch_jets1.x[:,0].unsqueeze(1)
                elif self.input_dim==3:
                    target0 = batch_jets0.x
                    target1 = batch_jets1.x
                

                out0, edge_out0 = self.model(batch_jets0)
                out1, edge_out1 = self.model(batch_jets1)
        
                # Node reconstruction loss.
                loss_node0 = self.criterion(out0, target0)
                loss_node1 = self.criterion(out1, target1)
                
                # Edge reconstruction loss.
                target_edge0 = batch_jets0.edge_attr 
                target_edge1 = batch_jets1.edge_attr 
                loss_edge0 = self.criterion(edge_out0, target_edge0)
                loss_edge1 = self.criterion(edge_out1, target_edge1)
                
                # Combine losses (you may weight them if necessary).
                loss = loss_node0 + loss_node1 + loss_edge0 + loss_edge1

            loss_cum += loss.item() * length
            count += length
            
        return loss_cum / count

    #---------------------------------------------------------------
    @torch.no_grad()
    def _plot_loss(self, extended_plots=False):
        self.model.eval()
        event_losses = []
        # A nodewise criterion
        criterion_node = torch.nn.MSELoss(reduction='none')  
        criterion_edge = torch.nn.MSELoss(reduction='none')

        with torch.no_grad():
            for batch_jets0, batch_jets1 in self.test_loader:
                batch_jets0 = batch_jets0.to(self.torch_device)
                batch_jets1 = batch_jets1.to(self.torch_device)             
                if self.input_dim == 4:
                    out1 = self.model(batch_jets0)
                    out2 = self.model(batch_jets1)

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
                    out1, edge_out1 = self.model(batch_jets0)
                    out2, edge_out2 = self.model(batch_jets1)

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
        #print(f'--------------------------------')
        #print the average loss
        #print(f"Average loss per event: {np.mean(loss_tot):.4f} ")
        #print(f"Quantiles for the loss per jet: {quantiles_loss}")
        #print()

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

        if self.ext_plot:
            print(f'--------------------------------')
            print(f'Plotting extended plots...')
            print()
            
            # Plot the loss distribution of train and val with the AUC for each epoch
            plot_file = os.path.join(self.plot_path, 'auc_vs_loss_distribution.pdf')
            fig, ax1 = plt.subplots(figsize=(8, 6))

            # Plot the loss distributions
            epochs = range(1, len(self.auc_list) + 1)
            ax1.plot(epochs, self.val_loss_list, color='blue', label='Val Loss', marker='o', linestyle='-')
            ax1.plot(epochs, self.train_loss_list, color='red', label='Train Loss', marker='o', linestyle='-')
            ax1.set_ylabel('Loss', color='black', fontsize='large')
            ax1.set_xlabel('epochs', fontsize='large')
            ax1.set_title(f'Loss Distribution and AUC, GCN, n_part = {self.n_part}', fontsize='x-large')
            ax1.set_ylim(0.009, 0.07)
            ax1.grid()
            

            # Add the secondary y-axis for AUC
            ax2 = ax1.twinx()
            print(f'self.auc_list: {self.auc_list}')
            ax2.plot(epochs, self.auc_list, color='darkgreen', label='AUC', marker='o', linestyle='-')
            ax2.set_ylabel('AUC', color='black', fontsize='large')
            ax2.tick_params(axis='y', labelcolor='black', labelsize='medium')
            # Scale the AUC axis to the range 0.5 to 1
            ax2.set_ylim(0.775, 0.845)
            ax2.grid()

            # Add legend for AUC
            fig.legend(loc='center', bbox_to_anchor=(0.77, 0.5), fontsize='large', frameon=True, shadow=True, borderpad=1)

            # Save the figure
            plt.tight_layout()
            plt.savefig(plot_file)
            plt.close()     

            # Save the data to an HDF5 file
            h5_file = os.path.join('/pscratch/sd/d/dimathan/LHCO/GAE', f'loss_and_auc_data_gae_e{len(self.auc_list)}_n{self.n_part}.h5')
            with h5py.File(h5_file, 'w') as h5f:
                h5f.create_dataset('train_loss', data=self.train_loss_list)
                h5f.create_dataset('val_loss', data=self.val_loss_list)
                h5f.create_dataset('auc', data=self.auc_list)

               
        return quantiles_loss
