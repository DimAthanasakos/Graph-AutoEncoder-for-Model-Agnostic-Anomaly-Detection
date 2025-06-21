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
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

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

from models.models import  EdgeNet, RelGAE, EdgeNet_edge_VGAE, AE, VAE
import ml_anomaly
from utils import load_jet_observables
import networkx
import energyflow as ef 
import random

import h5py 

def kl_anneal(epoch, total_epochs, max_kl_weight=1.0):
    return max_kl_weight * min(1, epoch / (total_epochs * 0.3))  # warm-up over 30% of total epochs

class gae(): 
    def __init__(self, model_info, plot_path='/global/homes/d/dimathan/gae_for_anomaly/plots_gae/plot_test'):
        self.model_info = model_info
        self.ddp = model_info['ddp']
        self.torch_device = model_info['torch_device']
        self.rank = 0
        self.plot_path = plot_path
        self.ext_plot = model_info['ext_plot']
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        if self.ddp: 
            self.rank = int(os.getenv("LOCAL_RANK"))
            if not dist.is_initialized():
                # initialize the process group only if it hasn't been initialized already
                print(f"Initializing DDP on rank {self.rank}...")
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
        if self.rank==0: print(f'gae, device: {self.torch_device}')
        self.output_dir = model_info['output_dir']
        

        
        self.n_total = model_info['n_total']
        self.n_train = model_info['n_train']

        self.n_val = model_info['n_val'] 
        self.n_part = model_info['n_part']
        
        self.unsupervised = model_info['model_settings']['unsupervised']
        self.path = ''
        if self.model_info['model'] not in  ['AE', 'VAE']:
            self.path = model_info['path_SB'] # path to the data (pyg dataset)
            if self.unsupervised:
                self.path = model_info['path_SR'] # For unsupervised we never use the sideband, so we use the SR path

        self.batch_size = self.model_info['model_settings']['batch_size']
        self.lossname = self.model_info['model_settings']['lossname']
        self.input_dim = self.model_info['model_settings']['input_dim']
        if 'pair_input_dim' in self.model_info['model_settings']:
            self.pair_input_dim = self.model_info['model_settings']['pair_input_dim']
        else: self.pair_input_dim = 0

        self.graph_structure = ''
        if 'graph_types' in self.model_info['model_settings']:
            self.graph_structure = self.model_info['model_settings']['graph_types'][0]
        
        if self.lossname == 'MSE': self.criterion = torch.nn.MSELoss()
        else: sys.exit(f'Error: loss {self.lossname} not recognized.')


        # Use custon training parameters
        self.epochs = self.model_info['model_settings']['epochs']
        self.learning_rate = self.model_info['model_settings']['learning_rate']
        
        if self.epochs > 20 and self.n_train > 10000: # otherwise its just testing, no need to store the plots on a permanent folder
            self.plot_path = f'/global/homes/d/dimathan/gae_for_anomaly/plots_gae/plot_n{self.n_part}_e{self.epochs}_lr{self.learning_rate}_N{self.n_train//1000}k'
            if not os.path.exists(self.plot_path):
                os.makedirs(self.plot_path)

        self.train_loader, self.val_loader = self.init_data()

        self.model = self.init_model().to(self.torch_device)


        if self.ddp:
            if self.rank ==0:    print("Converting BatchNorm layers to SyncBatchNorm...")
            # Convert all BatchNorm layers (like BatchNorm1d) in the model to SyncBatchNorm
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            if self.rank ==0:    print("SyncBatchNorm conversion complete.")
            self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank, find_unused_parameters=True)
            self.batch_size = self.batch_size // torch.cuda.device_count()
            if self.rank ==0:
                print(f'Using DDP with batch size: {self.batch_size} and learning rate: {self.learning_rate}')


        self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,  # Initial learning rate (use a smaller value, e.g., 1e-4 or 5e-5)
                weight_decay=0.01)  # Regularization to reduce overfitting
                
        # optimizer + Scheduler
        if self.model_info['model'] not in ['AE', 'VAE']:   

            if self.ddp:
                num_training_steps = len(self.train_loader) * self.epochs * dist.get_world_size()
            else:
                num_training_steps = len(self.train_loader) * self.epochs
            num_warmup_steps = int(0.02 * num_training_steps)  # Warmup for 20% of training steps
   
            start_lr_div = 5
            end_lr_div = 3

            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.learning_rate,  # Peak learning rate during warmup
                total_steps=num_training_steps,
                anneal_strategy='linear',  # Linearly decay the learning rate after warmup
                pct_start=num_warmup_steps / num_training_steps,  # Proportion of warmup steps
                div_factor=start_lr_div,  # Initial LR is 1/10th of max_lr
                final_div_factor= end_lr_div ) # Final LR is 1/5th of max_lr
        else:
            self.scheduler = None
            

    #---------------------------------------------------------------
    def init_data(self):
        if self.model_info['model'] in ['AE', 'VAE'] : 
            # Load observables from the sideband (or adjust use_SR as needed)
            obs, y = load_jet_observables(self.n_train + self.n_val, use_SR=False)  # shape: (n_train, 2, 5)     
            # shuffle with sk learn the obs and y
            obs, y = sklearn.utils.shuffle(obs, y, random_state=0)
            obs_train, y_train = obs[:self.n_train], y[:self.n_train]
            obs_val, y_val = obs[self.n_train:], y[self.n_train:]
            
            if self.model_info['model'] == 'VAE':
                s_over_b = self.model_info['model_settings']['s_over_b']
                print(f'==========='*5)
                print(f's/b: {s_over_b}')
                print(f'==========='*5)
                
                n_train_sig = int(s_over_b * self.n_train)
                n_val_sig = int(s_over_b * self.n_val) 

                ntrain_bkg_sr, nval_bkg_sr = n_train_sig, n_val_sig

                ntrain_bkg_sb =  self.n_train - n_train_sig -  ntrain_bkg_sr
                nval_bkg_sb =  self.n_val - n_val_sig - nval_bkg_sr

                print(f'ntrain_bkg_sb: {ntrain_bkg_sb}, nval_bkg_sb: {nval_bkg_sb}')
                print(f'ntrain_bkg_sr: {ntrain_bkg_sr}, nval_bkg_sr: {nval_bkg_sr}')
                print(f'n_train_sig: {n_train_sig}, n_val_sig: {n_val_sig}')

                obs_sb, y_sb = load_jet_observables(ntrain_bkg_sb  + nval_bkg_sb, use_SR=False)  # shape: (n_train, 2, 5)

                obs_sr, y_sr = load_jet_observables(20000, use_SR=True)  # shape: (n_train, 2, 5)


                obs_train_bkg_sr, y_train_bkg_sr = obs_sr[:ntrain_bkg_sr], y_sr[:ntrain_bkg_sr]
                obs_val_bkg_sr, y_val_bkg_sr = obs_sr[ntrain_bkg_sr: ntrain_bkg_sr + nval_bkg_sr], y_sr[ntrain_bkg_sr: ntrain_bkg_sr + nval_bkg_sr]

                obs_train_bkg, y_train_bkg = np.concatenate((obs_sb[:ntrain_bkg_sb], obs_train_bkg_sr), axis=0), np.concatenate((y_sb[:ntrain_bkg_sb], y_train_bkg_sr), axis=0)
                obs_val_bkg, y_val_bkg = np.concatenate((obs_sb[ntrain_bkg_sb:], obs_val_bkg_sr), axis=0), np.concatenate((y_sb[ntrain_bkg_sb:], y_val_bkg_sr), axis=0)


                obs_train_sig, y_train_sig = obs_sr[10000:10000+n_train_sig], y_sr[10000:10000+n_train_sig]
                obs_val_sig, y_val_sig = obs_sr[10000+n_train_sig:10000+n_train_sig+n_val_sig], y_sr[10000+n_train_sig:10000+n_train_sig+n_val_sig]



                print(f'obs_train_sig.shape: {obs_train_sig.shape}, y_train_sig.shape: {y_train_sig.shape}')
                print(f'obs_train_bkg.shape: {obs_train_bkg.shape}, y_train_bkg.shape: {y_train_bkg.shape}')

                obs_train, y_train = np.concatenate((obs_train_bkg, obs_train_sig), axis=0), np.concatenate((y_train_bkg, y_train_sig), axis=0)
                obs_val, y_val = np.concatenate((obs_val_bkg, obs_val_sig), axis=0), np.concatenate((y_val_bkg, y_val_sig), axis=0)



            print(f'obs_train.shape: {obs_train.shape}, y_train.shape: {y_train.shape}')
            print(f'obs_val.shape: {obs_val.shape}, y_val.shape: {y_val.shape}')

            # --- SORTING STEP ---
            print("Sorting jets within each event by mass (descending)...")
            # Sort obs_train
            masses_train = obs_train[:, :, 0]  # Get masses for sorting (n_events, 2)
            # Get indices that would sort masses in descending order for each event
            sort_indices_train = np.argsort(-masses_train, axis=1) # Argsort on negative mass for descending
            # Use take_along_axis to gather jets in the sorted order
            # Add new axis to sort_indices for broadcasting across features
            obs_train = np.take_along_axis(obs_train, sort_indices_train[:, :, np.newaxis], axis=1)

            # Sort obs_val
            masses_val = obs_val[:, :, 0]
            sort_indices_val = np.argsort(-masses_val, axis=1)
            obs_val = np.take_along_axis(obs_val, sort_indices_val[:, :, np.newaxis], axis=1)
            print("Sorting complete.")
            print(f'-------'*10)
            # --- END SORTING STEP ---

            obs_train = obs_train[:, :, :self.input_dim]
            obs_train = obs_train.reshape(obs_train.shape[0], -1)
            #print(f'obs_train.shape: {obs_train.shape}, y_train.shape: {y_train.shape}')
            obs_val = obs_val[:, :, :self.input_dim]
            obs_val = obs_val.reshape(obs_val.shape[0], -1)

            sc = 'std'
            if sc == 'std':
                scaler = StandardScaler()
                print('Using StandardScaler')
            elif sc == 'maxabs':
                scaler = MaxAbsScaler()
                print('Using MaxAbsScaler')

            self.scaler = scaler
            # Create torch datasets and loaders.
            train_data = torch.tensor(obs_train, dtype=torch.float)
            y_train = torch.tensor(y_train, dtype=torch.float)
            val_data = torch.tensor(obs_val, dtype=torch.float)
            y_val = torch.tensor(y_val, dtype=torch.float)

            #combine the observables and the labels
            train_data = torch.utils.data.TensorDataset(train_data, y_train)
            val_data = torch.utils.data.TensorDataset(val_data, y_val)

            train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)


        else: 
            # Load the dataset on all ranks
            t_st = time.time()
            if self.rank == 0: print(f'Loading data from {self.path}')
            dataset = torch.load(self.path, map_location='cpu', weights_only=False) # Load directly to the target device, allowing pickled objects
            if self.rank == 0:
                print(f'Loaded dataset with {len(dataset)} samples in {time.time()-t_st:.1f} seconds')
                if self.n_total > len(dataset):
                    print(f'\n==========================================')
                    print(f'Warning: n_total ({self.n_total}) is greater than the dataset length ({len(dataset)}).')
                    print(f'==========================================\n')
                print(f'Using {min(len(dataset), self.n_total)} samples for training, validation and testing')

            if not self.unsupervised: # just load everything since the same dataset is NOT used for testing on the anomaly
                # Slice the dataset on all ranks
                dataset = dataset[:self.n_total]

            elif self.unsupervised: # unsupervised training, be careful about the dataset 
                dataset_size = len(dataset)
                print(f'dataset_size: {dataset_size}')
                # find the index of the first signal event
                t_st = time.time()
                for i in range(len(dataset)):
                    if dataset[i][0].y == 1:
                        num_bkg = i
                        print(f'first signal index: {num_bkg}')
                        break
                else:  sys.exit(f'Error: no signal found in the dataset')

                dataset_bkg = dataset[:num_bkg]
                dataset_sig = dataset[num_bkg:]

                if self.n_train + self.n_val > 0.9 * num_bkg: #  recommending an 80% split for training, to keep at least 10k + 10k for testing on anomalies 
                    print(f'Warning: n_train + n_val is too large for unsupervised training. Setting n_train to 0.85 * {num_bkg} and n_val to 0.05 * {num_bkg}')
                    self.n_train = int(0.85 * num_bkg)
                    self.n_val = int(0.05 * num_bkg)
                print(f'n_train: {self.n_train}, n_val: {self.n_val}')
                s_over_b = self.model_info['model_settings']['s_over_b'] 
                print(f's/b: {s_over_b}')
                print(f'------------')

                n_train_sig = int(s_over_b * self.n_train) 
                n_val_sig = int(s_over_b * self.n_val)
                n_train_bkg = self.n_train - n_train_sig
                n_val_bkg = self.n_val - n_val_sig

                print(f'n_train_bkg: {n_train_bkg}, n_val_bkg: {n_val_bkg}')
                print(f'n_train_sig: {n_train_sig}, n_val_sig: {n_val_sig}')

                bkg = dataset_bkg[:n_train_bkg + n_val_bkg]
                sig = dataset_sig[:n_train_sig + n_val_sig] 
                print(f'in the training + validation dataset we have: {len(bkg)} bkg and {len(sig)} sig events')
                dataset = torch.utils.data.ConcatDataset([bkg, sig])

            # shuffle the dataset
            dataset = sklearn.utils.shuffle(dataset, )
            train_dataset = dataset[:self.n_train]
            val_dataset = dataset[self.n_train:self.n_train + self.n_val] # we are not careful about maintaining the exact s/b in both train and val, only at their sum.    
                                                                          # but for typical run of ~100k events, and s/b = 1%, the statistical variation of this random sampling
                                                                          # without replacement is ~O(10) signal events for the training and the val split. Since s/b does not change 
                                                                          # the performance of the model, we can safely ignore this correction.
            if self.ddp:
                # Use DistributedSampler to handle data sharding per GPU
                self.train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True) # drop_last can sometimes help with uneven batches

                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=self.train_sampler, num_workers=0, pin_memory=False)
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=False)

            else: # Single GPU case
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
                val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=2, pin_memory=True)

        
        return train_loader, val_loader, 


    def _distribute_dataset(self, dataset):
        """Distributes the dataset to all ranks."""
        if self.ddp:
            world_size = dist.get_world_size()
            if self.rank == 0:
                dataset_list = [dataset] * world_size  # Create a list to broadcast
            else:
                dataset_list = [None] * world_size

            dist.broadcast_object_list(dataset_list, src=0)  # Broadcast the list from rank 0
            dataset = dataset_list[self.rank]  # Assign the dataset to each rank
            return dataset
        else:
            return dataset

    def _scatter_object(self, obj):
        """Scatters an object to all ranks."""
        obj = dist.scatter_object_list([None for _ in range(dist.get_world_size())], obj)
        return obj


    #---------------------------------------------------------------
    def init_model(self):
        '''
        :return: pytorch architecture
        '''
        self.model_to_choose = self.model_info['model']

        if self.model_to_choose == 'EdgeNet':
            model = EdgeNet(input_dim=self.input_dim, big_dim=32, latent_dim=2, dropout_rate=0.0)

        elif self.model_to_choose == 'RelGAE':
            model = RelGAE(input_dim=self.input_dim, edge_dim=self.pair_input_dim, big_dim=64, latent_dim=2, output_dim = self.input_dim, dropout_rate=0.0)

        elif self.model_to_choose == 'EdgeNet_edge_VGAE':
            model = EdgeNet_edge_VGAE(input_dim=self.input_dim, edge_dim=self.pair_input_dim, big_dim=32, latent_dim=2, output_dim = self.input_dim, dropout_rate=0.0)
            
        elif self.model_to_choose == 'AE':
            model = AE(input_dim=self.input_dim, hidden_dim=64, latent_dim=2) 

        elif self.model_to_choose == 'VAE':
            model = VAE(input_dim=2*self.input_dim, hidden_dim=62, latent_dim=2, activation='selu')

        else: sys.exit(f'Error: model {self.model_to_choose} not recognized.') 
        
        #########################################
        model = torch.compile(model)
        #########################################
        
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
        best_kl_loss = -math.inf
        pth = 'model_best_' + f'n{self.n_part}_' + f'{self.graph_structure}' + '.pt'
        best_model_path = os.path.join(self.output_dir, pth)  # Path for saving the best model

        self.auc_list = []
        self.max_sic_list = []
        self.train_loss_list = []
        self.val_loss_list = []
        if self.model_info['model'] in ['VAE', 'EdgeNet_edge_VGAE']: 
            self.kl_weight = self.model_info['model_settings']['kl_weight']
            print(f'-------'*10 )
            print(f'kl_weight: {self.kl_weight}')
            print(f'-------'*10 )
            print()
        else: self.kl_weight = 0

        anomaly_model = ml_anomaly.anomaly(self.model_info, plot=False, scaler = self.scaler if self.model_info['model'] in ['AE', 'VAE',] else None)
        for epoch in range(1, self.epochs+1):
            if self.ddp: self.train_sampler.set_epoch(epoch)  

            t_start = time.time()
            loss_train, _, _ = self._train_loop(ep=epoch)
            #loss_train = 0 
            loss_val, loss_kl_val = self._test_loop(self.val_loader)
            #loss_val, loss_kl_val = 0, 0 
            if self.rank==0:
                self.val_loss_list.append(loss_val)
                if epoch%10==0 or self.ext_plot:
                    auc, max_sic = anomaly_model.run(model=self.model)
                    self.auc_list.append(auc)
                    self.max_sic_list.append(max_sic)
                if self.model_info['model'] in ['VAE', 'EdgeNet_edge_VGAE']:
                    print(f'Epoch: {epoch:02d}, loss_train: {loss_train:.5f}, loss_val: {loss_val:.5f}, lr: {self.optimizer.param_groups[0]["lr"]:.5f}, Time: {time.time() - t_start:.1f} sec, loss_kl_val: {loss_kl_val:.7f}, ')
                    
                    if self.model_info['model'] in ['VAE'] and loss_kl_val > best_kl_loss:
                        best_kl_loss = loss_kl_val
                        torch.save(self.model.state_dict(), best_model_path)
                    elif self.model_info['model'] in ['EdgeNet_edge_VGAE'] and loss_val < best_val_loss:
                        best_val_loss = loss_val
                        torch.save(self.model.state_dict(), best_model_path)
                else: 
                    if loss_val < best_val_loss:
                        best_val_loss = loss_val
                        torch.save(self.model.state_dict(), best_model_path)
                    print(f'Epoch: {epoch:02d}, loss_train: {loss_train:.5f}, loss_val: {loss_val:.5f}, lr: {self.optimizer.param_groups[0]["lr"]:.5f}, Time: {time.time() - t_start:.1f} sec')

                print("--------------------------------")

        if self.rank==0:
            # lets plot the loss distribution of the test set
            print(f'--------------------------------')
            print(f'Finished training\n')
            
            # Load the best model before evaluating final metrics.
            self.model.load_state_dict(torch.load(best_model_path))
            print("Loaded the best model based on validation loss.")
            if self.ddp and isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.model = self.model.module
            if self.model_info['model'] in ['VAE', 'EdgeNet_edge_VGAE']:
                best_train_loss, best_kl_loss = self._test_loop(self.train_loader)
                best_val_loss, best_kl_loss = self._test_loop(self.val_loader)
            else: 
                best_train_loss, _ = self._test_loop(self.train_loader)
                best_val_loss, _  = self._test_loop(self.val_loader)
            print('initializing the anomaly model')
            anomaly_model = ml_anomaly.anomaly(self.model_info, plot=True, scaler = self.scaler if self.model_info['model'] in ['AE', 'VAE',] else None)
            print('running the anomaly model')
            auc, max_sic = anomaly_model.run(model=self.model)

            self._plot_loss()

            print(f'Best model train loss: {best_train_loss:.5f}')
            print(f'Best model validation loss: {best_val_loss:.5f}')
            print(f'AUC: {auc:.4f}')
            print(f'Max SIC: {max_sic:.4f}')
            if self.auc_list:
                auc_max = max(self.auc_list)
                print(f'max aux: {auc_max:.4f}')
            else: auc_max = 0 
            print(f'--------------------------------\n')
            # Return the best model and the metrics for this run.
            return self.model, best_train_loss, best_val_loss, auc, auc_max, max_sic
        else: 
            return None, None, None, None, None, None 

    #---------------------------------------------------------------
    def _train_loop(self, ep=-1):
        self.model.train()
        loss_cum = 0
        count = 0
        loss_cum_nodes, loss_cum_edges, loss_cum_kl = 0, 0, 0

        if self.model_info['model'] == 'AE':
            # Training loop for AE: each batch is a tensor of shape (batch_size, 10).
            for index, batch in enumerate(self.train_loader):
                inputs = batch[0].to(self.torch_device)
                self.optimizer.zero_grad()
                input1 = inputs[:, :inputs.shape[1] // 2]
                input2 = inputs[:, inputs.shape[1] // 2:]
                recon1, _ = self.model(input1)
                recon2, _ = self.model(input2)
                loss = self.criterion(recon1, input1) + self.criterion(recon2, input2)
                if index == -1: 
                    print(f'inputs[0]: {inputs[0]}')
                    print(f'recon1[0]: {recon1[0]}')
                    print(f'recon2[0]: {recon2[0]}')
                    time.sleep(0.5)
                loss.backward()
                self.optimizer.step()
                #self.scheduler.step()
                batch_size = inputs.shape[0]
                loss_cum += loss.item() * batch_size
                count += batch_size
                return loss_cum / count, None, None 
        elif self.model_info['model'] == 'VAE':
            # Training loop for VAE: each batch is a tensor of shape (batch_size, 10).
            for index, batch in enumerate(self.train_loader):
                inputs = batch[0].to(self.torch_device)
                self.optimizer.zero_grad()
                reconstruction, mu, logvar = self.model(inputs)
                reconstruction_loss = self.criterion(reconstruction, inputs)
                kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                loss =  reconstruction_loss +  self.kl_weight * kl_loss
                if index == -1: 
                    print(f'inputs[0]: {inputs[0]}')
                    print(f'reconstruction[0]: {reconstruction[0]}')
                    time.sleep(0.5)
                loss.backward()
                self.optimizer.step()

                batch_size = inputs.shape[0]
                loss_cum += loss.item() * batch_size
                loss_cum_kl += self.kl_weight*kl_loss.item() * batch_size
                count += batch_size
            #print(f'train: loss_cum: {loss_cum/count :.3f}, loss_kl_cum: {loss_cum_kl/count:.4f}')
            return loss_cum / count, None, None
        else:

            for batch_idx, (batch_jets0, batch_jets1) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                length = len(batch_jets0)
                batch_jets0 = batch_jets0.to(self.torch_device)
                batch_jets1 = batch_jets1.to(self.torch_device)

                if self.input_dim==1:
                    batch_jets0.x = batch_jets0.x[:, 0].unsqueeze(1)
                    batch_jets1.x = batch_jets1.x[:, 0].unsqueeze(1)
                elif self.input_dim==3: # redundant right now, since this is the info of batch_jets.x 
                    batch_jets0.x = batch_jets0.x[:, :3]
                    batch_jets1.x = batch_jets1.x[:, :3]

                target0 = batch_jets0.x
                target1 = batch_jets1.x

                # If using a variational model, it returns four outputs.
                if self.model_to_choose.endswith('VGAE'):
                    out0, edge_out0, mu0, logvar0 = self.model(batch_jets0)
                    out1, edge_out1, mu1, logvar1 = self.model(batch_jets1)
                    loss_node0 = self.criterion(out0, target0)
                    loss_node1 = self.criterion(out1, target1)
                    
                    target_edge0 = batch_jets0.edge_attr 
                    target_edge1 = batch_jets1.edge_attr
                    # (Handle special cases such as 'EdgeNet_laman' if needed.)
                    loss_edge0 = self.criterion(edge_out0, target_edge0)
                    loss_edge1 = self.criterion(edge_out1, target_edge1)
                    
                    # Compute KL divergence per batch (averaged over nodes)
                    kl0 = -0.5 * torch.mean(torch.sum(1 + logvar0 - mu0.pow(2) - logvar0.exp(), dim=1))
                    kl1 = -0.5 * torch.mean(torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp(), dim=1))
                    
                    reconstruction_loss = loss_node0 + loss_node1 + loss_edge0 + loss_edge1
                    kl_loss = kl0 + kl1
                    # KL annealing (optional)
                    #kl_weight = kl_anneal(ep, self.epochs, max_kl_weight=1e-3)  # tweak max_kl_weight as needed
                    #if batch_idx == 0:
                    #    print(f'recon loss: {reconstruction_loss.item():.3f}, kl loss: {kl_weight*kl_loss.item():.3f}')

                    loss = reconstruction_loss + self.kl_weight * kl_loss

                    # For logging purposes:
                    loss_nodes = (loss_node0.item() + loss_node1.item())
                    loss_edges = (loss_edge0.item() + loss_edge1.item())
                    loss_kl = self.kl_weight * kl_loss.item()
                    
                elif self.model_to_choose in ['RelGAE']:
                    # Original deterministic model code path
                    out0, edge_out0 = self.model(batch_jets0)
                    out1, edge_out1 = self.model(batch_jets1)
                    loss_node0 = self.criterion(out0, target0)
                    loss_node1 = self.criterion(out1, target1)
                    target_edge0 = batch_jets0.edge_attr 
                    target_edge1 = batch_jets1.edge_attr 
                    loss_edge0 = self.criterion(edge_out0, target_edge0)
                    loss_edge1 = self.criterion(edge_out1, target_edge1)
                    loss = loss_node0 + loss_node1 + loss_edge0 + loss_edge1
                    loss_nodes = loss_node0.item() + loss_node1.item()
                    loss_edges = loss_edge0.item() + loss_edge1.item()
                    loss_kl = 0

                    target_edge0 = batch_jets0.edge_attr 
                    target_edge1 = batch_jets1.edge_attr

                    loss_node0 = self.criterion(out0, target0)
                    loss_node1 = self.criterion(out1, target1)
                    loss_edge0 = self.criterion(edge_out0, target_edge0)
                    loss_edge1 =  self.criterion(edge_out1, target_edge1)

                    
                    loss = loss_node0 + loss_node1 + loss_edge0 + loss_edge1  
                    loss_nodes = loss_node0.item() + loss_node1.item()
                    loss_edges = loss_edge0.item() + loss_edge1.item()
                    loss_kl = 0 

                elif self.model_to_choose == 'EdgeNet': 
                    out0, out1 = self.model(batch_jets0), self.model(batch_jets1)
                    loss_node0 = self.criterion(out0, target0)
                    loss_node1 = self.criterion(out1, target1)

                    loss = loss_node0 + loss_node1
                    loss_nodes = loss_node0.item() + loss_node1.item()
                    loss_edges = 0
                    loss_kl = 0
                    
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                loss_cum += loss.item() * length
                loss_cum_nodes += loss_nodes * length
                loss_cum_edges += loss_edges * length
                loss_cum_kl += loss_kl * length
                count += length

            return loss_cum/count, loss_cum_nodes/count, loss_cum_edges/count  # (and log kl if needed)
        
    #---------------------------------------------------------------
    @torch.no_grad()
    def _test_loop(self, test_loader):
        self.model.eval()
        loss_cum, loss_kl_cum = 0, 0
        count = 0
        if self.model_info['model'] == 'AE':
            for batch in test_loader:
                inputs = batch[0].to(self.torch_device)
                input1 = inputs[:, :inputs.shape[1] // 2]
                input2 = inputs[:, inputs.shape[1] // 2:]
                recon1, _ = self.model(input1)
                recon2, _ = self.model(input2)
                loss = self.criterion(recon1, input1) + self.criterion(recon2, input2)
                batch_size = inputs.shape[0]
                loss_cum += loss.item() * batch_size
                count += batch_size
        elif self.model_info['model'] == 'VAE': 
            for batch in test_loader:
                inputs = batch[0].to(self.torch_device)
                reconstruction, mu, logvar = self.model(inputs)
                reconstruction_loss = self.criterion(reconstruction, inputs)
                kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                loss = reconstruction_loss +  self.kl_weight * kl_loss
                batch_size = inputs.shape[0]
                loss_cum += loss.item() * batch_size
                loss_kl_cum += self.kl_weight * kl_loss.item() * batch_size
                count += batch_size
            return loss_cum / count, loss_kl_cum / count
            #print(f'val:   loss_cum: {loss_cum/count:.3f}, loss_kl_cum: {loss_kl_cum/count:.4f}')
        else: 
            for batch_jets0, batch_jets1 in test_loader:
                length = len(batch_jets0)
                batch_jets0 = batch_jets0.to(self.torch_device)
                batch_jets1 = batch_jets1.to(self.torch_device)
                
                if self.input_dim in [1,3]:
                    if self.input_dim==1:
                        batch_jets0.x = batch_jets0.x[:, 0].unsqueeze(1)#.clone()
                        batch_jets1.x = batch_jets1.x[:, 0].unsqueeze(1)#.clone()
                    
                    target0 = batch_jets0.x
                    target1 = batch_jets1.x

                    if self.model_to_choose.endswith('VGAE'):
                        out0, edge_out0, mu0, logvar0 = self.model(batch_jets0)
                        out1, edge_out1, mu1, logvar1 = self.model(batch_jets1)
                        loss_node0 = self.criterion(out0, target0)
                        loss_node1 = self.criterion(out1, target1)
                        target_edge0 = batch_jets0.edge_attr 
                        target_edge1 = batch_jets1.edge_attr 

                        loss_edge0 = self.criterion(edge_out0, target_edge0)
                        loss_edge1 = self.criterion(edge_out1, target_edge1)
                        reconstruction_loss = loss_node0 + loss_node1 + loss_edge0 + loss_edge1
                        kl0 = -0.5 * torch.mean(torch.sum(1 + logvar0 - mu0.pow(2) - logvar0.exp(), dim=1))
                        kl1 = -0.5 * torch.mean(torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp(), dim=1))
                        loss = reconstruction_loss + self.kl_weight * (kl0 + kl1)
                        kl_loss = self.kl_weight * (kl0 + kl1)

                    elif self.model_to_choose in ['RelGAE']:
                        if self.ddp and isinstance(self.model, DDP):
                            out0, edge_out0 = self.model.module(batch_jets0)
                            out1, edge_out1 = self.model.module(batch_jets1)
                        else:
                            out0, edge_out0 = self.model(batch_jets0)
                            out1, edge_out1 = self.model(batch_jets1)
                        loss_node0 = self.criterion(out0, target0)
                        loss_node1 = self.criterion(out1, target1)
                        target_edge0 = batch_jets0.edge_attr 
                        target_edge1 = batch_jets1.edge_attr 
                        loss_edge0 = self.criterion(edge_out0, target_edge0)
                        loss_edge1 = self.criterion(edge_out1, target_edge1)
                        loss = loss_node0 + loss_node1 + loss_edge0 + loss_edge1

                    elif self.model_to_choose == 'EdgeNet': 
                        out0, out1 = self.model(batch_jets0), self.model(batch_jets1)
                        loss_node0 = self.criterion(out0, target0)
                        loss_node1 = self.criterion(out1, target1)
                        loss = loss_node0 + loss_node1

                loss_cum += loss.item() * length
                loss_kl_cum += kl_loss.item() * length if 'kl_loss' in locals() else 0
                count += length
        return loss_cum / count, loss_kl_cum / count if 'kl_loss' in locals() else 0
        
        
    #---------------------------------------------------------------
    @torch.no_grad()
    def _plot_loss(self, extended_plots=False):
        self.model.eval()
        event_losses = []
        criterion_node = torch.nn.MSELoss(reduction='none')  
        criterion_edge = torch.nn.MSELoss(reduction='none')

        with torch.no_grad():
            for batch in self.val_loader:
                if self.model_info['model'] == 'AE': 
                    inputs = batch[0].to(self.torch_device)
                    reconstruction, _ = self.model(inputs)
                    # Compute per-event MSE
                    loss_vals = torch.mean((reconstruction - inputs) ** 2, dim=1)
                    scores = loss_vals
                elif self.model_info['model'] == 'VAE':
                    inputs = batch[0].to(self.torch_device)
                    reconstruction, mu, logvar = self.model(inputs)
                    # Compute per-event MSE
                    loss_vals = torch.mean((reconstruction - inputs) ** 2, dim=1)
                    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                    scores = loss_vals + self.kl_weight * kl_loss
                else: 
                    batch_jets0 = batch[0]
                    batch_jets1 = batch[1]

                    batch_jets0 = batch_jets0.to(self.torch_device)
                    batch_jets1 = batch_jets1.to(self.torch_device)

                    if self.input_dim == 4:
                        out1 = self.model(batch_jets0)
                        out2 = self.model(batch_jets1)
                        loss1_nodewise = criterion_node(out1, batch_jets0.match)
                        loss2_nodewise = criterion_node(out2, batch_jets1.match)
                        loss1_per_graph = scatter_mean(loss1_nodewise.mean(dim=-1), batch_jets0.batch, dim=0)
                        loss2_per_graph = scatter_mean(loss2_nodewise.mean(dim=-1), batch_jets1.batch, dim=0)
                        scores = loss1_per_graph + loss2_per_graph
                    elif self.input_dim in [1, 3]:
                        if self.input_dim==1:
                            batch_jets0.x = batch_jets0.x[:, 0].unsqueeze(1).clone()
                            batch_jets1.x = batch_jets1.x[:, 0].unsqueeze(1).clone()
                            
                        target0 = batch_jets0.x
                        target1 = batch_jets1.x

                        if self.model_info['model'] == 'EdgeNet':
                            out1 = self.model(batch_jets0)
                            out2 = self.model(batch_jets1)
                            loss1_nodewise = criterion_node(out1, target0)
                            loss2_nodewise = criterion_node(out2, target1)
                            scores = scatter_mean(loss1_nodewise.mean(dim=-1), batch_jets0.batch, dim=0) + scatter_mean(loss2_nodewise.mean(dim=-1), batch_jets1.batch, dim=0)
                        
                        else:    
                            if self.model_info['model'] == 'EdgeNet_edge_VGAE':
                                out1, edge_out1, mu0, logvar0 = self.model(batch_jets0)
                                out2, edge_out2, mu1, logvar1 = self.model(batch_jets1)
                                kl0 = -0.5 * torch.mean(torch.sum(1 + logvar0 - mu0.pow(2) - logvar0.exp(), dim=1))
                                kl1 = -0.5 * torch.mean(torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp(), dim=1))

                            elif self.model_to_choose in ['RelGAE']:
                                out1, edge_out1 = self.model(batch_jets0)
                                out2, edge_out2 = self.model(batch_jets1)

                            elif self.model_to_choose == 'loss_global':
                                out1, edge_out1, pred_global0 = self.model(batch_jets0) 
                                out2, edge_out2, pred_global1 = self.model(batch_jets1)
                                loss_global0 = self.criterion(pred_global0, batch_jets0.global_features).mean(dim=-1)
                                loss_global1 = self.criterion(pred_global1, batch_jets1.global_features).mean(dim=-1)
                                #loss_global0 = scatter_mean(loss_global0, batch_jets0.batch, dim=0)
                                #loss_global1 = scatter_mean(loss_global1, batch_jets1.batch, dim=0)
                                loss_global = loss_global0 + loss_global1


                            loss1_nodewise = criterion_node(out1, target0)
                            loss2_nodewise = criterion_node(out2, target1) 

                            loss1_per_node = loss1_nodewise.mean(dim=-1)
                            loss2_per_node = loss2_nodewise.mean(dim=-1)
                            loss1_per_graph = scatter_mean(loss1_per_node, batch_jets0.batch, dim=0)
                            loss2_per_graph = scatter_mean(loss2_per_node, batch_jets1.batch, dim=0)
                            
                            target_edge1 = batch_jets0.edge_attr
                            target_edge2 = batch_jets1.edge_attr

                            loss1_edgewise = criterion_edge(edge_out1, target_edge1)
                            loss2_edgewise = criterion_edge(edge_out2, target_edge2)
                            loss1_per_edge = loss1_edgewise.mean(dim=-1)
                            loss2_per_edge = loss2_edgewise.mean(dim=-1)


                            loss1_edge_per_graph = scatter_mean(loss1_per_edge, batch_jets0.batch[batch_jets0.edge_index[0]], dim=0)
                            loss2_edge_per_graph = scatter_mean(loss2_per_edge, batch_jets1.batch[batch_jets1.edge_index[0]], dim=0)
                           
                            reconstruction_scores = loss1_per_graph + loss2_per_graph + loss1_edge_per_graph + loss2_edge_per_graph + (loss_global if 'loss_global' in locals() else 0)
                            
                            if self.model_info['model'] == 'EdgeNet_edge_VGAE':
                                kl_loss = kl0 + kl1
                                scores = reconstruction_scores + self.kl_weight * kl_loss
                            else:
                                scores = reconstruction_scores
                event_losses.extend(scores.cpu().tolist())

        plot_file = os.path.join(self.plot_path, 'val_loss_distribution.pdf')
        loss_tot = event_losses
        quantiles_loss = {
            "50%": np.quantile(loss_tot, 0.5),
            "70%": np.quantile(loss_tot, 0.7),
            "80%": np.quantile(loss_tot, 0.8),
            "90%": np.quantile(loss_tot, 0.9),
        }
        p99 = np.quantile(loss_tot, 0.99)
        num_bins = 75
        bins = np.linspace(0, p99, num_bins)
        
        plt.figure(figsize=(8, 6))
        plt.hist(loss_tot, bins=bins, color='blue', histtype='step',
                label='Loss of test set (bkg)', density=True)
        plt.xlabel('Loss')
        plt.ylabel('Counts')
        plt.title('Comparison of Two Loss Distributions')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300)
        plt.close()
        print(f"Saved loss distribution plot to: {plot_file}")

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
