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

from models.models import EdgeNet, EdgeNetVAE, EdgeNet2, EdgeNetDeeper

import networkx
import energyflow as ef 
import random


class gae(): 
    def __init__(self, model_info, plot_path='/global/homes/d/dimathan/gae_for_anomaly/plots4'):
        self.model_info = model_info
        self.path = model_info['path_SB'] # path to the data (pyg dataset)
        self.ddp = model_info['ddp']
        self.torch_device = model_info['torch_device']
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
        self.lossname = self.model_info['model_settings']['lossname']
        
        if self.lossname == 'MSE': self.criterion = torch.nn.MSELoss()
        else: sys.exit(f'Error: loss {self.lossname} not recognized.')


        # Use custon training parameters
        self.epochs = self.model_info['model_settings']['epochs']
        self.learning_rate = self.model_info['model_settings']['learning_rate']

        self.train_loader, self.val_loader, self.test_loader = self.init_data()

        self.model = self.init_model().to(self.torch_device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)

        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank,)
            #self.batch_size = self.batch_size // torch.cuda.device_count()
            self.learning_rate = self.learning_rate * torch.cuda.device_count()

    #---------------------------------------------------------------
    def init_data(self):

        dataset = torch.load(self.path)

        random.Random(0).shuffle(dataset)
        if self.rank==0 and self.n_total > len(dataset):
            print()
            print(f'==========================================')
            print(f'WARNING')
            print(f'Warning: n_total ({self.n_total}) is greater than the dataset length ({len(dataset)}).')
            print(f'==========================================')
            print()


        dataset = dataset[:self.n_total]

        train_dataset  = dataset[:self.n_train]
        test_dataset   = dataset[self.n_train:self.n_train + self.n_test]
        valid_dataset  = dataset[self.n_train + self.n_test:]


        if self.ddp:
            self.train_sampler = DistributedSampler(train_dataset, shuffle=True)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler = self.train_sampler)
            valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
            test_loader  = DataLoader(test_dataset,  batch_size=self.batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
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
            model = EdgeNet(input_dim=3, big_dim=32, hidden_dim=2, aggr='mean')

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

        for epoch in range(1, self.epochs+1):
            if self.ddp: self.train_sampler.set_epoch(epoch)  

            t_start = time.time()
            loss_train = self._train_loop()
            loss_val = self._test_loop(self.val_loader,)
            loss_test = self._test_loop(self.test_loader,)

            # only print if its the main process
            if self.rank == 0:
                print("--------------------------------")
                print(f'Epoch: {epoch:02d}, loss_train: {loss_train:.4f}, loss_val: {loss_val:.4f}, loss_test: {loss_test:.4f}, Time: {time.time() - t_start:.1f} sec')

        # lets plot the loss distribution of the test set
        if self.rank == 0:            
            print(f'--------------------------------')
            print(f'Finished training')
            print()
            self._plot_loss()

        return self.model


    #---------------------------------------------------------------
    def _train_loop(self,):
        self.model.train()
        loss_cum = 0
        count = 0
        for batch_jets0, batch_jets1 in self.train_loader:
            self.optimizer.zero_grad()
            length = len(batch_jets0)
            batch_jets0 = batch_jets0.to(self.torch_device)
            batch_jets1 = batch_jets1.to(self.torch_device)             

            out0 = self.model(batch_jets0)
            out1 = self.model(batch_jets1)

            loss0 = self.criterion(out0, batch_jets0.x)
            loss1 = self.criterion(out1, batch_jets1.x)
            loss = loss0 + loss1

            loss.backward()
            self.optimizer.step()
            loss_cum += loss.item() 
            count += length
            # Cache management
            #torch.cuda.empty_cache()

        return loss_cum/count * 1000

    
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

            loss0 = self.criterion(out0, batch_jets0.x)
            loss1 = self.criterion(out1, batch_jets1.x)

            loss = loss0 + loss1

            loss_cum += loss.item() 
            count += length
            # Cache management
            #torch.cuda.empty_cache()
        
        return loss_cum / count * 1000

    #---------------------------------------------------------------
    @torch.no_grad()
    def _plot_loss(self):
        self.model.eval()
        event_losses = []

        for batch_jets0, batch_jets1 in self.test_loader:
            batch_jets0 = batch_jets0.to(self.torch_device)
            batch_jets1 = batch_jets1.to(self.torch_device)

            list_jet0 = batch_jets0.to_data_list()
            list_jet1 = batch_jets1.to_data_list()

            for jet0, jet1 in zip(list_jet0, list_jet1):
                out0 = self.model(jet0)
                out1 = self.model(jet1)

                loss0 = self.criterion(out0, jet0.x)
                loss1 = self.criterion(out1, jet1.x)
                event_losses.append([loss0.item()*1000, loss1.item()*1000])

        plot_file = os.path.join(self.plot_path, 'val_loss_distribution.pdf')


        # Separate the two losses
        loss0_array = [vals[0] for vals in event_losses]
        loss1_array = [vals[1] for vals in event_losses]
        loss_tot  = [(vals[0] + vals[1])/2 for vals in event_losses]

        # Example: Compute quantiles for loss0_array and loss1_array
        quantiles_loss = {
            "50%": np.quantile(loss_tot, 0.5),  # 50% quantile (median)
            "70%": np.quantile(loss_tot, 0.7),  # 70% quantile
            "80%": np.quantile(loss_tot, 0.8),  # 80% quantile
            "90%": np.quantile(loss_tot, 0.9),  # 90% quantile
        }
        
        # Check how many events have both graphs' losses > 50% quantile
        over_50_count = np.sum(
            (loss0_array > quantiles_loss['50%']) &
            (loss1_array > quantiles_loss['50%']) )
        over_50_count = over_50_count/len(loss0_array)

        # Check how many events have both graphs' losses > 70% quantile
        over_70_count = np.sum(
            (loss0_array > quantiles_loss['70%']) &
            (loss1_array > quantiles_loss['70%']) )
        over_70_count = over_70_count/len(loss0_array)

        # Check how many events have both graphs' losses > 80% quantile
        over_80_count = np.sum(
            (loss0_array > quantiles_loss['80%']) &
            (loss1_array > quantiles_loss['80%']) )
        over_80_count = over_80_count/len(loss0_array)
        
        print(f'--------------------------------')
        #print the average loss
        print(f"Average loss per jet: {np.mean(loss_tot)*2:.2f} ")
        print(f"Quantiles for the loss per jet: {quantiles_loss}")
        print()
        print(f"Percentage of events with both graphs' losses > 50% quantile: {over_50_count*100:.1f}%")
        print(f"Percentage of events with both graphs' losses > 70% quantile: {over_70_count*100:.1f}%")
        print(f"Percentage of events with both graphs' losses > 80% quantile: {over_80_count*100:.1f}%")
        print(f'--------------------------------')

        # Plot each distribution with a different color
        plt.figure(figsize=(8, 6))
        plt.hist(loss0_array, bins=100, color='blue', histtype='step', label='Loss 0')
        plt.hist(loss1_array, bins=100, color='red',  histtype='step', label='Loss 1')
        plt.xlabel('Loss')
        plt.ylabel('Counts')
        plt.title('Comparison of Two Loss Distributions')
        plt.legend()
        plt.tight_layout()

        plt.savefig(plot_file, dpi=300)
        plt.close()
        print(f"Saved loss distribution plot to: {plot_file}")
        print()

        return quantiles_loss
