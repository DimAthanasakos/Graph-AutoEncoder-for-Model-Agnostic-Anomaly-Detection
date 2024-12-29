import os
import sys
import yaml
import pickle
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, DataListLoader
from torch_geometric.loader import DataLoader #.data.DataLoader has been deprecated

from torch_geometric.nn import EdgeConv, global_mean_pool, DataParallel
import random
sys.path.append('.')
from base import common_base
from models.models import EdgeNet
import gae_train, ml_anomaly, trans_train

torch.manual_seed(0)

################################################################
class MLAnalysis(common_base.CommonBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, config_file='', output_dir='', ddp=False, **kwargs):
        super(common_base.CommonBase, self).__init__(**kwargs)
        
        self.config_file = config_file
        self.output_dir = output_dir
        self.ddp = ddp
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
        
        self.n_part = config['n_part']
        self.n_train = config['n_train']
        self.n_val = config['n_val']
        self.n_test = config['n_test']
        self.n_total = self.n_train + self.n_val + self.n_test
        self.test_frac = 1. * self.n_test / self.n_total
        self.val_frac  = 1. * self.n_val /  self.n_total

        # Initialize model-specific settings
        self.models = config['models']
        self.model_settings = {}
        for model in self.models:
            self.model_settings[model] = config[model]

    
    #---------------------------------------------------------------
    # Train models
    #---------------------------------------------------------------
    def train_models(self):

        self.AUC = defaultdict(list)
        self.roc_curve_dict = self.recursive_defaultdict()
        for model in self.models:
            if self.rank == 0:
                print()
                print(f'------------- Training model: {model} -------------')
            model_settings = self.model_settings[model]
            model_info = {'model': model,
                        'model_settings': model_settings,
                        'n_part': self.n_part,
                        'n_total': self.n_total,
                        'n_train': self.n_train,
                        'n_val': self.n_val,
                        'n_test': self.n_test,
                        'torch_device': self.torch_device,
                        'output_dir': self.output_dir,
                        'ddp': self.ddp}                

            if model in ['transformer', 'transformer_graph']:
                model_key = f'{model}'
                model_info_temp = model_info.copy()
                model_info_temp['model_key'] = model_key
                model = trans_train.ParT(model_info_temp).train()
                #self.AUC = trans_anomaly.anomaly(model, model_info).run()

            else:
                model_key = f'{model}'


                batch_size = model_info['model_settings']['batch_size']
                n_total = model_info['n_total']

                for graph_structure in model_info['model_settings']['graph_types']: 
                    regions = ['SB', 'SR']
                    for region in regions:
                        graph_key = f'graphs_pyg_{region}__{graph_structure}'
                        path = os.path.join(self.output_dir, f'{graph_key}.pt')
                        model_info[f'graph_key_{region}'] = graph_key
                        model_info[f'path_{region}'] = path
                        print(f'graph_key_{region}: {graph_key}')
                        print(f'path_{region}: {path}')

                    model = gae_train.gae(model_info).train()
                    
                    self.AUC = ml_anomaly.anomaly(model, model_info).run()



