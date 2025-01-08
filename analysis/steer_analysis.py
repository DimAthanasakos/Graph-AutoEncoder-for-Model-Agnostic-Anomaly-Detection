import glob
import torch
import tables
import itertools
import numpy as np
import pandas as pd
import os.path as osp
import multiprocessing
from tqdm import tqdm
from pathlib import Path
#from torch_geometric.data import Dataset, Data, Batch
import argparse
import os, sys, yaml
import utils, ml_analysis
import time 

# Base class
sys.path.append('.')
from base import common_base

####################################################################################################################
class SteerAnalysis(common_base.CommonBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, input_file='', config_file='', output_dir='', regenerate_graphs=False, use_precomputed_graphs=False, ddp = False, model=None, ext_plot = False, n_part=-1):

        self.config_file = config_file
        self.input_file = input_file
        self.output_dir = output_dir
        self.regenerate_graphs = regenerate_graphs
        self.use_precomputed_graphs = use_precomputed_graphs
        self.ddp = ddp
        self.ext_plot = ext_plot
        self.rank = int(os.getenv("LOCAL_RANK", "0"))

        self.initialize(config_file)
        if n_part == -1: self.n_part = self.config['n_part']
        else:  self.n_part = n_part
        
        if model is None: self.models = self.config['models']
        else: self.models = [model]
        self.n_events = self.config['n_train'] + self.config['n_val'] + self.config['n_test']
        if self.rank == 0:
            print()
            print(self)


    #---------------------------------------------------------------
    # Initialize config
    #---------------------------------------------------------------
    def initialize(self, config_file):
        if self.rank == 0:
            print('Initializing class objects')

        with open(config_file, 'r') as stream:
            self.config = yaml.safe_load(stream)


    #---------------------------------------------------------------
    # Main function
    #---------------------------------------------------------------
    def run_analysis(self):
        t_start = time.time()

        # DEBUGGIN
        #utils.DataLoader(n_events=100000)
        # DEBUGGIN

        for model in self.models: 
            if model in ['EdgeNet', 'GATAE']:
                for graph_structure in self.config[model]['graph_types']: 
                    graph_key = f'graphs_pyg_SB__{graph_structure}'
                    path = os.path.join(self.output_dir, f'{graph_key}.pt')
                    
                    # this will create both the graphs for pure bkg (SB) and signal+bkg (SR)
                    if self.rank==0 and self.regenerate_graphs or not os.path.exists(path):
                        utils.construct_graphs(output_dir=self.output_dir, graph_structure=graph_structure, n_events = self.n_events, n_part = self.n_part)
                break # only one model for now

        if self.rank==0:
            print()
            print('========================================================================')
            print('Running ML analysis...')


        analysis = ml_analysis.MLAnalysis(self.config_file, self.output_dir, self.ddp, self.models, self.ext_plot, n_part=self.n_part)
        analysis.train_models()
    
        if self.rank==0:
            print()
            print('========================================================================')
            print(f'Analysis completed in {time.time()-t_start} seconds')
            print('========================================================================')
            print()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Graph Generation')
    parser.add_argument('-c', '--config_file', 
                        help='Path of config file for analysis',
                        action='store', type=str,
                        default='config/config.yaml', )
    parser.add_argument('-i' ,'--input_file', 
                        help='Path to subjets_unshuffled.h5 file with ndarrays for ML input',
                        action='store', type=str,
                        default='', )
    parser.add_argument('-o', '--output_dir',
                        help='Output directory for output to be written to',
                        action='store', type=str,
                        default='/pscratch/sd/d/dimathan/LHCO/GAE/graphs/', )
    parser.add_argument('--regenerate_graphs', 
                        help='construct graphs from subjets_unshuffled.h5', 
                        action='store_true', default=False)
    parser.add_argument('--use_precomputed_graphs', 
                        help='use graphs from subjets_unshuffled.h5', 
                        action='store_true', default=False)
    parser.add_argument('--model', 
                        help='Model to use for analysis', 
                        action='store', type=str,
                        default=None, )
    parser.add_argument('--multi', action='store_true', default=False,help='Mutli-GPU training')
    parser.add_argument('--ext_plot', action='store_true',default=False, help='plot auc vs loss per epoch: VERY SLOW')
    parser.add_argument('--n_part', action='store', type=int, default=-1, help='Number of particles in each event, for -1 it will be read from config file')

    
    args = parser.parse_args()

    # If invalid config_file or input_file is given, exit
    if not os.path.exists(args.config_file):
        print(f'Config file {args.config_file} does not exist! Exiting!')
        sys.exit(0)

    # If output dir does not exist, create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)


    analysis = SteerAnalysis(input_file= '' if not os.path.exists(args.input_file) else args.input_file,
                             config_file=args.config_file, 
                             output_dir=args.output_dir, 
                             regenerate_graphs=args.regenerate_graphs,
                             use_precomputed_graphs=args.use_precomputed_graphs,
                             ddp=args.multi, model = args.model, ext_plot = args.ext_plot, n_part = args.n_part)
    analysis.run_analysis()