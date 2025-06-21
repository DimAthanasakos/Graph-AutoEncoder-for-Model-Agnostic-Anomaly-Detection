import glob
import torch
#torch.autograd.set_detect_anomaly(True)
import torch.multiprocessing as mp
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

#torch.manual_seed(0)

####################################################################################################################
class SteerAnalysis(common_base.CommonBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, input_file='', config_file='', output_dir='', regenerate_graphs=False, ddp = False, model=None, ext_plot = False, n_part=-1, input_dim=-1, n_runs=1, graph_structure='', subjets = -1, angles = -1, **kwargs):

        self.config_file = config_file
        self.input_file = input_file
        self.output_dir = output_dir
        self.regenerate_graphs = regenerate_graphs
        self.ddp = ddp
        self.ext_plot = ext_plot
        self.rank = int(os.getenv("LOCAL_RANK", "0"))
        self.n_runs = n_runs

        self.initialize(config_file)
        if n_part == -1: self.n_part = self.config['n_part']
        else:  self.n_part = n_part
        self.input_dim = input_dim
        if model is None: self.models = self.config['models']
        else: self.models = [model]
        self.n_events = self.config['n_train'] + self.config['n_val'] 
        if subjets == -1: self.subjets = self.config['subjets']
        else: self.subjets = subjets 
        if self.rank==0: print(f'Using subjets: {self.subjets}')
        self.angles = angles
        
        self.graph_structures = [graph_structure]
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
        model = self.models[0]
        graph_structures = ''
        for n_part in self.n_part:
            if self.rank==0:
                if model in ['EdgeNet', 'RelGAE',  'EdgeNet_laman', 'EdgeNet_edge_VGAE', ]:
                    if self.graph_structures[0] == '': graph_structures= self.config[model]['graph_types']
                    else: graph_structures = self.graph_structures
                    unsupervised = self.config[model]['unsupervised'] if 'unsupervised' in self.config[model] else False
                    
                    for graph_structure in graph_structures: 
                        if graph_structure=='unique' and model in ['RelGAE', 'EdgeNet_laman', 'EdgeNet_edge_VGAE']:
                            edge_addition = self.config[model]['edge_addition']
                            if not unsupervised:
                                graph_key = f'graphs_pyg_SB__{graph_structure}_{edge_addition}_{n_part}'
                            elif unsupervised:
                                graph_key = f'graphs_pyg_SR__{graph_structure}_{edge_addition}_{n_part}_unsupervised'
                        else:
                            if not unsupervised:
                                graph_key = f'graphs_pyg_SB__{graph_structure}_{n_part}'
                            elif unsupervised:
                                graph_key = f'graphs_pyg_SR__{graph_structure}_{n_part}_unsupervised'
                                
                        print('--------'*5)
                        print(f'Using unsupervised: {unsupervised}')
                        print(f'graph_key: {graph_key}')
                        print('--------'*5)

                        path = os.path.join(self.output_dir, f'{graph_key}.pt')
                        if model in ['RelGAE', 'EdgeNet_laman', 'EdgeNet_edge_VGAE', ]: 
                            pair_input_dim = self.config[model]['pair_input_dim']
                            if self.angles == -1:
                                angles = self.config[model]['angles']
                            else: angles = self.angles
                            edge_addition = self.config[model]['edge_addition']
                            print('\n','====='*20)
                            print(f'pair_input_dim: {pair_input_dim}, angles: {angles}, edge_addition: {edge_addition}')
                            print('====='*20, '\n')
                        else: 
                            pair_input_dim = 0
                            angles = 0
                            edge_addition = 0

                        # this will create both the graphs for pure bkg (SB) and signal+bkg (SR)
                        print(f'self.regenerate_graphs: {self.regenerate_graphs}')
                        print(f'path: {path}')
                        print(f'os.path.exists(path): {os.path.exists(path)}')
                        if self.rank==0 and (self.regenerate_graphs or not os.path.exists(path)):
                            utils.construct_graphs(output_dir=self.output_dir, graph_structure=graph_structure, n_events = self.n_events, n_part = n_part, pair_input_dim=pair_input_dim,subjets = self.subjets, angles=angles, num_edges=edge_addition, unsupervised=unsupervised)
                    #break # only one model for now

            
                print()
                print('========================================================================')
                print('Running ML analysis...')

                if model in ['transformer', 'transformer_graph'] and self.input_dim == -1:
                    self.input_dim = self.config[model]['input_dim']
                    
            lst_part = [n_part]
            analysis = ml_analysis.MLAnalysis(self.config_file, self.output_dir, self.ddp, self.models, self.ext_plot, n_part=lst_part, input_dim=self.input_dim, graph_structures=self.graph_structures, n_runs=self.n_runs)
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
    parser.add_argument('-o', '--output_dir', help='Output directory for output to be written to',
                        action='store', type=str,
                        default='/pscratch/sd/d/dimathan/LHCO/GAE/graphs/', )
    parser.add_argument('-regen', '--regenerate_graphs', help='construct graphs from subjets_unshuffled.h5', 
                        action='store_true', default=False)
    parser.add_argument('-m', '--model',  help='Model to use for analysis', 
                        action='store', type=str, default=None, )
    parser.add_argument('--multi', action='store_true', default=False,help='Mutli-GPU training')
    parser.add_argument('--ext_plot', action='store_true',default=False, help='plot auc vs loss per epoch: VERY SLOW')
    parser.add_argument('--n_part', action='store', type=int, default=-1, help='Number of particles in each event, for -1 it will be read from config file')
    parser.add_argument('--input_dim', action='store', type=int, default=-1, help='Number of particles in each event, for -1 it will be read from config file')
    parser.add_argument('-gr', '--graph_structure', action='store', type=str, default='', help='Type of graph to use for analysis')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of independent runs (each with fresh initialization)')
    parser.add_argument('-s', '--subjets', type=int, default=-1, help=' Subjet level or hadron level. 0 for hadron, 1 for subjet')
    parser.add_argument('-a', '--angles', type=int, default=-1, help='Number of angles to add for RelGAE')

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
                             ddp=args.multi, model = args.model, ext_plot = args.ext_plot, 
                             n_part = args.n_part, input_dim = args.input_dim, 
                             graph_structure = args.graph_structure, n_runs = args.n_runs, 
                             subjets = args.subjets,
                             angles = args.angles)
    analysis.run_analysis()
    
    if args.multi: # Only call destroy if DDP was initialized
        torch.distributed.destroy_process_group()

    print("Script finished.") # Optional: Add a final print to confirm exit
