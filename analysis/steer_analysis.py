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
    def __init__(self, input_file='', config_file='', output_dir='', regenerate_graphs=False, use_precomputed_graphs=False, ddp = False, **kwargs):

        self.config_file = config_file
        self.input_file = input_file
        self.output_dir = output_dir
        self.regenerate_graphs = regenerate_graphs
        self.use_precomputed_graphs = use_precomputed_graphs
        self.ddp = ddp
        self.rank = int(os.getenv("LOCAL_RANK", "0"))

        self.initialize(config_file)
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
        self.models = self.config['models']


    #---------------------------------------------------------------
    # Main function
    #---------------------------------------------------------------
    def run_analysis(self):
        '''
        For now, we will assume that MC samples have been used to
        generate a dataset of particles and subjets (subjets_unshuffled.h5).
        
        Existing datasets are listed here:
        https://docs.google.com/spreadsheets/d/1DI_GWwZO8sYDB9FS-rFzitoDk3SjfHfgoKVVGzG1j90/edit#gid=0
        
        The graph_constructor module constructs the input graphs to the ML analysis:
          - graphs_numpy_subjet.h5: builds graphs from JFN output subjets_unshuffled.h5
          - graphs_pyg_subjet__{graph_key}.pt: builds PyG graphs from subjet_graphs_numpy.h5
          - graphs_pyg_particle__{graph_key}.pt: builds PyG graphs from energyflow dataset
        '''

        # If you want to use subjets instead of hadrons, we also need an input file to read the subjets from.
        # If so, the input file must be one of the files in the google spreadsheet: https://docs.google.com/spreadsheets/d/1DI_GWwZO8sYDB9FS-rFzitoDk3SjfHfgoKVVGzG1j90/edit#gid=0
        t_start = time.time()
        for model in self.models: 
            for graph_structure in self.config[model]['graph_types']: 
                graph_key = f'graphs_pyg_SB__{graph_structure}'
                path = os.path.join(self.output_dir, f'{graph_key}.pt')
                
                # this will create both the graphs for pure bkg (SB) and signal+bkg (SR)
                if self.rank==0 and self.regenerate_graphs or not os.path.exists(path):
                    utils.construct_graphs(output_dir=self.output_dir, graph_structure=graph_structure)

        if self.rank==0:
            print()
            print('========================================================================')
            print('Running ML analysis...')


        analysis = ml_analysis.MLAnalysis(self.config_file, self.output_dir, self.ddp)
        analysis.train_models()
    
        if self.rank==0:
            print()
            print('========================================================================')
            print(f'Analysis completed in {time.time()-t_start} seconds')
            print('========================================================================')
            print()


        # check if the code is running in parallel, and only plot results for one process (same results for all processes)



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
    
    parser.add_argument('--multi', action='store_true', default=False,help='Mutli-GPU training')

    
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
                             ddp=args.multi, )
    analysis.run_analysis()