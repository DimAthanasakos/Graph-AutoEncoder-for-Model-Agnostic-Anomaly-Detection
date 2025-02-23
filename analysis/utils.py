#!/usr/bin/env python3

"""
The graph_constructor module constructs the input graphs to the ML analysis:
    - graphs_numpy_subjet.h5: builds graphs from JFN output subjets_unshuffled.h5
    - graphs_pyg_subjet__{graph_key}.pt: builds PyG graphs from subjet_graphs_numpy.h5
    - graphs_pyg_particle__{graph_key}.pt: builds PyG graphs from energyflow dataset
"""

import os
import sys
import tqdm
import yaml
import numpy as np
import numba

import energyflow as ef

import torch
import torch_geometric
#from torch_geometric.data import Data, Dataset

import h5py as h5
import json, time
from sklearn.utils import shuffle
import time

# --- Your pairwise_lv_fts function ---
def pairwise_lv_fts(xi, xj, num_outputs=4, eps=1e-8):
    """
    Computes logarithmic differences between pairs of node features.
    Assumes that xi and xj are tensors of shape [E, 3] representing (pt, rap, phi)
    for each edge. Returns a tensor of shape [E, num_outputs_out] where num_outputs_out
    is 1 if num_outputs==1, or 3 if num_outputs > 1 (modify as needed).
    """
    # Split each node's feature vector into its components.
    # (Assumes the ordering is [pt, rap, phi].)
    pti, rapi, phii = xi.split((1, 1, 1), dim=1)
    ptj, rapj, phij = xj.split((1, 1, 1), dim=1)
    
    # Compute the squared angular distance.
    delta = delta_r2(rapi, phii, rapj, phij).sqrt()  # Ensure delta_r2 is defined
    lndelta = torch.log(delta.clamp(min=eps))
    
    if num_outputs == 1:
        return lndelta

    # For more outputs, compute additional features.
    if num_outputs < 4:
        ptmin = torch.minimum(pti, ptj)
        lnkt = torch.log((ptmin * delta).clamp(min=eps))
        lnz = torch.log((ptmin / (pti + ptj).clamp(min=eps)).clamp(min=eps))
        # Here we return three outputs. (If you want four outputs, add another feature.)
        outputs = [lndelta, lnkt, lnz ]
        # Concatenate along the feature dimension.
        return torch.cat(outputs, dim=1)
    
    if num_outputs == 4: 
        ptmin = torch.minimum(pti, ptj)
        lnkt = torch.log((ptmin * delta).clamp(min=eps))
        lnz = torch.log((ptmin / (pti + ptj).clamp(min=eps)).clamp(min=eps))
        lnm2 = torch.log(m2(pti, rapi, phii, ptj, rapj, phij).clamp(min=eps))
        outputs = [lndelta, lnkt, lnz, lnm2]
        return torch.cat(outputs, dim=1)

# --- A dummy delta_r2 function ---
def delta_r2(rapi, phii, rapj, phij):
    """
    Computes the squared angular distance delta^2 between two sets of (rap, phi).
    For phi, we use a simple difference and assume that the angles are already in the 
    correct range (or apply your preferred wrapping method).
    """
    # Here we assume simple Euclidean difference; adjust if you need proper angle wrapping.
    return (rapi - rapj) ** 2 + (phii - phij) ** 2


def m2(pti, rapi, phii, ptj, rapj, phij):
    """
    Computes the squared invariant mass between two sets of (pt, rap, phi).
    """

    m2 = pti * ptj * delta_r2(rapi, phii, rapj, phij)**2
    return m2

# Create a Laman Graph using a mod of the k nearest neighbors algorithm.
def laman_knn(x, angles = 0, extra_info = False):   
    # check if x is a numpy array, if not convert it to a numpy array
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(x, np.ndarray):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = torch.from_numpy(x).to(device)
    else:  device  = x.device

    batch_size, _, _, num_particles = x.size()
    x = x.reshape(2*batch_size, -1, num_particles)

    batch_size = 2*batch_size  # effective batch size

    non_zero_particles = torch.norm(x, p=2, dim=1) != 0
    valid_n = non_zero_particles.sum(axis = 1)

    pt, rapidity, phi, = x.split((1, 1, 1), dim=1)

    x = torch.cat((rapidity, phi), dim=1) # (batch_size, 2, num_points)

    t_pairwise = time.time()
    inner = -2 * torch.matmul(x.transpose(2, 1), x)                                    # x.transpose(2, 1): flips the last two dimensions
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)                               # (batch_size, num_points, num_points)
    
    # Connect the 3 hardest particles in the jet in a triangle 
    idx_3 = pairwise_distance[:, :3, :3].topk(k=3, dim=-1) # (batch_size, 3, 2)
    idx_3 = [idx_3[0][:,:,1:], idx_3[1][:,:,1:]] # (batch_size, 3, 1)
    
    # Connect the rest of the particles in a Henneberg construction: Connect the i-th hardest particle with the 2 closest particles, i_1 and i_2, where i_1,2 < j  
    pairwise_distance = pairwise_distance[:, 3:, :] # Remove the pairwise distances of 3 hardest particles from the distance matrix 
    
    # Make the upper right triangle of the distance matrix infinite so that we don't connect the i-th particle with the j-th particle if i > j 
    pairwise_distance = torch.tril(pairwise_distance, diagonal=2) - torch.triu(torch.ones_like(pairwise_distance)*float('inf'), diagonal=3)  # -inf because topk indices return the biggest values -> we've made all distances negative 

    # Find the indices of the 2 nearest neighbors for each particle
    idx = pairwise_distance.topk(k=2, dim=-1) # It returns two things: values, indices 
    idx = idx[1] # (batch_size, num_points - 3, 2)

    # Concatenate idx and idx_3 to get the indices of the 3 hardest particles and the 2 nearest neighbors for the rest of the particles
    idx = torch.cat((idx_3[1], idx), dim=1) # (batch_size, num_points, 3)

    # add 3 rows of -inf to the top of the pairwise_distance tensor to make it of shape (batch_size, num_particles, num_particles)
    # this is because we remove the 3 hardest particles from the graph and we don't want to connect them to the rest of the particles
    pairwise_distance = torch.cat((torch.ones((batch_size, 3, num_particles), device = device)*float('-inf'), pairwise_distance), dim=1)

    # Initialize a boolean mask with False (indicating no connection) for all pairs
    bool_mask = torch.zeros((batch_size, num_particles, num_particles), dtype=torch.bool).to(device)
    # Efficiently populate the boolean mask based on laman_indices
    for i in range(2):  # Assuming each particle is connected to two others as per laman_indices
        # Extract the current set of indices indicating connections
        current_indices = idx[:, :, i]

        # Generate a batch and source particle indices to accompany current_indices for scatter_
        batch_indices = torch.arange(batch_size).view(-1, 1).expand(-1, num_particles)
        src_particle_indices = torch.arange(num_particles).expand(batch_size, -1)

        # Use scatter_ to update the bool_mask; setting the connection locations to True
        bool_mask[batch_indices, src_particle_indices, current_indices] = True

    # ensure that the adjacency matrix is lower diagonal, useful for when we add angles later at random, to keep track of the connections we remove/already have
    mask_upper = ~torch.triu(torch.ones(num_particles, num_particles, dtype=torch.bool), diagonal=0).to(device)
    bool_mask = bool_mask & mask_upper.unsqueeze(0)

    # Remove the padded particles from the graph to save memory space when converting to sparse representation.
    range_tensor = torch.arange(num_particles, device = device).unsqueeze(0).unsqueeze(-1)  
    expanded_valid_n = valid_n.unsqueeze(-1).unsqueeze(-1)
    mask = (range_tensor >= expanded_valid_n).to(device)
    final_mask = mask | mask.transpose(1, 2)


    bool_mask = bool_mask & ~final_mask
    # Remove some angles at random between the particles. Default value of angles = 0.
    #bool_mask = angles_laman(x, bool_mask, angles, pairwise_distance = pairwise_distance) 

    # Make the Laman Edges bidirectional 
    bool_mask = bool_mask | bool_mask.transpose(1, 2)

    # transform to numpy. That's because we later transform everything on the dataset to pytorch 
    # TODO: Change this code to work with numpy from the start
    #bool_mask = bool_mask.numpy() 

    bool_mask = bool_mask.cpu()
    #if extra_info:
        # Calculate the Shannon Entropy and the number of connected components
    #    connected_components(bool_mask, x)
    #    shannon_entropy(bool_mask, x)

    bool_mask=bool_mask.reshape(-1, 2, num_particles, num_particles)

    return bool_mask 



def get_mjj_mask(mjj,use_SR=False,mjjmin=2300,mjjmax=5000, use_area=-1):
    if use_SR:
        mask_region = (mjj>=3300) & (mjj<=3700)
    elif use_area==3:
        print(f'-.'*25)
        print(f'use_high_only')
        print(f'-.'*25)
        mask_region = (mjj>=3700) & (mjj<=mjjmax) # check for Felix
    elif use_area==1:
        print(f'-.'*25)
        print(f'use_low_only')
        print(f'-.'*25)
        mask_region = (mjj>=mjjmin) & (mjj<=3300)
    else:
        mask_region = ((mjj<=3300) & (mjj>=mjjmin)) | ((mjj>=3700) & (mjj<=mjjmax))
    return mask_region

def revert_mjj(mjj,mjjmin=2300,mjjmax=5000):
    x = (mjj + 1.0)/2.0
    logmin = np.log(mjjmin)
    logmax = np.log(mjjmax)
    x = x * ( logmax - logmin ) + logmin
    return np.exp(x)

def prep_mjj(mjj,mjjmin=2300,mjjmax=5000):
    new_mjj = (np.log(mjj) - np.log(mjjmin))/(np.log(mjjmax) - np.log(mjjmin))
    new_mjj = 2*new_mjj -1.0
    return new_mjj



def LoadJson(file_name):
    import json,yaml
    JSONPATH = os.path.join(file_name)
    return yaml.safe_load(open(JSONPATH))

def SaveJson(save_file,data):
    with open(save_file,'w') as f:
        json.dump(data, f)


def _preprocessing(particles, save_json=True, norm = 'mean', scaled = False):
    n_part = particles.shape[2]
    batch_size = particles.shape[0]
    #print(f'particles.shape: {particles.shape}')
    if not scaled:
        return particles.astype(np.float32), # the data is already preprocessed and normalized in the dataset
    elif scaled:
        particles=particles.reshape(-1,particles.shape[-1]) #flatten
    
        #Transformations
        particles[:,0] = np.ma.log(1.0 - particles[:,0]).filled(0)

        if save_json:
            mask = particles[:,-1]
            mean_particle = np.average(particles[:,:-1],axis=0,weights=mask)
            std_particle = np.sqrt(np.average((particles[:,:-1] - mean_particle)**2,axis=0,weights=mask))
            #print(f'===============================')
            #print(f'mean_particle: {mean_particle}')
            #print(f'std_particle: {std_particle}')
            #print(f'===============================')
            data_dict = {
                    'max_particle':np.max(particles[:,:-1],0).tolist(),
                    'min_particle':np.min(particles[:,:-1],0).tolist(),
                    'mean_particle': mean_particle.tolist(),
                    'std_particle': std_particle.tolist(),                     
                    }                
            SaveJson('preprocessing_{}.json'.format(n_part),data_dict)
        else:
            data_dict = LoadJson(f'preprocessing_{n_part}.json')

                
        if norm == 'mean':
            particles[:,:-1]= np.ma.divide(particles[:,:-1]-data_dict['mean_particle'],data_dict['std_particle']).filled(0)
        elif norm == 'min':
            particles[:,:-1]= np.ma.divide(particles[:,:-1]-data_dict['min_particle'],np.array(data_dict['max_particle']) - data_dict['min_particle']).filled(0)            
        else:
            print("ERROR: give a normalization method!")
        particles = particles.reshape(batch_size,2,n_part,-1)

        #print(f'particles.shape: {particles.shape}')
        #print(f'particles[0, :, :5, :3]: ') 
        #print(particles[0, :, :5, :3])

        return particles.astype(np.float32),



def SimpleLoader(data_path,file_name,use_SR=False,
                 n_part=100,mjjmin=2300,mjjmax=5000):


    with h5.File(os.path.join(data_path,file_name),"r") as h5f:
        particles = h5f['constituents'][:, :, :n_part, :]
        jets = h5f['jet_data'][:, :, :n_part]
        mask = h5f['mask'][:, :, :n_part]
        particles = np.concatenate([particles,mask],-1)

    p4_jets = ef.p4s_from_ptyphims(jets)
    # get mjj from p4_jets
    sum_p4 = p4_jets[:, 0] + p4_jets[:, 1]
    mjj = ef.ms_from_p4s(sum_p4)
    jets = np.concatenate([jets,np.sum(mask,-2)],-1)    

    # train using only the sidebands

    mask_region = get_mjj_mask(mjj,use_SR,mjjmin,mjjmax)
    mask_mass = (np.abs(jets[:,0,0])>0.0) & (np.abs(jets[:,1,0])>0.0)
    
    particles = particles[(mask_region) & (mask_mass)]
    mjj = mjj[(mask_region) & (mask_mass)]
    jets = jets[(mask_region) & (mask_mass)]
    jets[:,:,-1][jets[:,:,-1]<0] = 0.
    
    if not use_SR: # should nt this be the 10% not used for training in order to have a fair comparison?
        #Load validation split
        particles = particles[:int(0.9*jets.shape[0])]
        mask = mask[:int(0.9*jets.shape[0])]
        jets = jets[:int(0.9*jets.shape[0])]
    
    return particles,jets,mjj



def class_loader(data_path='/pscratch/sd/d/dimathan/LHCO/Data', 
                 file_name='processed_data_background_rel.h5' , # the default file name for the background data
                 n_part=279,
                 use_SR=True,
                 nsig=15000,
                 nbkg=120000, # its actually around ~100k in the dataset
                 mjjmin=2300,
                 mjjmax=5000):
    

    parts_bkg, jets_bkg, mjj_bkg = SimpleLoader(data_path, file_name, use_SR=use_SR, n_part=n_part)
    #print(f'parts_bkg shape: {parts_bkg.shape}')
    parts_bkg = parts_bkg[:nbkg]
    mjj_bkg = mjj_bkg[:nbkg]
    jets_bkg = jets_bkg[:nbkg]

    if nsig>0:
        parts_sig,jets_sig,mjj_sig = SimpleLoader(data_path, 'processed_data_signal_rel.h5', use_SR=use_SR, n_part=n_part)
        #print(f'parts_sig shape: {parts_sig.shape}')
        parts_sig = parts_sig[:nsig]
        mjj_sig = mjj_sig[:nsig]
        jets_sig = jets_sig[:nsig]

        labels = np.concatenate([np.zeros_like(mjj_bkg), np.ones_like(mjj_sig)])
        particles = np.concatenate([parts_bkg, parts_sig],0)
        jets = np.concatenate([jets_bkg, jets_sig],0)
        mjj = np.concatenate([mjj_bkg, mjj_sig],0)
    else:
        labels = np.zeros_like(mjj_bkg)
        particles = parts_bkg
        jets = jets_bkg
        mjj = mjj_bkg

    return jets, particles, mjj, labels



def DataLoader(n_events,
               data_path='/pscratch/sd/d/dimathan/LHCO/Data', 
               file_name='processed_data_background_rel.h5' , # the default file name for the background data
               n_part=279,
               n_events_sample = 500,
               ddp = False,
               rank=0,size=1,
               batch_size=64,
               make_torch_data=True,
               use_SR=False,
               norm = None,
               use_area=2,
               mjjmin=2300,
               mjjmax=5000,):

    with h5.File(os.path.join(data_path,file_name),"r") as h5f:
        nevts = min(n_events, h5f['jet_data'][:].shape[0])          # number of events
        particles = h5f['constituents'][:, :, :n_part, :]          # particles
        jets = h5f['jet_data'][:, :, :n_part]
        mask = h5f['mask'][:, :, :n_part]
        particles = np.concatenate([particles,mask],-1)

    # print the analytics of the data
    if rank == 0:
        print(f'nevts: {nevts}')
        print(f'Particles shape: {particles.shape}')
        print(f'Jets shape: {jets.shape}')
        print()

  
    p4_jets = ef.p4s_from_ptyphims(jets)
    # get mjj from p4_jets
    sum_p4 = p4_jets[:, 0] + p4_jets[:, 1]
    mjj = ef.ms_from_p4s(sum_p4)
    jets = np.concatenate([jets,np.sum(mask,-2)],-1)

    # train using only the sidebands
    mask_region = get_mjj_mask(mjj,use_SR=False, use_area=use_area)              # Only events in the sideband region (~80% of the events)
    mask_mass = (np.abs(jets[:,0,0])>0.0) & (np.abs(jets[:,1,0])>0.0) # Both jet have non-zero p_T (not zero padded events)
  
    #how many events outside mask_region and mask_mass
    if rank == 0:
        print(f'Number of events outside mask_region: {np.sum(~mask_region)}')
        print()
        print(f'Number of events inside mask_region: {np.sum(mask_region)}')
        print()


    particles = particles[(mask_region) & (mask_mass)]
    mjj = mjj[(mask_region) & (mask_mass)]
    jets = jets[(mask_region) & (mask_mass)]

    jets[:,:,-1][jets[:,:,-1]<0] = 0.

    # should we preprocess the data before constructing the graphs?
    #particles, jets = _preprocessing(particles, jets, mjj)
    #mjj = prep_mjj(mjj, mjjmin, mjjmax)

    return particles[:nevts], jets[:nevts], mjj[:nevts]



#---------------------------------------------------------------
# Construct graphs from input_data and write them to file
#---------------------------------------------------------------
def construct_graphs(output_dir, use_precomputed_graphs=False, sub_or_part='particle', graph_structure='fully_connected', n_events=12000, n_part=100, pair_input_dim=4, subjets = False):
    '''
    Construct graphs:
      - Particle graphs are constructed from energyflow dataset
      - Subjet graphs are constructed from JFN dataset

    Several graph structures are generated:
      - Subjet graphs: Fully connected, Laman graphs (naive, 1N, 1N2N)
      - Particle graphs: Fully connected

    There are several different feature constructions as well:
      - Node features: 
          - Subjet graphs: (z)
          - Particle graphs: (z,y,phi)
      - Edge features:
          - Subjet graphs: pairwise angles
          - Particle graphs: no edge features
    TODO: implement more comprehensive options

    The graphs are saved in several formats:
      - graphs_numpy_subjet.h5: numpy arrays
      - graphs_pyg_subjet__{graph_key}.pt: PyG data objects
      - graphs_pyg_particle__{graph_key}.pt: PyG data objects
    '''
    t_st = time.time()
    # PyG format
    _construct_particle_graphs_pyg(output_dir, graph_structure, n_events=n_events, use_SR=False, n_part=n_part, num_outputs=pair_input_dim, subjets = subjets)
    _construct_particle_graphs_pyg(output_dir, graph_structure, use_SR=True, n_part = n_part, num_outputs=pair_input_dim, subjets = subjets)

    print(f'Finished constructing graphs in {time.time() - t_st:.2f} seconds.')
    



#---------------------------------------------------------------
# Construct graphs from input_data and write them to file
#---------------------------------------------------------------
def _construct_particle_graphs_pyg(output_dir, graph_structure, n_events=500000, rank=0, use_SR=False, n_part=10, num_outputs=4, subjets = False):
    '''
    Construct a list of PyG graphs for the particle-based GNNs, loading from the energyflow dataset

    Graph structure:
        - Nodes: particle four-vectors
        - Edges: no edge features
        - Connectivity: fully connected (TODO: implement other connectivities)
    '''
    print(f'Constructing PyG particle graphs from energyflow dataset...')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'subjets = {subjets}')
    # Load data
    if not subjets: 
        if not use_SR:
            particles, jets, mjj = DataLoader(n_events=n_events, rank=rank, n_part=n_part)
            labels = [0]*len(particles)       
        else:  
            n_dataset = 10000
            _, particles, _, labels = class_loader(use_SR=True, nbkg = n_dataset, nsig = n_dataset, n_part=n_part)
    if subjets: 
        file = '/pscratch/sd/d/dimathan/LHCO/Data/subjets/subjets_100000'
        file = file + '/SR.h5' if use_SR else file + '/SB.h5'
        with h5.File(file, 'r') as h5f: 
            particles = h5f[f'{n_part}'][:n_events if not use_SR else 20000] 
            labels = h5f['labels'][:n_events if not use_SR else 20000] 
        print(f'loaded from {file}')
    
    print(f'particles shape: {particles.shape}')
    print(f'labels shape: {len(labels)}')

    # preprocess the data before constructing the fully connected graphs 
    particles_old = particles
    particles, = _preprocessing(particles, norm = 'mean', scaled = True)

    if use_SR:
        graph_key = f'SR__{graph_structure}_{n_part}'
    else:
        graph_key = f'SB__{graph_structure}_{n_part}'

    graph_list = []
    n_events = particles.shape[0]
    n_part = particles.shape[2]

    # Option: choose whether to use the lv feature function.
    use_lv_feats = True   # Set to False to use simple difference instead.
    #num_outputs = 3       # For pairwise_lv_fts, typically 3 outputs (lnkt, lnz, lndelta)

    if graph_structure=='fully_connected':
        # Precompute edge_index once
        edge_pairs = [[i, j] for i in range(n_part) for j in range(n_part) if i != j]
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous().to(device)

        for event_idx in range(particles.shape[0]):
            event_graphs  = []
            event_label = labels[event_idx] 
            for jet_idx in range(2):  # Two jets per event
                # Get particle features for the current jet
                jet_particles = particles[event_idx, jet_idx][:, :3]  # Shape: (num_particles, (pt, eta, phi))
                jet_particles_old = particles_old[event_idx, jet_idx][:, :3]  # Shape: (num_particles, (pt, eta, phi))
                # Node features
                x = torch.tensor(jet_particles, dtype=torch.float).to(device)  # Shape: (n_part, features)
                x_old = torch.tensor(jet_particles_old, dtype=torch.float).to(device)  # Shape: (n_part, features)
                # Create PyG Data object
                data = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_index_fc=edge_index)
                data.y = torch.tensor([event_label], dtype=torch.long).to(device) # Add label, 0 if bkg, 1 if signal
                data.match = torch.tensor(jet_particles, dtype=torch.float).to(device)  # Shape: (n_part, features)
                # Compute the edge attribute for each edge.
                # Compute edge attributes.
                if use_lv_feats:
                    # For each edge, extract the corresponding node features.
                    xi = x_old[edge_index[0]]  # source node features (shape: [E, 3])
                    xj = x_old[edge_index[1]]  # target node features (shape: [E, 3])
                    # Compute custom pairwise features.
                    data.edge_attr = pairwise_lv_fts(xi, xj, num_outputs=num_outputs)
                else:
                    # Default: use the difference between node features.
                    data.edge_attr = x[edge_index[0]] - x[edge_index[1]]
                
                event_graphs.append(data)
                
            graph_list.append(event_graphs)

    elif graph_structure=='laman':
        # Convert the particles array to a torch tensor on the target device. particles shape: (6000, 2, 10, 4)
        particles_t = torch.tensor(particles, dtype=torch.float, device=device)
        # Use only the first 3 features for each particle.
        particles_t = particles_t[..., :3]  # shape becomes (6000, 2, 10, 3)
        # Rearrange dimensions so that the features come before the particles: We want shape (n_events, 2, num_features, n_part), i.e. (6000, 2, 3, 10)
        particles_t = particles_t.permute(0, 1, 3, 2)

        # The input has shape (6000, 2, 3, 10) and inside laman_knn it will be reshaped to (12000, 3, 10).
        bool_mask = laman_knn(particles_t)
        # We expect bool_mask to have shape (12000, 10, 10) (one 10x10 mask for each jet).
        # Reshape it back so that it is grouped by event:
        bool_mask = bool_mask.reshape(particles_t.size(0), particles_t.size(1), particles_t.size(-1), particles_t.size(-1))
        # Now bool_mask has shape: (6000, 2, 10, 10)

        graph_list = []
        n_events = particles_t.size(0)  # 6000 events
        n_part = particles_t.size(-1)   # 10 particles per jet


        edge_pairs_fc = [[i, j] for i in range(n_part) for j in range(n_part) if i != j]
        edge_index_fc = torch.tensor(edge_pairs_fc, dtype=torch.long).t().contiguous().to(device)

        for event_idx in range(n_events):
            event_graphs = []
            event_label = labels[event_idx]
            for jet_idx in range(2):  # Two jets per event
                # particles_t[event_idx, jet_idx] has shape (3, 10): (features, particles).
                # We need node features as (n_part, num_features) i.e. (10, 3).
                x = particles_t[event_idx, jet_idx].transpose(0, 1)  # shape: (10, 3)
                jet_particles_old = particles_old[event_idx, jet_idx][:, :3]  # Shape: (num_particles, (pt, eta, phi))
                
                x_old = torch.tensor(jet_particles_old, dtype=torch.float).to(device)  # Shape: (n_part, features)
                # Retrieve the boolean adjacency matrix for this jet.
                bm = bool_mask[event_idx, jet_idx]  # shape: (10, 10)
                # Convert the boolean mask to edge_index.
                edge_index = torch.nonzero(bm, as_tuple=False).t().contiguous()

                # Create the PyG Data object.
                data = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_index_fc=edge_index_fc)
                data.y = torch.tensor([event_label], dtype=torch.long, device=device)
                # For each edge, extract the corresponding node features.
                xi = x_old[edge_index[0]]  # source node features (shape: [E, 3])
                xj = x_old[edge_index[1]]  # target node features (shape: [E, 3])

                data.edge_attr = pairwise_lv_fts(xi, xj, num_outputs=num_outputs)
                event_graphs.append(data)
            graph_list.append(event_graphs)


    # Save to file using pytorch
    graph_filename = os.path.join(output_dir, f"graphs_pyg_{graph_key}.pt")
    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(graph_filename), exist_ok=True)

    torch.save(graph_list, graph_filename)
    print(f'Saved PyG graphs to {graph_filename}.')


