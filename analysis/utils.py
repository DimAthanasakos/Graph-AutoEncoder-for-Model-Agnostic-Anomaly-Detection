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
import math 


def SetStyle():
    from matplotlib import rc
    rc('text', usetex=True)

    import matplotlib as mpl
    rc('font', family='serif')
    rc('font', size=22)
    rc('xtick', labelsize=15)
    rc('ytick', labelsize=15)
    rc('legend', fontsize=15)

    # #
    mpl.rcParams.update({'font.size': 19})
    #mpl.rcParams.update({'legend.fontsize': 18})
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams.update({'xtick.labelsize': 18}) 
    mpl.rcParams.update({'ytick.labelsize': 18}) 
    mpl.rcParams.update({'axes.labelsize': 18}) 
    mpl.rcParams.update({'legend.frameon': False}) 
    mpl.rcParams.update({'lines.linewidth': 2})
    
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.style.use(hep.style.CMS)
     


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

def angles_laman(x, mask, angles = 0, pairwise_distance = None):
    print(f'angles_laman with angles: {angles}')
    batch_size, _, num_particles = mask.size()
    if isinstance(x, np.ndarray):
        non_zero_particles = np.linalg.norm(x, axis=1) != 0
    elif isinstance(x, torch.Tensor):
        non_zero_particles = torch.norm(x, p=2, dim=1) != 0
    
    valid_n = non_zero_particles.sum(axis = 1)
    valid_n_f = valid_n.float()

    # Summing True values for each tensor in the batch
    sum_true_per_batch = torch.sum(mask, dim=[1, 2])  # Sum over the last two dimensions

    # Calculating the average number of True values across the batch
    average_true = torch.mean(sum_true_per_batch.float())  # Convert to float for mean calculation
    # print(f'average_true: {average_true:.2f}')

    # remove angles
    if angles < 0:        

        angles = abs(angles)
        mask = mask.int() # torch.bool is not supported by torch.argmax :(
        for b in range(batch_size):
            # Generate a random permutation of n-2 numbers that starts with 2 
            if valid_n[b] <= 2:
                continue 
            permutation = torch.randperm(valid_n[b]-2) + 2      # Random generator with replacement excluding the 2 hardest particles 
                
            for i in range(min(angles, valid_n[b]-2)):          # Currently we remove angles in such a way so to ensure that the graph is still connected
                index = permutation[i]

                # the first True in the i-th row of the bool mask turned to False. SYSTEMATIC ERROR: We always remove connectivity to the hardest particle 
                first_true_index = torch.argmax(mask[b, index]) # maybe alternate betwen first and second True. ?? 
                mask[b, index, first_true_index] = 0

        mask = mask.bool() # convert back to bool
        

    # Add angles: For a particle i, add an edge to a particle j, where j < i, if there is no edge between i and j.
    elif angles > 0:
        # Mask the positions of the current edges so to make sorting easier 
        pairwise_distance[mask] = -float('inf')
        for b in range(batch_size):
            if valid_n[b] <= 3:
                continue 
            
            # mask the padded particles, i.e n >= valid_n[b]
            pairwise_distance[b, valid_n[b]:, :] = -float('inf')
            pairwise_distance[b, :, valid_n[b]:] = -float('inf')

            # Flatten the matrix
            flat_matrix = pairwise_distance[b].flatten()
            # Sort the flattened matrix
            sorted_values, flat_sorted_indices = torch.sort(flat_matrix, descending = True)
            # Convert flat indices to 2D row and column indices
            row_indices, col_indices = flat_sorted_indices//num_particles, flat_sorted_indices%num_particles
            
            max_angles = math.comb(valid_n[b], 2) - (2*valid_n[b] - 3) # The maximum number of angles we can add until it becomes a fully connected graph
            angles_to_add = min(angles, max_angles)
            mask[b, row_indices[:angles_to_add], col_indices[:angles_to_add]] = True
        
    # Summing True values for each tensor in the batch
    sum_true_per_batch = torch.sum(mask, dim=[1, 2])  # Sum over the last two dimensions

    # Calculating the average number of True values across the batch
    average_true = torch.mean(sum_true_per_batch.float())  # Convert to float for mean calculation

    # print(f'average_true: {average_true:.2f}')

    # if True: 
    #     tot_2n3 = 0
    #     tot_n = 0 
    #     total_n1=0
    #     tot_n2 = 0
    #     total_3n = 0
    #     for b in range(batch_size):
    #         edges = torch.sum(mask[b], dim=[0, 1])
    #         tot_2n3 += edges / (2*valid_n[b]-3)
    #         tot_n += edges / valid_n[b]
    #         tot_n2 += edges / (1/2*valid_n[b]*(valid_n[b]-1))
    #         total_n1 += edges / (valid_n[b]-1)
    #         total_3n += edges / (3*valid_n[b]-6)

    #     av_2n3 = tot_2n3 / batch_size
    #     av_n = tot_n / batch_size
    #     av_n1 = total_n1 / batch_size
    #     av_n2 = tot_n2 / batch_size
    #     av_3n = total_3n / batch_size

    #     if 2 * sum_true_per_batch[b] > valid_n[b]*(valid_n[b]-1) :
    #         print('found a graph with more than 100% connectivity')
    #         print(f'valid_n[b] = {valid_n[b]}')
    #         print(f'edges = {sum_true_per_batch[b] }')
    #         print(f'sum_true_per_batch[b]/valid_n[b]^2 * 2 = {2 * sum_true_per_batch[b] / ( valid_n[b]*(valid_n[b]-1) ) }')

    #     print(f'After addition')
    #     print(f"Average number of edges = {average_true.item()}")
    #     #print(f'edges/2n-3 = {average_true.item()/(2*torch.mean(valid_n_f)-3)}')
    #     print(f'actual edges/2n-3 = {av_2n3}')
    #     print()
    #     print(f'actual edges/(3n-6) = {av_3n:.4f}')
    #     print()
    #     #print(f'edges/n = {average_true.item()/(torch.mean(valid_n_f))}')
    #     print(f'actual edges/n = {av_n}')
    #     print()
    #     print(f'actual edges/n-1= {av_n1}')
    #     print()
    #     #print(f'edges/(n^2/2) = {average_true.item()/(1/2*torch.mean(valid_n_f)**2)}')
    #     print(f'actual edges/(n^2/2) = {av_n2}')

    #     print()
               
    # print(f'mask.shape: {mask.shape}')
    # print(f'mask[0]: {mask[0]}')
    # print()
    # print(f'mask[1]: {mask[1]}')
    # print()
    # print(f'mask[2]: {mask[2]}')
    # print()
    # print(f'mask[3]: {mask[3]}')
    # print()
    # print(f'mask[4]: {mask[4]}')

    return mask 

def unique_graph(x, angles=0, extra_info=False, num_edges=3):
    print(f'unique_graph with num_edges: {num_edges}')
    # Convert x to tensor if it's a numpy array
    if isinstance(x, np.ndarray):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = torch.from_numpy(x).to(device)
    else:
        device = x.device

    batch_size, _, _, num_particles = x.size()
    x = x.reshape(2 * batch_size, -1, num_particles)
    batch_size = 2 * batch_size  # effective batch size

    pt, rapidity, phi = x.split((1, 1, 1), dim=1)
    x = torch.cat((rapidity, phi), dim=1)  # (batch_size, 2, num_points)

    non_zero_particles = torch.norm(x, p=2, dim=1) != 0
    valid_n = non_zero_particles.sum(axis=1)

    # Compute pairwise distance matrix using negative squared Euclidean distance
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (batch_size, num_points, num_points)

    k = num_edges  # number of edges to connect per particle

    # ===== Connect the k hardest particles in the jet =====
    # For the k hardest particles, take top-(k+1) entries along the last dimension, so that after removing self-connection we have k edges.
    idx_hard = pairwise_distance[:, :k+1, :k+1].topk(k=k+1, dim=-1)  # shape: (batch_size, k, k+1)
    # Remove the self connection (assumed to be the first index)
    idx_hard = [idx_hard[0][:, :, 1:], idx_hard[1][:, :, 1:]]  # each now (batch_size, k, k)

    # ===== Process the remaining particles =====
    # Remove the k hardest particles from the distance matrix
    pairwise_distance = pairwise_distance[:, k+1:, :]  # shape: (batch_size, num_particles - k, num_particles)
    # Mask the upper triangle: we want to connect each remaining particle with its k nearest neighbors.
    pairwise_distance = torch.tril(pairwise_distance, diagonal=k) - \
                        torch.triu(torch.ones_like(pairwise_distance) * float('inf'), diagonal=k+1)
    # For the remaining particles, select the top k nearest neighbors
    idx_rest = pairwise_distance.topk(k=k, dim=-1)[1]  # shape: (batch_size, num_particles - k, k)

    # ===== Concatenate indices =====
    idx_full = torch.cat((idx_hard[1], idx_rest), dim=1)  # shape: (batch_size, num_particles, k)

    # Prepend k rows of -inf to pairwise_distance to recover shape (batch_size, num_particles, num_particles)
    pad = torch.ones((batch_size, k, num_particles), device=device) * float('-inf')
    pairwise_distance = torch.cat((pad, pairwise_distance), dim=1)

    # ===== Build the boolean mask for graph connections =====
    bool_mask = torch.zeros((batch_size, num_particles, num_particles), dtype=torch.bool).to(device)
    # Populate the mask: for each particle, add k edges using the indices from idx_full
    for i in range(k):
        current_indices = idx_full[:, :, i]  # shape: (batch_size, num_particles)
        batch_indices = torch.arange(batch_size).view(-1, 1).expand(-1, num_particles)
        src_particle_indices = torch.arange(num_particles).expand(batch_size, -1)
        bool_mask[batch_indices, src_particle_indices, current_indices] = True

    # Make adjacency matrix lower triangular
    mask_upper = ~torch.triu(torch.ones(num_particles, num_particles, dtype=torch.bool), diagonal=0).to(device)
    bool_mask = bool_mask & mask_upper.unsqueeze(0)

    # Remove padded particles to save memory in sparse representation
    range_tensor = torch.arange(num_particles, device=device).unsqueeze(0).unsqueeze(-1)
    expanded_valid_n = valid_n.unsqueeze(-1).unsqueeze(-1)
    mask = (range_tensor >= expanded_valid_n).to(device)
    final_mask = mask | mask.transpose(1, 2)
    bool_mask = bool_mask & ~final_mask

    # Call the angles_laman function (if defined) to remove some angles at random, if desired
    bool_mask = angles_laman(x, bool_mask, angles, pairwise_distance=pairwise_distance)

    # Make edges bidirectional
    bool_mask = bool_mask | bool_mask.transpose(1, 2)

    return bool_mask.cpu()



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
    bool_mask = angles_laman(x, bool_mask, angles, pairwise_distance = pairwise_distance) 

    # Make the Laman Edges bidirectional 
    # print(f'bool_mask before bidirectional')
    # print(f'bool_mask.shape: {bool_mask.shape}')
    # print(f'bool_mask[0]: {bool_mask[0]}')
    bool_mask = bool_mask | bool_mask.transpose(1, 2)
    # print()
    # print(f'bool_mask[0]: {bool_mask[0]}')

    bool_mask = bool_mask.cpu()

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
            SaveJson('preprocessing/preprocessing_{}.json'.format(n_part),data_dict)
        else:
            data_dict = LoadJson(f'preprocessing/preprocessing_{n_part}.json')

                
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
                 n_part=100,
                 mjjmin=2300,mjjmax=5000, 
                 unsupervised=False):


    with h5.File(os.path.join(data_path,file_name),"r") as h5f:
        particles = h5f['constituents'][:, :, :n_part, :]
        jets = h5f['jet_data'][:, :, :n_part]
        mask = h5f['mask'][:, :, :n_part]
        particles = np.concatenate([particles,mask],-1)
    
    print('-----------------------------')
    # If unsupervised, we use the full mjj range
    # else we follow the instructions of the use_SR flag if we want to use the signal region or the SB 
    print(f'simple loader with use_SR: {use_SR} and unsupervised: {unsupervised}, file_name: {file_name}')
    print(f'particles shape: {particles.shape}')

    p4_jets = ef.p4s_from_ptyphims(jets)

    # get mjj from p4_jets
    sum_p4 = p4_jets[:, 0] + p4_jets[:, 1]
    mjj = ef.ms_from_p4s(sum_p4)
    jets = np.concatenate([jets,np.sum(mask,-2)],-1)    
    
    # train using only the sidebands for supervised, otherwise use the full mjj range
    if not unsupervised:
        mask_region = get_mjj_mask(mjj,use_SR,mjjmin,mjjmax)
        mask_mass = (np.abs(jets[:,0,0])>0.0) & (np.abs(jets[:,1,0])>0.0)
        
        particles = particles[(mask_region) & (mask_mass)]
        mjj = mjj[(mask_region) & (mask_mass)]
        jets = jets[(mask_region) & (mask_mass)]
        jets[:,:,-1][jets[:,:,-1]<0] = 0.
    
    print(f'after masking with mass')
    print(f'particles shape: {particles.shape}')


    particles = particles[:, :, :, :3] # keep only the first 3 features
    print(f'particles shape: {particles.shape}')
    print('-----------------------------')
    return particles,jets,mjj



def class_loader(data_path='/pscratch/sd/d/dimathan/LHCO/Data', 
                 file_name='processed_data_background_rel.h5' , # the default file name for the background data
                 n_part=279,
                 use_SR=True,
                 nsig=15000,
                 nbkg=120000, # its actually around ~100k in the dataset
                 mjjmin=2300,
                 mjjmax=5000, 
                 unsupervised=False):
    

    parts_bkg, jets_bkg, mjj_bkg = SimpleLoader(data_path, file_name, use_SR=use_SR, n_part=n_part, unsupervised=unsupervised)
    #print(f'parts_bkg shape: {parts_bkg.shape}')
    parts_bkg = parts_bkg[:nbkg]
    mjj_bkg = mjj_bkg[:nbkg]
    jets_bkg = jets_bkg[:nbkg]

    if nsig>0:
        parts_sig,jets_sig,mjj_sig = SimpleLoader(data_path, 'processed_data_signal_rel.h5', use_SR=use_SR, n_part=n_part, unsupervised=unsupervised)
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
               mjjmax=5000,
               unsupervised=False,):

    with h5.File(os.path.join(data_path,file_name),"r") as h5f:
        nevts = min(n_events, h5f['jet_data'][:].shape[0])          # number of events
        particles = h5f['constituents'][:, :, :n_part, :]           # particles
        jets = h5f['jet_data'][:, :, :n_part]
        mask = h5f['mask'][:, :, :n_part]
        particles = np.concatenate([particles,mask],-1)

        
    p4_jets = ef.p4s_from_ptyphims(jets)
    # get mjj from p4_jets
    sum_p4 = p4_jets[:, 0] + p4_jets[:, 1]
    mjj = ef.ms_from_p4s(sum_p4)
    

    jets = np.concatenate([jets,np.sum(mask,-2)],-1)

    # train using only the sidebands for weakly-supervised
    if not unsupervised:
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
    
    # for unsupervised, we use the full mjj range since we do not do a split of SB and SR 

    jets[:,:,-1][jets[:,:,-1]<0] = 0.

    # should we preprocess the data before constructing the graphs?
    #particles, jets = _preprocessing(particles, jets, mjj)
    #mjj = prep_mjj(mjj, mjjmin, mjjmax)

    return particles[:nevts], jets[:nevts], mjj[:nevts]


########################################################
# Load the jet observables from the precomputed dataset 
########################################################
def load_jet_observables(n_events, use_SR=False):
    """
    Load precomputed jet observables for each event from an HDF5 file.
    Assumes that the observables (mass, pt, n_particles, tau21, tau32) for each jet
    are stored in a dataset with key f"{n_part}_observables". There are 2 jets per event.

    Parameters:
       n_part (int): Number of particles (used in the original preprocessing).
       n_events (int): Number of events to load.
       use_SR (bool): If True, load signal region data; otherwise, sideband.

    Returns:
       observables (np.ndarray): Array of shape (n_events, 2, 5) containing the observables.
    """
    import h5py as h5

    base_file = '/pscratch/sd/d/dimathan/LHCO/Data/subjets/subjets_and_global_20000'
    file_path = base_file + '/SR.h5' if use_SR else base_file + '/SB.h5'
    with h5.File(file_path, 'r') as h5f:
        # Assumes observables stored as a dataset with key e.g. "100_observables" 
        # if n_part==100. Adjust the key according to your saving routine.
        dataset_key = f'observables'
        if dataset_key not in h5f:
            raise ValueError(f"Dataset {dataset_key} not found in {file_path}")
        obs_data = h5f[dataset_key][:(n_events if not use_SR else 20000)]
        labels = h5f['labels'][:n_events if not use_SR else 20000] 
    #print(f"Loaded jet observables from {file_path}")
    return obs_data, labels


#---------------------------------------------------------------
# Construct graphs from input_data and write them to file
#---------------------------------------------------------------
def construct_graphs(output_dir, use_precomputed_graphs=False, sub_or_part='particle', graph_structure='fully_connected', n_events=12000, n_part=100, pair_input_dim=4, subjets = False, load_global_obs=False, angles = 0, num_edges = 3, unsupervised=False): 
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
    if load_global_obs and not subjets:
        print("Warning: load_global_obs=True has no effect when subjets=False. Global observables are not loaded for particle graphs in this function.")
        load_global_obs = False # Ensure flag is off if not loading subjets

    # --- Construct Sideband Graphs ---
    if not unsupervised:
        print("\n--- Constructing Sideband (SB) Graphs ---")
        _construct_particle_graphs_pyg(
            output_dir=output_dir,
            graph_structure=graph_structure,
            n_events=n_events,
            use_SR=False,
            n_part=n_part,
            num_outputs=pair_input_dim,
            subjets=subjets,
            load_global_obs=load_global_obs, # Pass flag
            angles=angles,
            num_edges=num_edges
        )

    # --- Construct Signal Region Graphs ---
    print("\n--- Constructing Signal Region (SR) Graphs ---")
    _construct_particle_graphs_pyg(
        output_dir=output_dir,
        graph_structure=graph_structure,
        n_events=n_events, # Note: actual loaded SR events might differ
        use_SR=True,
        n_part=n_part,
        num_outputs=pair_input_dim,
        subjets=subjets,
        load_global_obs=load_global_obs, # Pass flag
        angles=angles,
        num_edges=num_edges,
        unsupervised=unsupervised,
    )

    print(f'\nFinished constructing graphs in {time.time() - t_st:.2f} seconds.')
    



#---------------------------------------------------------------
# Construct graphs from input_data and write them to file
#---------------------------------------------------------------
def _construct_particle_graphs_pyg(output_dir, graph_structure, n_events=500000, rank=0, use_SR=False, n_part=10, num_outputs=4, subjets = False, load_global_obs=False, angles = 0, num_edges = 3, unsupervised=False):
    '''
    Construct a list of PyG graphs for the particle-based GNNs, loading from the energyflow dataset

    Graph structure:
        - Nodes: particle four-vectors
        - Edges: no edge features
        - Connectivity: fully connected (TODO: implement other connectivities)
    '''
    print(f'Constructing PyG particle graphs from energyflow dataset...')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global_observables = None
    print(f'subjets = {subjets}')
    # Load data
    if not subjets: 
        if not use_SR: # SB
            particles, jets, mjj = DataLoader(n_events=n_events, rank=rank, n_part=n_part)
            labels = [0]*len(particles)       
        else:          # SR
            if unsupervised: n_dataset = n_events
            else: n_dataset = 10000
            _, particles, _, labels = class_loader(use_SR=True, nbkg = n_dataset, nsig = n_dataset, n_part=n_part)
    
    if subjets: 
        if unsupervised: 
            #file = '/pscratch/sd/d/dimathan/LHCO/Data/subjets/rnd_unsupervised_250000'
            if n_part in [5,10,15,20,25]:
                file = '/pscratch/sd/d/dimathan/LHCO/Data/subjets/rnd_unsupervised_125000' # for 5,10,15,20,25 
            elif n_part in [30,35,40,50,75]:
                file = '/pscratch/sd/d/dimathan/LHCO/Data/subjets/rnd_unsupervised_126000' # for 30,35,40,50,75
        else:
            file = '/pscratch/sd/d/dimathan/LHCO/Data/subjets/rnd_subjets_100000'
            
            ############# REMOVE THIS LINE #############
            file = '/pscratch/sd/d/dimathan/LHCO/Data/subjets/rnd_subjets_only_150000'
            #############################################

            if load_global_obs: file='/pscratch/sd/d/dimathan/LHCO/Data/subjets/subjets_and_global_20000'
        file = file + '/SR.h5' if use_SR else file + '/SB.h5'
        max_load = int(1e6) # just load everything since the regen is only meant to be done once. For subsequent runs, we load the graphs from the file.


        with h5.File(file, 'r') as h5f: 
            print(f'h5f.keys(): {h5f.keys()}')
            particles = h5f[f'{n_part}'][:max_load] 
            labels = h5f['labels'][:max_load] 
            print(f'loaded from {file}')
            # Load Global Observables if requested
            if load_global_obs:
                global_obs_key = 'observables' # Key used in the modified process_subjets
                if global_obs_key not in h5f:
                    raise ValueError(f"Global observable dataset key '{global_obs_key}' not found in {file}. "
                                    "Ensure process_subjets was run with compute_subjets_and_global_obs=True.")
                global_observables = h5f[global_obs_key][:max_load] # Shape: (n_events, 2, 5)

    
    print(f'particles shape: {particles.shape}')
    print(f'labels shape: {len(labels)}')
    if global_observables is not None: print(f'Global observables shape: {global_observables.shape}')

    if graph_structure == 'unique':
        if use_SR: graph_key = f'SR__{graph_structure}_{num_edges}_{n_part}{"_unsupervised" if unsupervised else ""}'
        else: graph_key = f'SB__{graph_structure}_{num_edges}_{n_part}{"_unsupervised" if unsupervised else ""}'
    else:
        if use_SR: graph_key = f'SR__{graph_structure}_{n_part}{"_unsupervised" if unsupervised else ""}'
        else: graph_key = f'SB__{graph_structure}_{n_part}{"_unsupervised" if unsupervised else ""}'

    # preprocess the data before constructing the fully connected graphs 
    particles_old = particles
    particles, = _preprocessing(particles, norm = 'mean', scaled = True)

    total_size = particles.shape[0]
    chunk_size = 1024*8
    chunks = (total_size - 1) // chunk_size + 1
    n_part = particles.shape[2]
    final_graph_list = []
    
    for i in range(chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_size)
        print(f"  Processing chunk {i+1}/{chunks} (events {start_idx} to {end_idx})...")

        prtcls = particles[start_idx:end_idx]
        prtcls_old = particles_old[start_idx:end_idx]
        lbls = labels[start_idx:end_idx]
        chunk_global_obs = global_observables[start_idx:end_idx] if global_observables is not None else None
        print(f'i: {i}, use_SR: {use_SR}, prtcls shape: {prtcls.shape}, lbls shape: {len(lbls)}')
        n_events_chunk = prtcls.shape[0]
        graph_list = []

        # Option: choose whether to use the lv feature function.
        use_lv_feats = True   # Set to False to use simple difference instead.
        #num_outputs = 3       # For pairwise_lv_fts, typically 3 outputs (lnkt, lnz, lndelta)

        if graph_structure=='fully_connected':
            # Precompute edge_index once
            edge_pairs = [[i, j] for i in range(n_part) for j in range(n_part) if i != j]
            edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous().to(device)

            for event_idx in range(n_events_chunk):
                event_graphs  = []
                event_label = lbls[event_idx] 
                for jet_idx in range(2):  # Two jets per event
                    # Get particle features for the current jet
                    jet_particles = prtcls[event_idx, jet_idx][:, :3]  # Shape: (num_particles, (pt, eta, phi))
                    jet_particles_old = prtcls_old[event_idx, jet_idx][:, :3]  # Shape: (num_particles, (pt, eta, phi))
                    # Node features
                    x = torch.tensor(jet_particles, dtype=torch.float).to(device)  # Shape: (n_part, features)
                    x_old = torch.tensor(jet_particles_old, dtype=torch.float).to(device)  # Shape: (n_part, features)
                    # Create PyG Data object
                    data = torch_geometric.data.Data(x=x, edge_index=edge_index, fc_edge_index=edge_index)
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
                    
                    # --- Add Global Observables (if requested) ---
                    if load_global_obs and chunk_global_obs is not None:
                        jet_global_obs = chunk_global_obs[event_idx, jet_idx] # Shape (5,)
                        data.global_features = torch.tensor(jet_global_obs, dtype=torch.float).to(device)

                    event_graphs.append(data)
                    
                graph_list.append(event_graphs)

        elif graph_structure in ['laman', 'unique']:
            # Convert the particles array to a torch tensor on the target device. particles shape: (6000, 2, 10, 4)
            particles_t = torch.tensor(prtcls, dtype=torch.float, device=device)
            # Use only the first 3 features for each particle.
            particles_t = particles_t[..., :3]  # shape becomes (6000, 2, 10, 3)
            # Rearrange dimensions so that the features come before the particles: We want shape (n_events, 2, num_features, n_part), i.e. (6000, 2, 3, 10)
            particles_t = particles_t.permute(0, 1, 3, 2)

            # The input has shape (6000, 2, 3, 10) and inside laman_knn it will be reshaped to (12000, 3, 10).
            if graph_structure == 'laman':
                bool_mask = laman_knn(particles_t, angles = angles)
            elif graph_structure == 'unique':
                bool_mask = unique_graph(particles_t, angles = angles, num_edges=num_edges)
            # We expect bool_mask to have shape (12000, 10, 10) (one 10x10 mask for each jet).
            # Reshape it back so that it is grouped by event:
            bool_mask = bool_mask.reshape(particles_t.size(0), particles_t.size(1), particles_t.size(-1), particles_t.size(-1))
            # Now bool_mask has shape: (6000, 2, 10, 10)

            graph_list = []
            n_events_chunk = particles_t.size(0)  # 6000 events
            n_part = particles_t.size(-1)   # 10 particles per jet


            edge_pairs_fc = [[i, j] for i in range(n_part) for j in range(n_part) if i != j]
            fc_edge_index = torch.tensor(edge_pairs_fc, dtype=torch.long).t().contiguous().to(device)

            for event_idx in range(n_events_chunk):
                event_graphs = []
                event_label = lbls[event_idx]
                for jet_idx in range(2):  # Two jets per event
                    # particles_t[event_idx, jet_idx] has shape (3, 10): (features, particles).
                    # We need node features as (n_part, num_features) i.e. (10, 3).
                    x = particles_t[event_idx, jet_idx].transpose(0, 1)  # shape: (10, 3)
                    jet_particles_old = prtcls_old[event_idx, jet_idx][:, :3]  # Shape: (num_particles, (pt, eta, phi))
                    
                    x_old = torch.tensor(jet_particles_old, dtype=torch.float).to(device)  # Shape: (n_part, features)
                    # Retrieve the boolean adjacency matrix for this jet.
                    bm = bool_mask[event_idx, jet_idx]  # shape: (10, 10)
                    # Convert the boolean mask to edge_index.
                    edge_index = torch.nonzero(bm, as_tuple=False).t().contiguous()

                    # Create the PyG Data object.
                    data = torch_geometric.data.Data(x=x, edge_index=edge_index, fc_edge_index=fc_edge_index)
                    data.y = torch.tensor([event_label], dtype=torch.long, device=device)
                    # Compute edge attributes for the laman graph 
                    xi = x_old[edge_index[0]]  # source node features (shape: [E, 3])
                    xj = x_old[edge_index[1]]  # target node features (shape: [E, 3])

                    data.edge_attr = pairwise_lv_fts(xi, xj, num_outputs=num_outputs)

                    # Compute edge attributes for the fc graph 
                    xi = x_old[fc_edge_index[0]]
                    xj = x_old[fc_edge_index[1]]
                    data.fc_edge_attr = pairwise_lv_fts(xi, xj, num_outputs=num_outputs)
                    
                    # --- Add Global Observables (if requested) ---
                    if load_global_obs and chunk_global_obs is not None:
                        jet_global_obs = chunk_global_obs[event_idx, jet_idx] # Shape (5,)
                        data.global_features = torch.tensor(jet_global_obs, dtype=torch.float).to(device)

                    event_graphs.append(data)
                graph_list.append(event_graphs)

        final_graph_list.extend(graph_list)
    print(f'Constructed {len(final_graph_list)} PyG graphs for {total_size} events.')
    # Save to file using pytorch
    graph_filename = os.path.join(output_dir, f"graphs_pyg_{graph_key}.pt")
    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(graph_filename), exist_ok=True)

    torch.save(final_graph_list, graph_filename)
    print(f'Saved PyG graphs to {graph_filename}.')


