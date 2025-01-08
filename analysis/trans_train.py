''' Particle Transformer

Paper: "Particle Transformer for Jet Tagging" - https://arxiv.org/abs/1902.08570

Here we use 3 different versions of ParticleNet, the original one (7dim features for each particle and nearest neighbors algorithm at each layer), 
a 3dim modified one (3dim features for each particle and nearest neighbors algorithm at each layer, to make comparisons with Laman Graphs easier) 
and  a Laman one (3dim features for each particle and Laman Graphs for the first layer only, after that we use nearest neighbors).
'''

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

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_scatter import scatter_mean
from sklearn.metrics import roc_curve, auc

# Enable high precision for float32 matmul
torch.set_float32_matmul_precision('high')


import utils

import networkx
import energyflow
from analysis.models import transformer_gae
  
import random
import h5py 



def laman_graph(x): # For now this is not used 
    batch_size, _, seq_len = x.shape
    indices = np.zeros((batch_size, seq_len, seq_len))
    for i in range(seq_len-2):
        indices[:, i, i+1] = 1
        indices[:, i, i+2] = 1
    indices[seq_len-2, seq_len-1] = 1 
    
    return indices

def nearest_neighbors(x):
    x = torch.from_numpy(x) 

    batch_size, _, num_particles = x.size()
    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)

    rapidity = 0.5 * torch.log(1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    phi = torch.atan2(py, px)
    
    x = torch.cat((rapidity, phi), dim=1) # (batch_size, 2, num_points)

    inner = -2 * torch.matmul(x.transpose(2, 1), x)                                    # x.transpose(2, 1): flips the last two dimensions
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)                               # (batch_size, num_points, num_points)

    # Make the upper right triangle of the distance matrix infinite so that we don't connect the i-th particle with the j-th particle if i > j 
    # This also avoids double-counting the same distance (ij and ji)
    pairwise_distance = torch.tril(pairwise_distance) - torch.triu(torch.ones_like(pairwise_distance)*float('inf'))  # -inf because topk indices return the biggest values -> we've made all distances negative 

    # Initialize a boolean mask with False (indicating no connection) for all pairs
    bool_mask = torch.zeros((batch_size, num_particles, num_particles), dtype=torch.bool)

    # The non-padded particles for each jet
    non_zero_particles = np.linalg.norm(x, axis=1) != 0
    valid_n = non_zero_particles.sum(axis = 1)

    # Find the indices of the 2N-3 nearest neighbors for each jet and connect them: 
    for b in range(batch_size):
            if valid_n[b] <= 1:
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
            
            angles = 2*valid_n[b] - 3 # The maximum number of angles we can add until it becomes a fully connected graph
            
            bool_mask[b, row_indices[:angles], col_indices[:angles]] = True

    # Make the Laman Edges bidirectional 
    bool_mask = bool_mask | bool_mask.transpose(1, 2)

    # transform to numpy. That's because we later transform everything on the dataset to pytorch 
    # TODO: Change this code to work with numpy from the start
    bool_mask = bool_mask.numpy() 

    return bool_mask 

def random_laman_graph(x): 
    batch_size, _, num_particles = x.shape
    # for each b in batch size, calculate the number of non-zero particles and permute them 
    non_zero_particles = np.linalg.norm(x, axis=1) != 0
    valid_n = non_zero_particles.sum(axis = 1)
    idx = np.zeros((batch_size, num_particles, 2))
    # keep track of all perms so to remove the upper-triagonal part that stems from the first 3 particles forming a triangle
    perms = []
    for b in range(batch_size):
        if valid_n[b] <= 3:
            continue
        permutation = np.random.permutation(valid_n[b])
        perms.append(permutation)

        # connect the 3 first particles that are permuted
        idx[b, permutation[0] ] = [permutation[1], permutation[2]]
        idx[b, permutation[1] ] = [permutation[0], permutation[2]]
        idx[b, permutation[2] ] = [permutation[0], permutation[1]]
        # connect the rest of the particles in a Henneberg construction: Connect the i-th hardest particle with the 2 closest particles, i_1 and i_2, where i_1,2 < j
        for i in range(3, valid_n[b]):
            # for each particle i, add an edge to 2 particles j < i, at random 
            idx[b, permutation[i] ] = random.sample(list(permutation[:i]), 2)
        # fill the rest of the indices with valid_n[b] - 1, valid_n[b] - 2
        idx[b, valid_n[b]:] = [valid_n[b] - 1, valid_n[b] - 2]
    
    # Initialize a boolean mask with False (indicating no connection) for all pairs
    bool_mask = torch.zeros((batch_size, num_particles, num_particles), dtype=torch.bool)

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
    #mask_upper = ~torch.triu(torch.ones(num_particles, num_particles, dtype=torch.bool), diagonal=0)
    #bool_mask = bool_mask & mask_upper.unsqueeze(0)

    # Remove some angles at random between the particles. Default value of angles = 0.
    #bool_mask = angles_laman(x, bool_mask, angles, pairwise_distance = pairwise_distance) 

    # Make the Laman Edges bidirectional 
    bool_mask = bool_mask | bool_mask.transpose(1, 2)

    # transform to numpy. That's because we later transform everything on the dataset to pytorch 
    # TODO: Change this code to work with numpy from the start
    bool_mask = bool_mask.numpy() 

    return bool_mask 
        

def angles_laman(x, mask, angles = 0, pairwise_distance = None):
    batch_size, _, num_particles = mask.size()
    if isinstance(x, np.ndarray):
        non_zero_particles = np.linalg.norm(x, axis=1) != 0
    elif isinstance(x, torch.Tensor):
        non_zero_particles = torch.norm(x, p=2, dim=1) != 0
    
    valid_n = non_zero_particles.sum(axis = 1)
    valid_n_f = valid_n.float()

    # calculate the number of edges in the graph and the average number of edges/2n-3
    # Summing True values for each tensor in the batch
    #sum_true_per_batch = torch.sum(mask, dim=[1, 2])  # Sum over the last two dimensions

    # Calculating the average number of True values across the batch
    #average_true = torch.mean(sum_true_per_batch.float())  # Convert to float for mean calculation
    
    #print()
    #print(f"Average number of edges = {average_true.item()}")
    #print(f'edges/2n-3 = {average_true.item()/(2*torch.mean(valid_n_f)-3)}')
    #print()

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
                                                                # This creates a problem when we have a small number of particles. 
                                                                # My guess is that for add_angles <~ 15 it doesn't matter on average. 
                                                                # TODO: Check this with a more systematic study.
                                                                # An improved way, but slower computationally, to ensure full connectivity is:
                                                                # after transposing the adj, we remove one connection at random for particle "index" 
                                                                # if we ensure that there are other edges connecting this particle to the graph 
                                                                # NOTE: This could lead to two disconnected graphs potentially. 
                                                                # since n_edges >= n-1 for a connected graph, add_angles can at most be of order n anyways
                                                                # and the first way ensures that we have a single connected graph
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
            
            mask[b, row_indices[:min(angles, max_angles)], col_indices[:min(angles, max_angles)]] = True
        
    # Summing True values for each tensor in the batch
    sum_true_per_batch = torch.sum(mask, dim=[1, 2])  # Sum over the last two dimensions

    # Calculating the average number of True values across the batch
    average_true = torch.mean(sum_true_per_batch.float())  # Convert to float for mean calculation

    # Print info on the graphs. Very intensive computationally, so we keep it as an option 
    if False: 
        tot_2n3 = 0
        tot_n = 0 
        total_n1=0
        tot_n2 = 0
        total_3n = 0
        for b in range(batch_size):
            edges = torch.sum(mask[b], dim=[0, 1])
            tot_2n3 += edges / (2*valid_n[b]-3)
            tot_n += edges / valid_n[b]
            tot_n2 += edges / (1/2*valid_n[b]*(valid_n[b]-1))
            total_n1 += edges / (valid_n[b]-1)
            total_3n += edges / (3*valid_n[b]-6)

        av_2n3 = tot_2n3 / batch_size
        av_n = tot_n / batch_size
        av_n1 = total_n1 / batch_size
        av_n2 = tot_n2 / batch_size
        av_3n = total_3n / batch_size

        #    if 2 * sum_true_per_batch[b] > valid_n[b]*(valid_n[b]-1) :
        #        print('found a graph with more than 100% connectivity')
        #        print(f'valid_n[b] = {valid_n[b]}')
        #        print(f'edges = {sum_true_per_batch[b] }')
        #        print(f'sum_true_per_batch[b]/valid_n[b]^2 * 2 = {2 * sum_true_per_batch[b] / ( valid_n[b]*(valid_n[b]-1) ) }')

        print(f'After addition')
        print(f"Average number of edges = {average_true.item()}")
        #print(f'edges/2n-3 = {average_true.item()/(2*torch.mean(valid_n_f)-3)}')
        print(f'actual edges/2n-3 = {av_2n3}')
        print()
        print(f'actual edges/(3n-6) = {av_3n:.4f}')
        print()
        #print(f'edges/n = {average_true.item()/(torch.mean(valid_n_f))}')
        print(f'actual edges/n = {av_n}')
        print()
        print(f'actual edges/n-1= {av_n1}')
        print()
        #print(f'edges/(n^2/2) = {average_true.item()/(1/2*torch.mean(valid_n_f)**2)}')
        print(f'actual edges/(n^2/2) = {av_n2}')

        print()
               
    return mask 


def shannon_entropy(adjacency_matrices, x):
    x = x.cpu().numpy()
    batch_size, _, num_particles = x.shape
    if isinstance(x, np.ndarray):
        non_zero_particles = np.linalg.norm(x, axis=1) != 0
    elif isinstance(x, torch.Tensor):
        non_zero_particles = torch.norm(x, p=2, dim=1) != 0
    valid_n = non_zero_particles.sum(axis = 1) 

    valid_mask = np.arange(num_particles) < valid_n[:, None]
    
    # Use the mask to select elements from adjacency_matrices and sum to count neighbors
    n_neighbors = np.where(valid_mask[:, :, None], adjacency_matrices, 0).sum(axis=2)
    
    # Adjust n_neighbors based on valid_n, setting counts to 1 beyond valid particles
    # This step is no longer necessary as np.where and broadcasting handle the adjustment implicitly

    # Compute Shannon entropy
    # Avoid division by zero or log of zero by replacing non-valid n_neighbors with 1
    n_neighbors[n_neighbors == 0] = 1
    epsilon = 1e-8
    shannon_entropy_batch = np.log(n_neighbors).sum(axis=1) / ( valid_n + epsilon ) / np.log(valid_n - 1 + epsilon)
    shannon_entropy_batch = np.nan_to_num(shannon_entropy_batch)  # Handle divisions resulting in NaN
    shannon_entropy = np.mean(shannon_entropy_batch)
    print(f"Shannon Entropy = {shannon_entropy}")
    print()

    return shannon_entropy

def connected_components(adjacency_matrices, x):
    batch_size, _, num_particles = x.shape
    if isinstance(x, np.ndarray):
        non_zero_particles = np.linalg.norm(x, axis=1) != 0
    elif isinstance(x, torch.Tensor):
        non_zero_particles = torch.norm(x, p=2, dim=1) != 0

    valid_n = non_zero_particles.sum(axis = 1) 

    # use scipy.sparse.csgraph.connected_components to calculate the connected components for each graph in the batch
    connected_components = np.zeros((batch_size, num_particles))
    avg_n_components = 0
    for b in range(batch_size):
        adjacency_matrix = adjacency_matrices[b, :valid_n[b], :valid_n[b]]
        n_components, labels = scipy.sparse.csgraph.connected_components(adjacency_matrix, directed=False)
        avg_n_components += n_components
    # Average number of connected components
    avg_n_components = avg_n_components / batch_size
    
    print()
    print(f"Average number of connected components = {avg_n_components}")
    print()
    return 


# Create a knn Graph  
def knn(x, k, angles = 0, extra_info = False): 
    print()  
    print(f"Constructing a pure knn graph with k = {k}")
    print()
    x = torch.from_numpy(x) 

    batch_size, _, num_particles = x.size()

    non_zero_particles = np.linalg.norm(x, axis=1) != 0
    valid_n = non_zero_particles.sum(axis = 1)

    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)

    rapidity = 0.5 * torch.log(1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    phi = torch.atan2(py, px)
    
    x = torch.cat((rapidity, phi), dim=1) # (batch_size, 2, num_points)

    inner = -2 * torch.matmul(x.transpose(2, 1), x)                                    # x.transpose(2, 1): flips the last two dimensions
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)                               # (batch_size, num_points, num_points)

    # Mask the diagonal 
    # Create a mask for the diagonal elements across all matrices in the batch
    eye_mask = torch.eye(num_particles, dtype=torch.bool).expand(batch_size, num_particles, num_particles)
    pairwise_distance[eye_mask] = -float('inf')

    # Mask all the padded particles, i.e. n >= valid_n[b]
    # Create an indices tensor
    indices = torch.arange(pairwise_distance.size(1), device=pairwise_distance.device).expand_as(pairwise_distance)

    valid_n_tensor = torch.tensor(valid_n, device=pairwise_distance.device).unsqueeze(1).unsqueeze(2)

    # Now you can use valid_n_tensor in your operation
    mask_row = indices >= valid_n_tensor
    mask_col = indices.transpose(-2, -1) >= valid_n_tensor

    # Apply the masks
    pairwise_distance[mask_row] = -float('inf')
    pairwise_distance[mask_col] = -float('inf')

    # Find the indices of the 2 nearest neighbors for each particle        
    idx = pairwise_distance.topk(k=k, dim=-1) # It returns two things: values, indices 
    idx = idx[1] # (batch_size, num_points, 2)
    
    # Initialize a boolean mask with False (indicating no connection) for all pairs
    bool_mask = torch.zeros((batch_size, num_particles, num_particles), dtype=torch.bool)

    # Efficiently populate the boolean mask based on laman_indices
    for i in range(k):  # Assuming each particle is connected to two others as per laman_indices
        # Extract the current set of indices indicating connections
        current_indices = idx[:, :, i]

        # Generate a batch and source particle indices to accompany current_indices for scatter_
        batch_indices = torch.arange(batch_size).view(-1, 1).expand(-1, num_particles)
        src_particle_indices = torch.arange(num_particles).expand(batch_size, -1)

        # Use scatter_ to update the bool_mask; setting the connection locations to True
        bool_mask[batch_indices, src_particle_indices, current_indices] = True

    # Make the Edges bidirectional 
    bool_mask = bool_mask | bool_mask.transpose(1, 2)

    av = 0
    for b in range(batch_size):
        edges = torch.sum(bool_mask[b, :valid_n[b], :valid_n[b]], dim=(0,1)).item() / 2
        av += edges / (2*valid_n[b]-3) 
    av = av / batch_size
    print(f"Average number of edges/2n-3 = {av}")
    print()

    # transform to numpy. That's because we later transform everything on the dataset to pytorch 
    # TODO: Change this code to work with numpy from the start
    bool_mask = bool_mask.numpy() 

    # Calculate the Shannon Entropy and the number of connected components
    if extra_info:
        connected_components(bool_mask, x)
        #shannon_entropy(bool_mask, x)

    return bool_mask 


# Create a Laman Graph connecting 1N and 2N
def unique_1N2N3N(x, extra_info = False):

    if isinstance(x, np.ndarray):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = torch.from_numpy(x).to(device)
    else: 
        device  = x.device
    
    #device = "cpu"
    #x = x.to(device)

    non_zero_particles = torch.norm(x, p=2, dim=1) != 0
    valid_n = non_zero_particles.sum(axis = 1)
    
    t_start = time.time()
    batch_size, _, num_particles = x.size()

    
    bool_mask = torch.zeros((batch_size, num_particles, num_particles), dtype=torch.bool).to(device)
    #print(f'time to initialize bool mask = {time.time() - t_start:.3f}')

    # Create indices for the particles, excluding the first three
    indices = torch.arange(3, num_particles)

    bool_mask[:, 0, indices] = True
    bool_mask[:, 1, indices] = True
    bool_mask[:, 2, indices] = True
    bool_mask[:, 0, 1] = True
    bool_mask[:, 0, 2] = True
    bool_mask[:, 1, 2] = True
    

    #print(f'time to construct unique 1N2N3N graph = {time.time() - t_start:.3f}')
    #print()

    # Remove the padded particles from the graph to save memory space when converting to sparse representation.
    range_tensor = torch.arange(num_particles, device = device).unsqueeze(0).unsqueeze(-1)  
    expanded_valid_n = valid_n.unsqueeze(-1).unsqueeze(-1)
    mask = (range_tensor >= expanded_valid_n).to(device)
    final_mask = mask | mask.transpose(1, 2)

    bool_mask = bool_mask & ~final_mask


    #bool_mask = angles_laman(x, bool_mask, angles=0, pairwise_distance = None)
    if extra_info:
        # Calculate the Shannon Entropy and the number of connected components
        connected_components(bool_mask, x)
        shannon_entropy(bool_mask, x)
    
    bool_mask = bool_mask | bool_mask.transpose(1, 2)

    # transform to numpy. That's because we later transform everything on the dataset to pytorch 
    # TODO: Change this code to work with numpy from the start
    #bool_mask = bool_mask.numpy() 

    bool_mask = bool_mask.cpu()
    
    return bool_mask

# Create a Laman Graph connecting 1N and 2N
def laman_1N2N(x, extra_info = False):

    if isinstance(x, np.ndarray):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = torch.from_numpy(x).to(device)
    else: 
        device  = x.device
    
    non_zero_particles = torch.norm(x, p=2, dim=1) != 0
    valid_n = non_zero_particles.sum(axis = 1)
    
    t_start = time.time()
    batch_size, _, num_particles = x.size()

    
    bool_mask = torch.zeros((batch_size, num_particles, num_particles), dtype=torch.bool).to(device)
    #print(f'time to initialize bool mask = {time.time() - t_start:.3f}')
    # Create indices for the particles, excluding the first two (for i0 and i1)
    indices = torch.arange(2, num_particles)

    bool_mask[:, 0, indices] = True
    bool_mask[:, 1, indices] = True
    bool_mask[:, 0, 1] = True
    

    #print(f'time to construct laman 1N2N graph = {time.time() - t_start:.3f}')
    #print()

    # Remove the padded particles from the graph to save memory space when converting to sparse representation.
    range_tensor = torch.arange(num_particles, device = device).unsqueeze(0).unsqueeze(-1)  
    expanded_valid_n = valid_n.unsqueeze(-1).unsqueeze(-1)
    mask = (range_tensor >= expanded_valid_n).to(device)
    final_mask = mask | mask.transpose(1, 2)

    bool_mask = bool_mask & ~final_mask


    #bool_mask = angles_laman(x, bool_mask, angles=0, pairwise_distance = None)
    if extra_info:
        # Calculate the Shannon Entropy and the number of connected components
        connected_components(bool_mask, x)
        shannon_entropy(bool_mask, x)
    
    bool_mask = bool_mask | bool_mask.transpose(1, 2)

    # transform to numpy. That's because we later transform everything on the dataset to pytorch 
    # TODO: Change this code to work with numpy from the start
    #bool_mask = bool_mask.numpy() 

    bool_mask = bool_mask.cpu()
    
    return bool_mask


# Create a Laman Graph using a mod of the k nearest neighbors algorithm.
def laman_knn(x, angles = 0, extra_info = False):   
    # check if x is a numpy array, if not convert it to a numpy array
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(x, np.ndarray):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = torch.from_numpy(x).to(device)
        #x = torch.from_numpy(x)
    else: 
        device  = x.device

    non_zero_particles = torch.norm(x, p=2, dim=1) != 0
    valid_n = non_zero_particles.sum(axis = 1)

    
    t_start = time.time()
    batch_size, _, num_particles = x.size()
    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)

    rapidity = 0.5 * torch.log(1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    phi = torch.atan2(py, px)
    
    x = torch.cat((rapidity, phi), dim=1) # (batch_size, 2, num_points)
    #print()
    #print(f'time to calculate padded particles, rapidity and phi = {time.time() - t_start:.3f}')
    t_pairwise = time.time()
    inner = -2 * torch.matmul(x.transpose(2, 1), x)                                    # x.transpose(2, 1): flips the last two dimensions
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)                               # (batch_size, num_points, num_points)
    #print(f'time to calculate pairwise distance = {time.time() - t_pairwise:.3f}')
    #print()
    #print(f'pairwise_distance.device = {pairwise_distance.device}')
    #print(f'pairwise_distance.shape = {pairwise_distance.shape}')
    #print()

    # Connect the 3 hardest particles in the jet in a triangle 
    time_triangle = time.time()
    idx_3 = pairwise_distance[:, :3, :3].topk(k=3, dim=-1) # (batch_size, 3, 2)
    idx_3 = [idx_3[0][:,:,1:], idx_3[1][:,:,1:]] # (batch_size, 3, 1)
    
    # Connect the rest of the particles in a Henneberg construction: Connect the i-th hardest particle with the 2 closest particles, i_1 and i_2, where i_1,2 < j  
    pairwise_distance = pairwise_distance[:, 3:, :] # Remove the pairwise distances of 3 hardest particles from the distance matrix 
    
    # Make the upper right triangle of the distance matrix infinite so that we don't connect the i-th particle with the j-th particle if i > j 
    pairwise_distance = torch.tril(pairwise_distance, diagonal=2) - torch.triu(torch.ones_like(pairwise_distance)*float('inf'), diagonal=3)  # -inf because topk indices return the biggest values -> we've made all distances negative 
    #print(f'time to create triangle = {time.time() - time_triangle:.3f}')

    # Find the indices of the 2 nearest neighbors for each particle
    time_knn = time.time()
    idx = pairwise_distance.topk(k=2, dim=-1) # It returns two things: values, indices 
    #print(f'time to calculate knn = {time.time() - time_knn:.3f}')
    t_idx = time.time()
    idx = idx[1] # (batch_size, num_points - 3, 2)
        
    # Concatenate idx and idx_3 to get the indices of the 3 hardest particles and the 2 nearest neighbors for the rest of the particles
    idx = torch.cat((idx_3[1], idx), dim=1) # (batch_size, num_points, 3)
    
    # add 3 rows of -inf to the top of the pairwise_distance tensor to make it of shape (batch_size, num_particles, num_particles)
    # this is because we remove the 3 hardest particles from the graph and we don't want to connect them to the rest of the particles
    #print(f'pairwise_distance.device = {pairwise_distance.device}')
    pairwise_distance = torch.cat((torch.ones((batch_size, 3, num_particles), device = device)*float('-inf'), pairwise_distance), dim=1)
    #print(f'time to create idx = {time.time() - t_idx:.3f}')

    # Initialize a boolean mask with False (indicating no connection) for all pairs
    time_fillbool = time.time()
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
    #print(f'time to fill bool mask = {time.time() - time_fillbool:.3f}')

    t_memory = time.time()
    # Remove the padded particles from the graph to save memory space when converting to sparse representation.
    range_tensor = torch.arange(num_particles, device = device).unsqueeze(0).unsqueeze(-1)  
    expanded_valid_n = valid_n.unsqueeze(-1).unsqueeze(-1)
    mask = (range_tensor >= expanded_valid_n).to(device)
    final_mask = mask | mask.transpose(1, 2)

    bool_mask = bool_mask & ~final_mask
    #print(f'time to remove padded particles = {time.time() - t_memory:.3f}')
    t_end = time.time()
    # Remove some angles at random between the particles. Default value of angles = 0.
    bool_mask = angles_laman(x, bool_mask, angles, pairwise_distance = pairwise_distance) 

    # Make the Laman Edges bidirectional 
    bool_mask = bool_mask | bool_mask.transpose(1, 2)

    # transform to numpy. That's because we later transform everything on the dataset to pytorch 
    # TODO: Change this code to work with numpy from the start
    #bool_mask = bool_mask.numpy() 

    bool_mask = bool_mask.cpu()
    if extra_info:
        # Calculate the Shannon Entropy and the number of connected components
        connected_components(bool_mask, x)
        shannon_entropy(bool_mask, x)
    #print(f'time to construct laman knn graph = {time.time() - t_start:.3f}')
    #print()
    return bool_mask 



def edge_indices_to_boolean_adjacency(adjacency_matrices, n):
    batch_size, _, max_edges = adjacency_matrices.shape
    # Initialize boolean adjacency matrix with False
    boolean_adjacency = np.zeros((batch_size, n, n), dtype=bool)
    
    for b in range(batch_size):
        for e in range(max_edges):
            i, j = adjacency_matrices[b, :, e]
            # Check for padding and update adjacency matrix
            if i != -1 and j != -1:
                boolean_adjacency[b, i, j] = True
    
    return boolean_adjacency

def rand_graph(x):
    batch_size, _, n = x.shape
    max_edges = 2 * n - 3 
    adjacency_matrices = np.full((batch_size, 2, max_edges), -1, dtype=np.int64)
    ns = []
    for b in range(batch_size):
        # Identify non-zero particles
        non_zero_particles = np.linalg.norm(x[b], axis=0) != 0

        valid_n = non_zero_particles.sum().item()
        ns.append(valid_n)
        edges = []

        num_edges = 2 * valid_n - 3
        if num_edges >= 1:    
            added_edges = set()
                
            while len(added_edges) < num_edges:
                # Select two different valid nodes
                i, j = random.sample(range(valid_n), 2)
                edge = (min(i, j), max(i, j))

                if edge not in added_edges:
                    added_edges.add(edge)
                    # Add edge indices for non-zero particles only
                    edges.append(edge) 
        else: # edge case with n=1
            edges = [(0, 0)] 
        
        # Convert edges to a sparse tensor
        # Edges and their transposes (since the graph is undirected)
        edge_indices = np.array(edges, dtype=np.int64).T
  
    adjacency_matrices = edge_indices_to_boolean_adjacency(adjacency_matrices, n)
    
    adjacency_matrices = adjacency_matrices | adjacency_matrices.transpose(1, 2)

    return adjacency_matrices



def unique_graph(x, angles = 0, extra_info = False):
    # check if x is a numpy array, if not convert it to a numpy array
    if isinstance(x, np.ndarray):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = torch.from_numpy(x).to(device)
    else: 
        device  = x.device

    non_zero_particles = torch.norm(x, p=2, dim=1) != 0
    valid_n = non_zero_particles.sum(axis = 1)

    batch_size, _, num_particles = x.size()
    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)

    rapidity = 0.5 * torch.log(1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    phi = torch.atan2(py, px)
    
    x = torch.cat((rapidity, phi), dim=1) # (batch_size, 2, num_points)

    inner = -2 * torch.matmul(x.transpose(2, 1), x)                                    # x.transpose(2, 1): flips the last two dimensions
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)                               # (batch_size, num_points, num_points)

    # Connect the 4 hardest particles in the jet in a fully connected graph
    idx_3 = pairwise_distance[:, :4, :4].topk(k=4, dim=-1) # (batch_size, 3, 2)
    idx_3 = [idx_3[0][:,:,1:], idx_3[1][:,:,1:]] # (batch_size, 3, 1)

    # Connect the rest of the particles in a Henneberg construction: Connect the i-th hardest particle with the 2 closest particles, i_1 and i_2, where i_1,2 < j  
    pairwise_distance = pairwise_distance[:, 4:, :] # Remove the pairwise distances of 3 hardest particles from the distance matrix 
    
    # Make the upper right triangle of the distance matrix infinite so that we don't connect the i-th particle with the j-th particle if i > j 
    pairwise_distance = torch.tril(pairwise_distance, diagonal=3) - torch.triu(torch.ones_like(pairwise_distance)*float('inf'), diagonal=4)  # -inf because topk indices return the biggest values -> we've made all distances negative 

    # Find the indices of the 3 nearest neighbors for each particle
        
    idx = pairwise_distance.topk(k=3, dim=-1) # It returns two things: values, indices 
    idx = idx[1] # (batch_size, num_points - 3, 2)

    # Concatenate idx and idx_3 to get the indices of the 3 hardest particles and the 3 nearest neighbors for the rest of the particles
    idx = torch.cat((idx_3[1], idx), dim=1) # (batch_size, num_points, 3)

    # add 3 rows of -inf to the top of the pairwise_distance tensor to make it of shape (batch_size, num_particles, num_particles)
    # this is because we remove the 3 hardest particles from the graph and we don't want to connect them to the rest of the particles
    #print(f'pairwise_distance.device = {pairwise_distance.device}')
    pairwise_distance = torch.cat((torch.ones((batch_size, 3, num_particles), device = device)*float('-inf'), pairwise_distance), dim=1)

    # Initialize a boolean mask with False (indicating no connection) for all pairs
    bool_mask = torch.zeros((batch_size, num_particles, num_particles), dtype=torch.bool).to(device)

    # Efficiently populate the boolean mask based on laman_indices
    for i in range(3):  # Assuming each particle is connected to two others as per laman_indices
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
    bool_mask = bool_mask | bool_mask.transpose(1, 2)

    # transform to numpy. That's because we later transform everything on the dataset to pytorch 
    # TODO: Change this code to work with numpy from the start
    #bool_mask = bool_mask.numpy() 
    bool_mask = bool_mask.cpu()

    if extra_info:
        # Calculate the number of connected components
        connected_components(bool_mask, x)

    return bool_mask 


def split_into_batches(tensor, batch_size):
    if tensor is None:
        return None
    num_batches = (tensor.size(0) + batch_size - 1) // batch_size
    return [tensor[i * batch_size:min((i + 1) * batch_size, tensor.size(0))] for i in range(num_batches)]


def broadcast_batches(batches, src):
    if batches is None:
        return
    for batch in batches:
        dist.broadcast(batch, src=src)


def to_pt2(x, eps=1e-8):
    pt2 = x[:, :2].square().sum(dim=1, keepdim=True)
    if eps is not None:
        pt2 = pt2.clamp(min=eps)
    return pt2

def atan2(y, x):
    sx = torch.sign(x)
    sy = torch.sign(y)
    pi_part = (sy + sx * (sy ** 2 - 1)) * (sx - 1) * (-math.pi / 2)
    atan_part = torch.arctan(y / (x + (1 - sx ** 2))) * sx ** 2
    return atan_part + pi_part

# Transform the 4-momentum vector to the (pt, rapidity, phi, mass) representation
def to_ptrapphim(x, eps=1e-8): 
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)
    pt = torch.sqrt(to_pt2(x, eps=eps))
    # rapidity = 0.5 * torch.log((energy + pz) / (energy - pz))
    rapidity = 0.5 * torch.log(1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    phi = torch.atan2(py, px)
    
    return torch.cat((pt, rapidity, phi), dim=1)

class ParT():
    
    #---------------------------------------------------------------
    def __init__(self, model_info, plot_path='/global/homes/d/dimathan/gae_for_anomaly/plots_trans/plots_test2'):
        '''
        :param model_info: Dictionary of model info, containing the following keys:
                                'model_settings': dictionary of model settings
                                'n_total': total number of training+val+test examples
                                'n_train': total number of training examples
                                'n_test': total number of test examples
                                'torch_device': torch device
                                'output_dir': output directory
                                'Laman': boolean variable to choose between the original ParticleNet and the Laman implementation 
                                'three_momentum_features': ONLY RELEVENAT IF Laman = false. Boolean variable to choose between the 7 dimensional representation and the 3 dimensional one 

                           In the case of subjet GNNs, the following keys are also required, originating from the graph_constructor:
                                'r': subjet radius
                                'n_subjets_total': total number of subjets per jet
                                'subjet_graphs_dict': dictionary of subjet graphs
        '''
        
        self.model_info = model_info
        self.set_ddp = model_info['ddp']
        print(f'set_ddp = {self.set_ddp}')
        self.plot_path = plot_path
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        self.ext_plot = model_info['ext_plot']

        if self.set_ddp: 
            self.local_rank = int(os.getenv("LOCAL_RANK"))
            # initialize the process group and set a timeout of 70 minutes, so that the process does not terminate
            # while rank=0 calculates the graph and the other ranks wait for the graph to be calculated
            dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(minutes=200)) 
            torch.cuda.set_device(self.local_rank)
            self.torch_device = torch.device('cuda', self.local_rank)
            if self.local_rank == 0:
                print()
                print('Running on multiple GPUs...')
                print()
                print('setting up DDP...')
                print("MASTER_ADDR:", os.getenv("MASTER_ADDR"))
                print("MASTER_PORT:", os.getenv("MASTER_PORT"))
                print("WORLD_SIZE:", os.getenv("WORLD_SIZE"))
                print("RANK:", os.getenv("RANK"))
                print("LOCAL_RANK:", os.getenv("LOCAL_RANK"))
           
        else: 
            self.torch_device = model_info['torch_device']
            self.local_rank = 0

        self.output_dir = model_info['output_dir']
        
        self.n_part = model_info['n_part']

        self.n_total = model_info['n_total']
        self.n_train = model_info['n_train']
        self.n_test = model_info['n_test']
        self.n_val = model_info['n_val'] 
 
        self.n_bkg = model_info['model_settings']['n_bkg'] # For classification
        self.n_sig = model_info['model_settings']['n_sig'] # For classification


        self.batch_size = self.model_info['model_settings']['batch_size']
        self.patience = self.model_info['model_settings']['patience']
        #if self.set_ddp: # Adjust batch size for DDP
        #    self.batch_size = self.batch_size // torch.cuda.device_count()


        self.lossname = self.model_info['model_settings']['lossname']
        
        if self.lossname == 'MSE': self.criterion = torch.nn.MSELoss()
        else: sys.exit(f'Error: loss {self.lossname} not recognized.')

        self.pair_input_dim = model_info['model_settings']['pair_input_dim']  # how many interaction terms for pair of particles. 
                                                                              # If 3: use (dR, k_t = min(pt_1, pt_2)*dR, z = min(pt_1, pt_2)/(pt_1 + pt_2) ),
                                                                              # if 4: also use m^2 = (E1 + E2)^2 - (p1 + p2)^2    
        
        # Use custon training parameters
        self.epochs = self.model_info['model_settings']['epochs']
        self.learning_rate = self.model_info['model_settings']['learning_rate']        
        if self.set_ddp and self.local_rank == 0:
            self.learning_rate = self.learning_rate #* torch.cuda.device_count() # Adjust learning rate for DDP
            self.batch_size = self.batch_size // torch.cuda.device_count()
            print(f'for ddp training with {torch.cuda.device_count()} GPUs, the learning rate is adjusted accordingly.')

        self.plot_path = f'/global/homes/d/dimathan/gae_for_anomaly/plots_trans/plot_n{self.n_part}_e{self.epochs}_lr{self.learning_rate}_N{self.n_train//1000}k'
        if self.local_rank == 0:
            if not os.path.exists(self.plot_path):
                os.makedirs(self.plot_path)

        # Default values:            
        self.graph_transformer = False
        self.graph_type = None
        self.add_angles = 0
        self.sorting_key = 'pt'
        self.load_model = False
        if model_info['model_key'].endswith('graph'): # Graph-Transfomer, e.g. Laman Graph or a KNN Graph
            self.graph_transformer = True
            self.input_dim = 1
            self.graph_type = self.model_info['model_settings']['graph']
            self.add_angles = model_info['model_settings']['add_angles']
        else: 
            self.input_dim = model_info['model_settings']['input_dim'] # 4 for (px, py, pz, E) as input for each particle, 1 for pt
            if self.input_dim not in [1, 3, 4]:
                raise ValueError('Invalid input_dim at the config file for ParT. Must be 1, 3 or 4') 


        self.train_loader, self.val_loader, self.test_loader = self.init_tr_data()
        if self.local_rank == 0: # only need to run the classification in the end in one gpu since its one pass through the data
            self.cl_loader = self.init_cl_data()

        self.model = self.init_model()

    #---------------------------------------------------------------
    def init_cl_data(self):
        # since we are not training on the cl data, but we are only evaluating the model on it once, we don't need to use ddp 
        # Load the data for the main process (rank 0)
        X_ParT, Y_ParT = self.load_particles(use_SR=True)

        # Delete the last-column (pid or masses or E) of the particles
        X_ParT = X_ParT[:,:,:,:3]
        if self.local_rank == 0:
            non_zero_particles = np.linalg.norm(X_ParT, axis=3) != 0
            valid_n = non_zero_particles.sum(axis = 2)
            print('average number of particles per jet:', np.mean(valid_n))
        
    
        # Change the order of the features from (pt, eta, phi, pid) to (px, py, pz, E) to agree with the architecture.ParticleTransformer script
        #X_ParT = energyflow.p4s_from_ptyphims(X_ParT)

        # (E, px, py, pz) -> (px, py, pz, E)
        #X_ParT[:,:,:, [0, 1, 2, 3]] = X_ParT[:,:, :, [1, 2, 3, 0]] 
            
        # Transpose the data to match the ParticleNet/ParT architecture convention which is (batch_size, 2, n_features, n_particles) 
        # instead of the current shape (batch_size, 2, n_particles, n_features)
        X_ParT = np.transpose(X_ParT, (0, 1, 3, 2))
            
        if self.graph_transformer:
            X, Y, graph = self.load_data(X_ParT, Y_ParT, graph_transformer = self.graph_transformer, train=False) 
        else:
            X, Y = self.load_data(X_ParT, Y_ParT, graph_transformer = self.graph_transformer, train=False) 
            graph = None

        X = torch.from_numpy(X).to(self.torch_device).float()
        Y = torch.from_numpy(Y).to(self.torch_device).float()
        if self.graph_transformer:
            cl_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y, graph), batch_size=self.batch_size, shuffle=False)
        else:
            cl_dataset = torch.utils.data.TensorDataset(X, Y)  
            cl_loader = torch.utils.data.DataLoader(cl_dataset, batch_size=self.batch_size, shuffle=True)

        return cl_loader

    #---------------------------------------------------------------
    def init_tr_data(self):
        # Choose the dataset to load
        # The jetclass dataset origin: https://github.com/jet-universe/particle_transformer 
        # It has 10 different classes of jets, each class has 10M jets. 
        # Currently we are using the Z vs QCD dataset. For more details on the classes of jets look at the dataloader.py script and the 
        # github repository mentioned above.

        if self.local_rank == 0:
            # Load the data for the main process (rank 0)
            self.X_ParT, self.Y_ParT = self.load_particles(use_SR=False)

            # Delete the last-column (pid or masses or E) of the particles
            self.X_ParT = self.X_ParT[:,:,:,:3]

            print(f'X_ParT.shape: {self.X_ParT.shape}')
            print(f'self.X_ParT[0, :, :5, :]: {self.X_ParT[0, :, :5, :]}')
            print()
            
            non_zero_particles = np.linalg.norm(self.X_ParT, axis=3) != 0
            valid_n = non_zero_particles.sum(axis = 2)
            print('average number of particles per jet:', np.mean(valid_n))
            print()
            # TODO:
                
            # Change the order of the features from (pt, eta, phi, pid) to (px, py, pz, E) to agree with the architecture.ParticleTransformer script
            #self.X_ParT = energyflow.p4s_from_ptyphims(self.X_ParT)
            #print(f'self.X_ParT[0, :, :5, :]: {self.X_ParT[0, :, :5, :]}')
            #print()
            # (E, px, py, pz) -> (px, py, pz, E)
            #self.X_ParT[:,:,:, [0, 1, 2, 3]] = self.X_ParT[:,:, :, [1, 2, 3, 0]] 
            #print(f'self.X_ParT[0, :, :5, :]: {self.X_ParT[0, :, :5, :]}')
            #print()
            # Transpose the data to match the ParticleNet/ParT architecture convention which is (batch_size, 2, n_features, n_particles) 
            # instead of the current shape (batch_size, 2, n_particles, n_features)
            self.X_ParT = np.transpose(self.X_ParT, (0, 1, 3, 2))
            #print(f'X_ParT.shape: {self.X_ParT.shape}')
            #print(f'self.X_ParT[0, :, :5, :]: {self.X_ParT[0, :, :5, :]}')
            #print()
            if self.graph_transformer:
                features_train, features_val, features_test, Y_ParT_train, Y_ParT_val, Y_ParT_test, graph_train, graph_val, graph_test = self.load_data(self.X_ParT, self.Y_ParT, graph_transformer = self.graph_transformer, sorting_key = self.sorting_key)
            else:   
                features_train, features_val, features_test, Y_ParT_train, Y_ParT_val, Y_ParT_test = self.load_data(self.X_ParT, self.Y_ParT, graph_transformer = self.graph_transformer, sorting_key = self.sorting_key)
                graph_train, graph_val, graph_test = (None,) * 3
            #print(f'features_train.shape: {features_train.shape}')
            #print(f'features_train[0, :, :5, :]: {features_train[0, :, :5, :]}')
            #print()
        else: # Initialize the data loaders for all but the main process (rank 0)
            features_train, features_val, features_test, Y_ParT_train, Y_ParT_val, Y_ParT_test, graph_train, graph_val, graph_test = (None,) * 9
        
        objects = [features_train, features_val, features_test, Y_ParT_train, Y_ParT_val, Y_ParT_test, graph_train, graph_val, graph_test]
        
        if self.local_rank == 0:
            memory_usage = 0 
            for index, obj in enumerate(objects):
                # print dtype and memory usage of the object
                if obj is not None:
                    if index < 6:
                        memory_usage += obj.nbytes/1024**2
                    print(f"Object dtype: {obj.dtype}, memory usage: {obj.nbytes/1024**2} MB")
            print('Total memory usage of particle features + Y values: ', memory_usage, 'MB')


        if graph_train is not None and self.set_ddp:
            sparse_matrices = [csr_matrix(graph_train[i]) for i in range(graph_train.shape[0])]
            graph_train = sparse_matrices 
            sparse_matrices = [csr_matrix(graph_val[i]) for i in range(graph_val.shape[0])]
            graph_val = sparse_matrices
            sparse_matrices = [csr_matrix(graph_test[i]) for i in range(graph_test.shape[0])]
            graph_test = sparse_matrices
            
            total_memory_usage = sum(csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes for csr in graph_train)
            total_memory_usage += sum(csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes for csr in graph_val)
            total_memory_usage += sum(csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes for csr in graph_test) 

            # Convert bytes to megabytes for readability
            total_memory_usage_mb = total_memory_usage / (1024 ** 2)
            print()
            print(f"Total memory usage of sparse graphs: {total_memory_usage_mb:.3f} MB")


        objects_tobroadcast = [features_train, features_val, features_test, Y_ParT_train, Y_ParT_val, Y_ParT_test]
        objects_tobroadcast_graph = [graph_train, graph_val, graph_test]

        if self.set_ddp:
            # Broadcast the objects from rank 0 to all other processes
            # The adjacency matrix is very sparse and cannot be broadcasted using the default broadcast_object_list function
            dist.broadcast_object_list(objects_tobroadcast, src=0)
            dist.broadcast_object_list(objects_tobroadcast_graph, src=0)


        # Now, all GPUs have the same "objects_tobroadcast". Unpack it.
        if self.local_rank != 0:
            # Unpack the objects on other processes
            features_train, features_val, features_test, \
            Y_ParT_train, Y_ParT_val, Y_ParT_test, = objects_tobroadcast

            graph_train, graph_val, graph_test = objects_tobroadcast_graph

     
        # Transform the adjacency matrices back to dense format
        if  graph_train is not None and self.set_ddp:
            # back to dense format 
            dense_2d_arrays = [csr.toarray() for csr in graph_train]
            graph_train = np.array(dense_2d_arrays)
            dense_2d_arrays = [csr.toarray() for csr in graph_val]
            graph_val = np.array(dense_2d_arrays)
            dense_2d_arrays = [csr.toarray() for csr in graph_test]
            graph_test = np.array(dense_2d_arrays)


        if self.graph_transformer:
            train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(features_train).float(), torch.from_numpy(Y_ParT_train).long(), torch.from_numpy(graph_train).bool()  )
            self.train_sampler = DistributedSampler(train_dataset) if self.set_ddp else None
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = self.batch_size, sampler = self.train_sampler, num_workers = 4)

            val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(features_val).float(), torch.from_numpy(Y_ParT_val).long(), torch.from_numpy(graph_val).bool() )
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = self.batch_size, num_workers = 4)
            
            test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(features_test).float(), torch.from_numpy(Y_ParT_test).long(), torch.from_numpy(graph_test).bool() )
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = self.batch_size, num_workers = 4)

        else: 
            train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(features_train).float(), torch.from_numpy(Y_ParT_train).long())
            self.train_sampler = DistributedSampler(train_dataset) if self.set_ddp else None
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = self.batch_size, sampler = self.train_sampler, num_workers = 4)

            val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(features_val).float(), torch.from_numpy(Y_ParT_val).long())
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = self.batch_size, num_workers = 4, shuffle=True)
            
            test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(features_test).float(), torch.from_numpy(Y_ParT_test).long()) 
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = self.batch_size, num_workers = 4, shuffle=True)

        return train_loader, val_loader, test_loader
    
    #---------------------------------------------------------------
    def load_particles(self, use_SR=False):
        ''' 
        Load the particles from the jetclass/energyflow dataset. 
        '''
        
        device = self.torch_device
        # TODO: n_part, n_events
        if not use_SR:
            particles, jets, mjj = utils.DataLoader(n_events=self.n_total, rank=0, n_part=self.n_part)
            labels = [0]*len(particles)       
        elif use_SR:
            jets, particles, mjj, labels = utils.class_loader(use_SR=True, nbkg = self.n_bkg, nsig = self.n_sig, n_part=self.n_part)

        if self.local_rank == 0:
            print(f'particles shape: {particles.shape}')
            print(f'jets shape: {jets.shape}')
            print(f'mjj shape: {mjj.shape}')
            print(f'labels shape: {len(labels)}')

        particles, jets = utils._preprocessing(particles, jets, mjj, norm = 'mean')

        return particles, labels


    #---------------------------------------------------------------
    def load_data(self, X, Y, graph_transformer = False, sorting_key = None, train=True):
        ''' 
        Split the data into training, validation and test sets depending on the specifics of the model.
        '''
        if graph_transformer:
            print()

            sorted_indices = np.argsort( -(X[:, 0, :]**2 + X[:, 1, :]**2), axis=-1)[:, np.newaxis, :]

            X = np.take_along_axis(X, sorted_indices, axis=-1)

            t_st = time.time()
            
            # We need to constuct the graph in chunks to avoid memory issues when n_total > 10^6
            chunk_size = 20*256  # Adjust this based on your memory constraints and the size of self.X_ParT
            total_size = X.shape[0]  # Assuming the first dimension is the batch size
            chunks = (total_size - 1) // chunk_size + 1  # Calculate how many chunks are needed
            if self.local_rank == 0: # To avoid calculating the graph multiple times
                if self.graph_type == 'laman_random_graph': 
                    graph = np.concatenate([random_laman_graph(X[i * chunk_size:(i + 1) * chunk_size]) for i in range(chunks)] )

                elif self.graph_type == 'laman_1N2N':
                    graph = torch.cat([laman_1N2N(X[i * chunk_size:(i + 1) * chunk_size], extra_info=False if i==0 else False) for i in range(chunks)] )
                    
                elif self.graph_type == 'laman_knn_graph': 
                    print('Constructing a Laman Graph using a mod of the k nearest neighbors algorithm.')
                    graph = torch.cat([laman_knn(X[i * chunk_size:(i + 1) * chunk_size], angles = self.add_angles, extra_info=True if i==0 else False) for i in range(chunks)])

                elif self.graph_type == '2n3_nearest_neighbors': 
                    graph = np.concatenate([nearest_neighbors(X[i * chunk_size:(i + 1) * chunk_size]) for i in range(chunks)] ) 
                    
                elif self.graph_type == 'knn_graph':
                    k = self.k
                    graph = np.concatenate([knn(X[i * chunk_size:(i + 1) * chunk_size], k = k, extra_info=True if i==0 else False) for i in range(chunks)] )
                
                elif self.graph_type == 'unique_graph':
                    print('Constructing a Unique Graph using a mod of the k nearest neighbors algorithm.')
                    graph = torch.cat([unique_graph(X[i * chunk_size:(i + 1) * chunk_size], angles = self.add_angles, extra_info=True if i==0 else False) for i in range(chunks)] )
                
                elif self.graph_type == 'unique_1N2N3N':
                    graph = torch.cat([unique_1N2N3N(X[i * chunk_size:(i + 1) * chunk_size], extra_info=False if i==0 else False) for i in range(chunks)] )

                else: 
                    sys.exit("Invalid graph type for Laman Graphs. Choose between 'laman_random_graph', 'laman_knn_graph, '2n3_nearest_neighbors', 'knn_graph' and 'unique_graph'") 

                print(f"Time to create the graph = {time.time() - t_st} seconds")
             
            if not train: # classification 
                return X, Y, graph
            
            (features_train, features_val, features_test, Y_ParT_train, Y_ParT_val, Y_ParT_test, 
            graph_train, graph_val, graph_test) = energyflow.utils.data_split(X, Y, graph.numpy(),
                                                                              val=self.n_val, test=self.n_test, shuffle = True)

            return (features_train, features_val, features_test, Y_ParT_train, Y_ParT_val, Y_ParT_test, 
            graph_train, graph_val, graph_test)
       
        # For the case of Vanilla Particle Transformer
        else: 
            Y = np.array(Y) # cause we've defined Y as a python list
            if not train: # classification
                return X, Y
            
            (features_train, features_val, features_test, Y_ParT_train, Y_ParT_val, Y_ParT_test) = energyflow.utils.data_split(X, Y, val=self.n_val, test=self.n_test, shuffle = False)
            return (features_train, features_val, features_test, Y_ParT_train, Y_ParT_val, Y_ParT_test)
        
    #---------------------------------------------------------------
    def init_model(self):
        '''
        :return: pytorch architecture
        '''

        # Define the model 
        enc_cfg = dict(input_dim = self.input_dim, pair_input_dim = self.pair_input_dim) 
        dec_cfg = dict()
        model = transformer_gae.TAE(enc_cfg, dec_cfg) # 4 features: (px, py, pz, E)
        
        # ==================================================
        model = torch.compile(model)
        # ==================================================
        
        model = model.to(self.torch_device)
        
        # Print the model architecture if master process
        if self.local_rank == 0:
            print()
            print(model)
            print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')
            print()

        if self.load_model:
            self.path = f'/global/homes/d/dimathan/GNNs-and-Jets/Saved_Model_weights/{self.model_info["model_key"]}_p{self.input_dim}_{self.pair_input_dim}.pth'
            print(f"Loading pre-trained model from {self.path}")
            model.load_state_dict(torch.load(self.path, map_location=self.torch_device))
    
        return model 


    #---------------------------------------------------------------
    def train(self):
        if self.local_rank == 0:
            print(f'Training...')
            print()

        time_start = time.time()

        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)
        #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.33, patience=self.patience, verbose=True)
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,  # Initial learning rate (use a smaller value, e.g., 1e-4 or 5e-5)
            weight_decay=0.01  # Regularization to reduce overfitting
        )

        # Scheduler
        num_training_steps = len(self.train_loader) * self.epochs  # Total number of training steps
        num_warmup_steps = int(0.2 * num_training_steps)  # Warmup for 20% of training steps

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,  # Peak learning rate during warmup
            total_steps=num_training_steps,
            anneal_strategy='linear',  # Linearly decay the learning rate after warmup
            pct_start=num_warmup_steps / num_training_steps,  # Proportion of warmup steps
            div_factor=20.0,  # Initial LR is 1/10th of max_lr
            final_div_factor=10.0  # Final LR is 1/5th of max_lr
        )




        best_val_loss = math.inf        
        best_model_path = os.path.join(self.output_dir, 'tae_best.pt')  # Path for saving the best model
        
        self.auc_list = []
        self.train_loss_list = []
        self.val_loss_list = []
        
        if self.set_ddp:
            torch.cuda.set_device(self.local_rank)
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model) # SyncBatchNorm for DDP, else it will throw an error due to inplace operations in encoder-decoder
            self.model = DDP(self.model, device_ids=[self.local_rank],) #find_unused_parameters=True)
                
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(1, self.epochs+1):
            if self.set_ddp: self.train_sampler.set_epoch(epoch)  

            t_start = time.time()
            loss_train = self._train_part(epoch)
            
            if self.local_rank == 0:
                loss_test = self._test_part(self.test_loader,)
                loss_val = self._test_part(self.val_loader, )
                if epoch%4==0 or self.ext_plot:
                    auc = self.run_anomaly(plot=False, ep=epoch)
                    actual_train_loss = self._test_part(self.train_loader, )
                    print(f'actual_train_loss={actual_train_loss:.4f}')
                    self.auc_list.append(auc)
                    self.train_loss_list.append(actual_train_loss)
                    self.val_loss_list.append(loss_val)
            
            
                # Save the model with the best test AUC
                if loss_val < best_val_loss:
                    best_val_loss = loss_val
                    torch.save(self.model.state_dict(), best_model_path)

                print("--------------------------------")
                print(f'Epoch: {epoch:02d}, loss_train: {loss_train:.4f}, loss_val: {loss_val:.4f}, loss_test: {loss_test:.4f}, lr: {self.optimizer.param_groups[0]["lr"]:.5f}, Time: {time.time() - t_start:.1f} sec')
            
            #self.scheduler.step(loss_val)
                
            # Synchronize all processes 
            if self.set_ddp:
                dist.barrier()  # Ensures all processes finish training before proceeding

        # Load the best model before returning
        self.model.load_state_dict(torch.load(best_model_path))  # Load best model's state_dict
        if self.local_rank == 0:
            print("Loaded the best model based on validation loss.")
            loss_val = self._test_part(self.val_loader,)
            loss_test = self._test_part(self.test_loader,)
            print(f'Best model validation loss: {loss_val:.4f}')
            print(f'Best model test loss: {loss_test:.4f}')

            # lets plot the loss distribution of the test set
            print(f'--------------------------------')
            print(f'Finished training')
            print()
            self._plot_loss()

        # Synchronize all processes before loading the model
        #if self.set_ddp:
        #    dist.barrier()  # Ensures all processes finish training before proceeding


        return self.model


        
    #---------------------------------------------------------------
    def _train_part(self, ep=-1):


        self.model.train()        # Set model to training mode. This is necessary for dropout, batchnorm etc layers 
                                  # that behave differently in training mode vs eval mode (which is the default)
                                  # We need to include this since we have particlenet.eval() in the test_particlenet function
        
        loss_cum = 0              # Cumulative loss
        count = 0                 
        
        for index, data in enumerate(self.train_loader):
            inputs, labels = data[0], data[1]   
            inputs = inputs.to(self.torch_device)

            jet0, jet1 = inputs[:,0, :, :], inputs[:,1, :, :]
            length = jet0.shape[0]

            if self.graph_transformer: graph = data[2].to(self.torch_device)
            else:  graph = None
            
            # zero the parameter gradients
            self.optimizer.zero_grad()

            #p3_0 = to_ptrapphim(jet0)
            #p3_1 = to_ptrapphim(jet1)
            p3_0 = jet0 
            p3_1 = jet1
            
            if self.input_dim == 1:
                # create pt of each particle instead of (px, py, pz, E) for the input 
                x0 = p3_0[:, 0, :].unsqueeze(1)
                x1 = p3_1[:, 0, :].unsqueeze(1)

            elif self.input_dim == 3: 
                x0 = p3_0
                x1 = p3_1

            else:
                x0, x1 = jet0, jet1
            #print(f'x0.shape: {x0.shape}')
            #print(f'jet0.shape: {jet0.shape}')
            #print(f'p3_0.shape: {p3_0.shape}')
            # forward + backward + optimize

            out0 = self.model(x = x0, v = jet0, graph = graph,) # pr=True if index==0 else False)
            out1 = self.model(x = x1, v = jet1, graph = graph)
            p3_0 = p3_0.permute(0, 2, 1)
            p3_1 = p3_1.permute(0, 2, 1)
            jet0 = jet0.permute(0, 2, 1)
            jet1 = jet1.permute(0, 2, 1)
                            
            loss0 = self.criterion(out0, p3_0)  
            loss1 = self.criterion(out1, p3_1) 
            loss = loss0 + loss1
            #if self.local_rank == 0:
            #    print(f'Index: {index}, loss: {loss.item()}')
            loss.backward()
            self.optimizer.step()
            self.scheduler.step() # Update learning rate based on each training step, not each epoch

            loss_cum += loss.item()*length
            count += length

            if index == 0 and self.local_rank == 0 and ep%4==-1:
                # Avoid division by zero: Create a mask where p3_0 is non-zero
                residual = p3_0 - out0  
                nonzero_mask = (p3_0 != 0)
                
                l = torch.nn.MSELoss(reduction='none')(out0, p3_0)
                l_clamped = torch.clamp(l, min=-3, max=3)
                l_clamped = torch.round(l_clamped*1000)/1000

                print(f'l.shape: {l.shape}')

                print(f'===============')
                print(f'p3_0[:2, :5, :5]:')
                print(p3_0[:1, :5, :5])
                print("====================================================\n")
                print(f'out0[:2, :5, :5]:')
                print(out0[:1, :5, :5])
                print("====================================================\n")
                print(f'l_clamped[:2, :5]')
                print(l_clamped[:1, :5])
                print("====================================================\n")
                l_node = l.mean(dim=-1)
                l_graph = l_node.mean(dim=-1)

            # Cache management
            torch.cuda.empty_cache()
            
        return loss_cum/count


    #---------------------------------------------------------------
    @torch.no_grad()
    def _test_part(self, test_loader,):
        self.model.eval()
        loss_cum = 0
        count = 0
        
        for index, data in enumerate(test_loader):
            inputs, labels = data[0], data[1]   
            inputs = inputs.to(self.torch_device)

            jet0, jet1 = inputs[:,0, :, :], inputs[:,1, :, :]
            length = jet0.shape[0]


            if self.graph_transformer: graph = data[2].to(self.torch_device)
            else:  graph = None
            
            # zero the parameter gradients
            self.optimizer.zero_grad()

            #p3_0 = to_ptrapphim(jet0)
            #p3_1 = to_ptrapphim(jet1)
            p3_0 = jet0 
            p3_1 = jet1
            if self.input_dim == 1:
                # create pt of each particle instead of (px, py, pz, E) for the input 
                x0 = p3_0[:, 0, :].unsqueeze(1)
                x1 = p3_1[:, 0, :].unsqueeze(1)

            elif self.input_dim == 3: 
                x0 = p3_0
                x1 = p3_1

            else:
                x0, x1 = jet0, jet1

            out0 = self.model(x = x0, v = jet0, graph = graph) 
            out1 = self.model(x = x1, v = jet1, graph = graph) 
            p3_0 = p3_0.permute(0, 2, 1)
            p3_1 = p3_1.permute(0, 2, 1)
            
            loss0 = self.criterion(out0, p3_0) 
            loss1 = self.criterion(out1, p3_1) 
            loss = loss0 + loss1

            loss_cum += loss.item()*length
            count += length
        
        return loss_cum/count



    #---------------------------------------------------------------
    @torch.no_grad()
    def _plot_loss(self):
        self.model.eval()
        event_losses = []
        # A nodewise criterion
        criterion_node = torch.nn.MSELoss(reduction='none') # This 
        with torch.no_grad():
            for index, data in enumerate(self.test_loader):
                inputs, labels = data[0], data[1]   
                inputs = inputs.to(self.torch_device)

                jet0, jet1 = inputs[:,0, :, :], inputs[:,1, :, :]
                length = jet0.shape[0]


                if self.graph_transformer: graph = data[2].to(self.torch_device)
                else:  graph = None
                
                # zero the parameter gradients
                self.optimizer.zero_grad()

                #p3_0 = to_ptrapphim(jet0)
                #p3_1 = to_ptrapphim(jet1)
                p3_0 = jet0 
                p3_1 = jet1
                if self.input_dim == 1:
                    # create pt of each particle instead of (px, py, pz, E) for the input 
                    x0 = p3_0[:, 0, :].unsqueeze(1)
                    x1 = p3_1[:, 0, :].unsqueeze(1)

                elif self.input_dim == 3: 
                    x0 = p3_0
                    x1 = p3_1

                else:
                    x0, x1 = jet0, jet1

                out0 = self.model(x = x0, v = jet0, graph = graph)
                out1 = self.model(x = x1, v = jet1, graph = graph) 
                p3_0 = p3_0.permute(0, 2, 1)
                p3_1 = p3_1.permute(0, 2, 1)

                # 1) Nodewise loss => shape [N, F]
                loss0_nodewise = criterion_node(out0, p3_0)  
                loss1_nodewise = criterion_node(out1, p3_1)   

                # 2) Average across features => shape [N]
                loss0_per_node = loss0_nodewise.mean(dim=-1)
                loss1_per_node = loss1_nodewise.mean(dim=-1) 

                loss0_per_graph = loss0_per_node.sum(dim=1)/p3_0.shape[1]
                loss1_per_graph = loss1_per_node.sum(dim=1)/p3_1.shape[1]

                # 3) Aggregate nodewise losses by graph ID => shape [G], where G = number of graphs in the batch
                #loss0_per_graph = scatter_mean(loss0_per_node, p3_0, dim=0)
                #loss1_per_graph = scatter_mean(loss1_per_node, p3_1, dim=0)

                # 4) Compute the combined loss for each graph and append to event_losses
                scores = (loss0_per_graph + loss1_per_graph)  # Shape: [G]
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
        plt.legend()
        plt.grid()
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
            ax1.set_title(f'Loss Distribution and AUC, Transformer, n_part = {self.n_part}', fontsize='x-large')
            ax1.set_ylim(0.007, 0.07)
            ax1.grid()
            

            # Add the secondary y-axis for AUC
            ax2 = ax1.twinx()
            print(f'self.auc_list: {self.auc_list}')
            ax2.plot(epochs, self.auc_list, color='darkgreen', label='AUC', marker='o', linestyle='-')
            ax2.set_ylabel('AUC', color='black', fontsize='large')
            ax2.tick_params(axis='y', labelcolor='black', labelsize='medium')
            # Scale the AUC axis to the range 0.5 to 1
            ax2.set_ylim(0.795, 0.865)
            ax2.grid()

            # Add legend for AUC
            fig.legend(loc='center', bbox_to_anchor=(0.77, 0.5), fontsize='large', frameon=True, shadow=True, borderpad=1)

            # Save the figure
            plt.tight_layout()
            plt.savefig(plot_file)
            plt.close()     

            # Save the data to an HDF5 file
            h5_file = os.path.join('/pscratch/sd/d/dimathan/LHCO/GAE', f'loss_and_auc_data_trans_e{len(self.auc_list)}_n{self.n_part}.h5')
            with h5py.File(h5_file, 'w') as h5f:
                h5f.create_dataset('train_loss', data=self.train_loss_list)
                h5f.create_dataset('val_loss', data=self.val_loss_list)
                h5f.create_dataset('auc', data=self.auc_list)

            print(f"Saved loss and AUC data to: {h5_file}")

        return quantiles_loss





    #---------------------------------------------------------------
    def run_anomaly(self, ep=-1, plot = True):
        if self.local_rank == 0:
            self.model.eval()
            # A nodewise criterion
            criterion_node = torch.nn.MSELoss(reduction='none')

            all_scores = []  # this will store the continuous anomaly score
            all_labels = []  # ground-truth anomaly labels (0 or 1)

            with torch.no_grad():
                for index, data in enumerate(self.cl_loader):
                    inputs, labels = data[0], data[1]   
                    inputs = inputs.to(self.torch_device)

                    jet0, jet1 = inputs[:,0, :, :], inputs[:,1, :, :]
                    length = jet0.shape[0]

                    if self.graph_transformer: graph = data[2].to(self.torch_device)
                    else:  graph = None
                    
                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    #p3_0 = to_ptrapphim(jet0)
                    #p3_1 = to_ptrapphim(jet1)
                    p3_0 = jet0 
                    p3_1 = jet1
                    if self.input_dim == 1:
                        # create pt of each particle instead of (px, py, pz, E) for the input 
                        x0 = p3_0[:, 0, :].unsqueeze(1)
                        x1 = p3_1[:, 0, :].unsqueeze(1)

                    elif self.input_dim == 3: 
                        x0 = p3_0
                        x1 = p3_1

                    else:
                        x0, x1 = jet0, jet1
 
                    out0 = self.model(x = x0, v = jet0, graph = graph)  
                    out1 = self.model(x = x1, v = jet1, graph = graph) 
                    p3_0 = p3_0.permute(0, 2, 1)
                    p3_1 = p3_1.permute(0, 2, 1)

                    if index == 0 and self.local_rank == 0 and ep%5==-1:
                        # Avoid division by zero: Create a mask where p3_0 is non-zero
                        residual = p3_0 - out0  
                        nonzero_mask = (p3_0 != 0)

                        # Scale the residual by p3_0 where nonzero, else set it to 0
                        scaled_residual = torch.zeros_like(residual)
                        scaled_residual[nonzero_mask] = residual[nonzero_mask] / p3_0[nonzero_mask]
                        # Clip scaled_residual to [-1, 1]
                        scaled_residual = torch.clamp(scaled_residual, min=-1, max=1)
                        scaled_residual = torch.round(scaled_residual*1000)/1000


                        print(f'===============')
                        print(f'Anomaly Detection')
                        print(f'p3_0[:2, :5, :5]:')
                        print(p3_0[:2, :5, :5])
                        print()
                        
                        print(f'out0[:2, :5, :5]:')
                        print(out0[:2, :5, :5])
                        print()
                        print(f'scaled_residual[:2, :5, :5]:')
                        print(scaled_residual[:2, :5, :5])
                        #print("\n".join(["\t".join([f"{val:.2f}" for val in row]) for row in scaled_residual]))
                        print()
                        print(f'===============')

                    # 1) Nodewise loss => shape [N, F]
                    loss0_nodewise = criterion_node(out0, p3_0) 
                    loss1_nodewise = criterion_node(out1, p3_1) 

                    # 2) Average across features => shape [N]
                    loss0_per_node = loss0_nodewise.mean(dim=-1)
                    loss1_per_node = loss1_nodewise.mean(dim=-1)

                    loss0_per_graph = loss0_per_node.sum(dim=1)/p3_0.shape[1]
                    loss1_per_graph = loss1_per_node.sum(dim=1)/p3_1.shape[1]

                    # 3) Aggregate nodewise losses by graph ID => shape [G], where G = number of graphs in the batch
                    #loss0_per_graph = scatter_mean(loss0_per_node, p3_0, dim=0)
                    #loss1_per_graph = scatter_mean(loss1_per_node, p3_1, dim=0)

                    # 4) Compute the combined loss for each graph and append to event_losses
                    scores = (loss0_per_graph + loss1_per_graph)  
                    all_scores.extend(scores.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())

                # ----- Compute ROC and AUC -----
                fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
                auc_val = auc(fpr, tpr)


            print(f"Area Under Curve (AUC): {auc_val:.4f}")
            if plot:
                # ----- Plot the ROC curve -----
                plot_file = os.path.join(self.plot_path, "roc_curve.pdf")
                plt.figure(figsize=(6, 5))
                plt.plot(fpr, tpr, label=f'ROC (AUC = {auc_val:.3f})', color='b')
                plt.plot([0, 1], [0, 1], 'k--')  # diagonal line for "random" classification
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve")
                plt.legend(loc="lower right")
                # Add grid lines every 0.1
                plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
                plt.xticks(np.arange(0, 1.1, 0.1))  # X-axis grid at 0.1 intervals
                plt.yticks(np.arange(0, 1.1, 0.1))  # Y-axis grid at 0.1 intervals
                plt.tight_layout()

                # Save the plot
                plt.savefig(plot_file, dpi=300)
                plt.close()


                # ------------------------------------------------
                # 2) Plot loss distribution by label (normalized)
                # ------------------------------------------------
                normal_scores = [s for s, lbl in zip(all_scores, all_labels) if lbl == 0]
                anomalous_scores = [s for s, lbl in zip(all_scores, all_labels) if lbl == 1]


                # Compute the 99th percentile for both distributions
                p99_normal = np.percentile(normal_scores, 95)
                p99_anomalous = np.percentile(anomalous_scores, 95)

                # Determine the x-axis limit (max of the two 95th percentiles)
                x_max = max(p99_normal, p99_anomalous)

                # Define the bin edges based on [0, x_max]
                num_bins = 75  # Number of bins
                bin_edges = np.linspace(0, x_max, num_bins + 1)  # Create bins in the range [0, x_max]

                #all_scores_combined = normal_scores + anomalous_scores  # Combine all scores
                #bin_edges = np.histogram_bin_edges(all_scores_combined, bins=100)  # Compute bin edges

                dist_plot_file = os.path.join(self.plot_path, "loss_distribution_of_sig_vs_bkg.pdf")

                plt.figure(figsize=(6, 5))
                # density=True => each histogram integrates to 1, letting you compare shapes
                plt.hist(normal_scores, bins=bin_edges, label="Background (label=0)", color="green", histtype='step', density=True)
                plt.hist(anomalous_scores, bins=bin_edges, label="Signal (label=1)", color="red", histtype='step', density=True)

                plt.xlabel("Loss Score")
                plt.xlim(0, x_max)
                plt.ylabel("Density")
                plt.title("Loss Distribution by Label (Normalized)")
                plt.legend(loc="upper right")
                plt.grid()
                plt.tight_layout()
                plt.savefig(dist_plot_file, dpi=300)
                plt.close()
                print(f"Saved loss distribution plot to: {dist_plot_file}")
            
            return auc_val