"""
    Model definitions.
"""
import time 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_scatter import scatter_mean, scatter
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn import MetaLayer, EdgeConv, global_mean_pool, DynamicEdgeConv, GATConv
from models.layers import GraphConvolution
from torch_geometric.nn import MessagePassing


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch.nn import TransformerEncoder, TransformerEncoderLayer




# Custom EdgeConv that incorporates edge features.
class EdgeConvWithEdgeFeatures(MessagePassing):
    def __init__(self, nn_module, aggr='mean'):
        super(EdgeConvWithEdgeFeatures, self).__init__(aggr=aggr)
        self.nn = nn_module

    def forward(self, x, edge_index, edge_attr, dec):
        """
        Args:
            x: Node features tensor of shape [N, F].
            edge_index: Graph connectivity (COO format) with shape [2, E].
            edge_attr: Edge features tensor of shape [E, E_feat_dim].
        Returns:
            Updated node features.
        """
            # If edge_attr is None, create a dummy tensor with shape [E, F]
        if edge_attr is None:
            # Here, we assume the feature dimension F should match x's feature size.
            # Adjust the shape if needed.
            edge_attr = torch.zeros((edge_index.size(1), x.size(1)), device=x.device)

        # This will call self.message() with the appropriate arguments.
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, dec = dec)

    def message(self, x_i, x_j, edge_attr, dec=False):
        """
        Computes messages for each edge.
        Args:
            x_i: Features of the target node for each edge, shape [E, F].
            x_j: Features of the source node for each edge, shape [E, F].
            edge_attr: Features for each edge, shape [E, E_feat_dim].
        Returns:
            Messages for each edge.
        """
        # Concatenate the target node features, source node features, and the edge attributes.
        if dec: 
            edge_attr = x_j - x_i

        if edge_attr is None:
            msg_input = torch.cat([x_i], dim=-1)
        else:
            msg_input = torch.cat([x_i, edge_attr], dim=-1)
        return self.nn(msg_input)

    def update(self, aggr_out):
        # In this simple example, we directly output the aggregated messages.
        return aggr_out
    
    def __repr__(self):
        return f"EdgeConvWithEdgeFeatures(\n  {self.nn}\n)"


class EdgeAttrPredictor(nn.Module):
    def __init__(self, latent_dim, edge_dim, hidden_dim=32, dropout_rate=0.1):
        """
        Args:
            latent_dim (int): Dimension of the encoder's output per node.
            edge_dim (int): Dimension of the edge attributes.
            hidden_dim (int): Hidden layer size.
            dropout_rate (float): Dropout probability.
        """
        super(EdgeAttrPredictor, self).__init__()
        
        self.fc_direct = nn.Linear(2*latent_dim, edge_dim)
        
        # First layer: from concatenated features (2*latent_dim) to hidden_dim.
        self.fc1 = nn.Linear(2 * latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, edge_dim)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout_rate)
        
    

    def forward(self, x, edge_index, edge_index_fc=None):
        """
        Args:
            x: Node latent features from the encoder, shape [N, latent_dim].
            edge_index: Tensor of shape [2, E] with source and target indices.
        Returns:
            Predicted edge attributes of shape [E, edge_dim].
        """
        if edge_index_fc is None: edge_index_fc = edge_index

        src, tgt = edge_index  # source and target node indices for each edge

        # --- Symmetrize the input ---
        # Instead of simply concatenating [x[src], x[tgt]], we build a symmetric representation
        x_cat = torch.cat([torch.min(x[src], x[tgt]), torch.max(x[src], x[tgt])], dim=-1)


        # Initial transformation
        h = self.relu(self.fc1(x_cat))
        h = self.dropout(h)
        # Shape: [E, hidden_dim]

        # First Residual Block (using fc2)
        h_input_block1 = h # Input to this block
        h_transformed = self.fc2(h_input_block1) # Main transformation

        # Add residual connection *before* activation
        h = h_transformed + h_input_block1
        h = self.relu(h)        # Activation *after* adding
        h = self.dropout(h)     # Shape: [E, hidden_dim]

        # Final linear layer for the deeper path
        edge_pred = self.fc3(h) # Shape: [E, edge_dim]

        # --- Direct Path ---
        if self.fc_direct is not None:
            pred_direct = self.fc_direct(x_cat)
        
        else: pred_direct = x_cat # Direct linear path
        # Shape: [E, edge_dim]

        # --- Combine Paths ---
        return edge_pred + pred_direct



class Block(nn.Module):
    def __init__(self, node_in_dim, node_out_dim, edge_dim, hidden_dim = 32, aggr='mean', dropout_rate=0.1, dec=False, final_block=False):
        """
        An encoder block that first applies a linear transformation followed by ReLU and dropout,
        and then applies an EdgeConvWithEdgeFeatures operation.

        Args:
            node_in_dim (int): Input dimension of node features.
            node_out_dim (int): Output dimension for node features.
            edge_dim (int): Dimension of edge features.
            aggr (str): Aggregation method for EdgeConv (e.g., 'mean' or 'max').
            dropout_rate (float): Dropout probability.
        """
        super(Block, self).__init__()
        # Linear transformation followed by non-linearity and dropout.
        if node_in_dim != node_out_dim:
            self.residual = nn.Linear(node_in_dim, node_out_dim)
        else:
            self.residual = nn.Identity()
        
        self.dec = dec
        if dec: enc_in_dim = 2*node_in_dim  
        else: enc_in_dim = node_in_dim + edge_dim
        
        encoder_nn = nn.Sequential(
            nn.Linear(enc_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate), 
            nn.Linear(hidden_dim, node_out_dim),
            nn.ReLU() if not (final_block) else nn.Identity(),
        )
        
        self.edgeconv = EdgeConvWithEdgeFeatures(nn_module=encoder_nn, aggr=aggr)
        
        
    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass of the encoder block.

        Args:
            x (torch.Tensor): Node features of shape [N, node_in_dim].
            edge_index (torch.Tensor): Edge indices of shape [2, E].
            edge_attr (torch.Tensor or None): Edge features of shape [E, edge_dim].

        Returns:
            torch.Tensor: Updated node features of shape [N, node_out_dim].
        """
        # Linear transformation with activation and dropout.
        x_out = self.edgeconv(x, edge_index, edge_attr, dec=self.dec)
        # Residual connection: clone x if no residual layer is defined to avoid in-place modifications
        res = self.residual(x)
        # Apply the residual connection
        x_out = x_out + res
        return x_out


class RelGAE(nn.Module):
    def __init__(self,
                 input_dim=1,
                 edge_dim=3,
                 big_dim=64,
                 latent_dim=2,
                 output_dim=1,
                 encoder_layers=[64, 64,],
                 decoder_layers=[64, ],
                 aggr='mean',
                 dropout_rate=0.1):
        """
        A GNN autoencoder that constructs its encoder and decoder blocks using for loops.
        
        Args:
            input_dim (int): Dimensionality of node features.
            edge_dim (int): Dimensionality of edge features.
            big_dim (int): Hidden size used within each Block.
            latent_dim (int): Dimension of the latent space (encoder's final output).
            output_dim (int): Dimensionality of the reconstructed node features.
            encoder_layers (list of int): List specifying the output dimension of each encoder block.
                                          The last element must equal latent_dim.
            decoder_layers (list of int): List specifying the output dimension of each decoder block.
                                          The last element must equal output_dim.
            aggr (str): Aggregation method for EdgeConv (e.g., 'mean' or 'max').
            dropout_rate (float): Dropout probability.
        """

        super(RelGAE, self).__init__()
        encoder_layers = list(encoder_layers)  # create a new copy to avoid memory leaking when running many models
        decoder_layers = list(decoder_layers)  # create a new copy

        encoder_layers.append(latent_dim)
        decoder_layers.append(output_dim)

        # Check that the provided layer lists end with the proper dimensions.
        assert encoder_layers[-1] == latent_dim, "Last element of encoder_layers must equal latent_dim."
        assert decoder_layers[-1] == output_dim, "Last element of decoder_layers must equal output_dim."
        
        self.input_dim = input_dim
        
        # Build encoder blocks.
        # For the first encoder block, the input dimension is the raw input_dim.
        # For subsequent blocks, the input dimension is the previous block's output.
        self.encoder_blocks = nn.ModuleList()
        current_dim = input_dim
        for out_dim in encoder_layers:
            block = Block(
                node_in_dim=current_dim,
                node_out_dim=out_dim,
                edge_dim=edge_dim,
                hidden_dim=big_dim,
                aggr=aggr,
                dropout_rate=dropout_rate
            )
            self.encoder_blocks.append(block)
            current_dim = out_dim
        
        # ----DECODER---- 

        # Build decoder blocks.
        # Here we use edge_dim=0 since the decoder will compute edge features internally (e.g., using differences).
        self.decoder_blocks = nn.ModuleList()
        current_dim = latent_dim
        for index, out_dim in enumerate(decoder_layers):
            block = Block(
                node_in_dim=current_dim,
                node_out_dim=out_dim,
                edge_dim=0,  # No external edge features used in the decoder.
                hidden_dim=big_dim,
                aggr=aggr,
                dropout_rate=dropout_rate,
                dec = True, 
                final_block = (index == len(decoder_layers)-1),
            )
            self.decoder_blocks.append(block)
            current_dim = out_dim
                

        # Edge predictor head that uses the encoder's latent features to predict edge attributes.
        self.edge_predictor = EdgeAttrPredictor(
            latent_dim=latent_dim,
            edge_dim=edge_dim,
            hidden_dim=big_dim,
            dropout_rate=dropout_rate
        )


    def forward(self, data):
        """
        Args:
            data: A PyG data object with:
                  - x: Node features [N, input_dim]
                  - edge_index: Connectivity [2, E]
                  - edge_attr: True edge attributes [E, edge_dim]
        Returns:
            A tuple:
                - x_recon: Reconstructed node features [N, output_dim]
                - pred_edge_attr: Predicted edge attributes [E, edge_dim]
        """
        x = data.x  # [N, input_dim]

        # Pass through the encoder blocks.
        for block in self.encoder_blocks:
            x = block(x, data.edge_index, data.edge_attr)

        # Save the latent representation for edge prediction.
        x_encoded = x

        # ---Node Decoder---

        # Pass through the decoder blocks.
        x_recon = x
        for block in self.decoder_blocks:
            x_recon = block(x_recon, data.edge_index, None)

        # ---Edge Decoder---
        pred_edge_attr = self.edge_predictor(x_encoded, data.edge_index)

        return x_recon, pred_edge_attr

# ========================================================================
# ========================================================================
# ========================================================================
class EdgeNet_edge_VGAE(nn.Module):
    def __init__(self,
                 input_dim=3,
                 edge_dim=3,
                 big_dim=32,
                 latent_dim=2,
                 output_dim=3,
                 encoder_layers=[32, 32],
                 decoder_layers=[32,],
                 aggr='mean',
                 dropout_rate=0.1):
        """
        A variational version of EdgeNet_edge.
        """
        super(EdgeNet_edge_VGAE, self).__init__()
        # Copy the original deterministic encoder/decoder construction.
        encoder_layers = list(encoder_layers)
        decoder_layers = list(decoder_layers)
        encoder_layers.append(latent_dim)
        decoder_layers.append(output_dim)

        assert encoder_layers[-1] == latent_dim, "Last element of encoder_layers must equal latent_dim."
        assert decoder_layers[-1] == output_dim, "Last element of decoder_layers must equal output_dim."

        self.input_dim = input_dim
        
        # Build encoder blocks.
        self.encoder_bn = nn.ModuleList()
        self.encoder_blocks = nn.ModuleList()
        current_dim = input_dim
        for out_dim in encoder_layers:
            self.encoder_bn.append(nn.BatchNorm1d(current_dim))
            block = Block(
                node_in_dim=current_dim,
                node_out_dim=out_dim,
                edge_dim=edge_dim,
                hidden_dim=big_dim,
                aggr=aggr,
                dropout_rate=dropout_rate
            )
            self.encoder_blocks.append(block)
            current_dim = out_dim

        # Linear layers to get mu and logvar.
        self.fc_mu = nn.Linear(current_dim, latent_dim)
        self.fc_logvar = nn.Linear(current_dim, latent_dim)

        # Build decoder blocks.
        self.decoder_bn = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        current_dim = latent_dim
        for index, out_dim in enumerate(decoder_layers):
            self.decoder_bn.append(nn.BatchNorm1d(current_dim))
            block = Block(
                node_in_dim=current_dim,
                node_out_dim=out_dim,
                edge_dim=0,  # decoder does not use external edge features.
                hidden_dim=big_dim,
                aggr=aggr,
                dropout_rate=dropout_rate,
                dec=True,
                final_block=(index == len(decoder_layers)-1)
            )
            self.decoder_blocks.append(block)
            current_dim = out_dim
                
        # Edge predictor head remains unchanged.
        self.edge_predictor = EdgeAttrPredictor(
            latent_dim=latent_dim,
            edge_dim=edge_dim,
            hidden_dim=big_dim,
            dropout_rate=dropout_rate
        )

        # Batch normalization for input.
        self.input_bn = nn.BatchNorm1d(input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data):
        """
        Args:
            data: A PyG data object with attributes:
                  - x: Node features [N, input_dim]
                  - edge_index: Connectivity [2, E]
                  - edge_attr: True edge attributes [E, edge_dim]
        Returns:
            A tuple containing:
                - x_recon: Reconstructed node features [N, output_dim]
                - pred_edge_attr: Predicted edge attributes [E, edge_dim]
                - mu: Latent mean [N, latent_dim]
                - logvar: Latent log variance [N, latent_dim]
        """
        # Process node features.
        if self.input_dim == 3:
            x = data.x  # [N, input_dim]
        elif self.input_dim == 1:
            x = data.x[:,0].reshape(-1,1)
        x = self.input_bn(x)

        # Encoder: Pass through encoder blocks.
        for bn_layer, block in zip(self.encoder_bn, self.encoder_blocks):
            x = bn_layer(x)
            x = block(x, data.edge_index, data.edge_attr)
        
        # Save the encoder output.
        x_encoded = x

        # Obtain the variational parameters.
        mu = self.fc_mu(x_encoded)
        logvar = self.fc_logvar(x_encoded)
        
        # Sample latent variable.
        z = self.reparameterize(mu, logvar)

        # Decoder: Pass z through decoder blocks.
        x_recon = z
        for bn_layer, block in zip(self.decoder_bn, self.decoder_blocks):
            x_recon = bn_layer(x_recon)
            x_recon = block(x_recon, data.edge_index, None)

        # Edge attribute prediction from the latent representation.
        pred_edge_attr = self.edge_predictor(z, data.edge_index)

        return x_recon, pred_edge_attr, mu, logvar



# GNN AE using EdgeConv (mean aggregation graph operation). Basic GAE model.
class EdgeNet(nn.Module):
    def __init__(self, input_dim=4, big_dim=32, latent_dim=2, aggr='mean', dropout_rate=0.1):
        super(EdgeNet, self).__init__()
        encoder_nn = nn.Sequential(nn.Linear(2*(input_dim), big_dim),
                               nn.ReLU(),
                               nn.Dropout(p=dropout_rate), 
                               nn.Linear(big_dim, 2*big_dim),
                               nn.ReLU(),
                               nn.Dropout(p=dropout_rate), 
                               nn.Linear(2*big_dim, big_dim),
                               nn.ReLU(),
                               nn.Dropout(p=dropout_rate),
                               nn.Linear(big_dim, latent_dim),
                               nn.ReLU(),
        )
        
        decoder_nn = nn.Sequential(nn.Linear(2*(latent_dim), big_dim),
                               nn.ReLU(),
                               nn.Dropout(p=dropout_rate), 
                               nn.Linear(big_dim, 2*big_dim),
                               nn.ReLU(),
                               nn.Dropout(p=dropout_rate), 
                               nn.Linear(2*big_dim, big_dim),
                               nn.ReLU(),
                               nn.Dropout(p=dropout_rate), 
                               nn.Linear(big_dim, input_dim)
        )
        
        #self.batchnorm = nn.BatchNorm1d(input_dim)
        self.encoder = EdgeConv(nn=encoder_nn,aggr=aggr)
        self.decoder = EdgeConv(nn=decoder_nn,aggr=aggr)

    def forward(self, data):
        x = data.x
        #x = self.batchnorm(data.x)
        x = self.encoder(x,data.edge_index)
        x = self.decoder(x,data.edge_index)
        return x


####################################################################
####################################################################
####################################################################
class AE(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=512, latent_dim=2):
        super(AE, self).__init__()
        # Encoder: two hidden layers of 512 nodes each, then a bottleneck with 2 nodes.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),  # Batch normalization after the first hidden layer.
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)  # Bottleneck: linear activation.
        )
        
        # Decoder: two hidden layers of 512 nodes each, then an output layer with 10 nodes.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)  # Output layer: linear activation.
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent




####################################################################
#                       VARIATIONAL AUTOENCODER (VAE)              #
####################################################################
class VAE(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, latent_dim=2, activation='relu'):
        """
        Variational Autoencoder (VAE) based on the architecture described in
        arXiv:2103.06595v2 and implemented in vae-multiD.py.

        Args:
            input_dim (int): Dimensionality of the input features (e.g., 10 for flattened 2x5 observables).
            hidden_dim (int): Number of nodes in the hidden layers (e.g., 64).
            latent_dim (int): Dimensionality of the latent space (e.g., 1 or more).
            activation (str): Activation function to use ('relu', 'selu', etc.).
        """
        super(VAE, self).__init__()

        # Activation function selection
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'selu':
            act_fn = nn.SELU()
        # Add other activations as needed
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # Encoder layers (matching the structure in vae-multiD.py)
        self.encoder_fc1 = nn.Linear(input_dim, hidden_dim)
        self.encoder_act1 = act_fn
        self.encoder_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.encoder_act2 = act_fn
        self.encoder_fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.encoder_act3 = act_fn

        # Layers to output latent space parameters (mu and log_var)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder layers (symmetric to encoder)
        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_act1 = act_fn
        self.decoder_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_act2 = act_fn
        self.decoder_fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_act3 = act_fn

        # Final output layer (linear activation for reconstruction)
        self.decoder_output = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = self.encoder_act1(self.encoder_fc1(x))
        h = self.encoder_act2(self.encoder_fc2(h))
        h = self.encoder_act3(self.encoder_fc3(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample z = mu + epsilon * std.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # Sample epsilon from N(0, 1)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_act1(self.decoder_fc1(z))
        h = self.decoder_act2(self.decoder_fc2(h))
        h = self.decoder_act3(self.decoder_fc3(h))
        reconstruction = self.decoder_output(h) # Linear activation for output
        return reconstruction

    def forward(self, x):
        """
        Forward pass through the VAE.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].

        Returns:
            tuple: (reconstruction, mu, logvar)
                   - reconstruction (torch.Tensor): Reconstructed input [batch_size, input_dim].
                   - mu (torch.Tensor): Latent mean [batch_size, latent_dim].
                   - logvar (torch.Tensor): Latent log variance [batch_size, latent_dim].
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
