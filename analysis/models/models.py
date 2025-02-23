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


##############################################
# Encoder & Decoder Layer Definitions
##############################################
class EncoderLayer(nn.Module):
    """
    A single encoder layer that applies an EdgeConv with a residual connection.
    The MLP inside the EdgeConv transforms the concatenated [node, edge] input.
    """
    def __init__(self, in_dim, edge_dim, out_dim, aggr='mean', dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.edgeconv = EdgeConvWithEdgeFeatures(
            nn_module=nn.Sequential(
                nn.Linear(in_dim + edge_dim, out_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate)
            ),
            aggr=aggr
        )
        # If dimensions differ, project the input for the residual connection.
        #if in_dim != out_dim:
        #    self.residual = nn.Linear(in_dim, out_dim)
        #else:
        #    self.residual = None
        self.residual = nn.Linear(in_dim, out_dim)
    def forward(self, x, edge_index, edge_attr):
        out = self.edgeconv(x, edge_index, edge_attr)
        res = self.residual(x) if self.residual is not None else x
        return F.relu(out + res)

class DecoderLayer(nn.Module):
    """
    A single decoder layer that applies an EdgeConv with a residual connection.
    """
    def __init__(self, in_dim, edge_dim, out_dim, aggr='mean', dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.edgeconv = EdgeConvWithEdgeFeatures(
            nn_module=nn.Sequential(
                nn.Linear(in_dim + edge_dim, out_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate)
            ),
            aggr=aggr
        )
        #if in_dim != out_dim:
        #    self.residual = nn.Linear(in_dim, out_dim)
        #else:
        #    self.residual = None
        self.residual = nn.Linear(in_dim, out_dim)
    def forward(self, x, edge_index, edge_attr):
        out = self.edgeconv(x, edge_index, edge_attr)
        res = self.residual(x) if self.residual is not None else x
        return F.relu(out + res)

##############################################
# Stacked Graph Autoencoder Model
##############################################
class StackedEdgeNet_edge(nn.Module):
    def __init__(self,
                 input_dim=1,      # e.g., only pt on the nodes
                 edge_dim=3,       # e.g., ΔR (and possibly other relational features)
                 hidden_dim=16,    # intermediate hidden dimension
                 latent_dim=2,     # small bottleneck to force compression
                 output_dim=3,     # e.g., reconstruct 3-momentum (pt, eta, phi)
                 num_encoder_layers=3,
                 num_decoder_layers=3,
                 aggr='mean',
                 dropout_rate=0.1):
        """
        Args:
            input_dim (int): Dimensionality of node features (e.g., 1 for pt).
            edge_dim (int): Dimensionality of edge features (e.g., 3 for ΔR-based features).
            hidden_dim (int): Hidden dimension used in intermediate layers.
            latent_dim (int): Dimension of the latent space (bottleneck).
            output_dim (int): Dimensionality of the reconstructed node features.
            num_encoder_layers (int): Number of stacked encoder layers.
            num_decoder_layers (int): Number of stacked decoder layers.
            aggr (str): Aggregation method ('mean', etc.).
            dropout_rate (float): Dropout probability.
        """
        super(StackedEdgeNet_edge, self).__init__()
        self.input_dim = input_dim
        # Normalize the node input (e.g., pt)
        self.batchnorm = nn.BatchNorm1d(input_dim)

        ##############################################
        # Build Encoder: Stack multiple encoder layers
        ##############################################
        self.encoder_layers = nn.ModuleList()
        # First encoder layer: from input_dim to hidden_dim.
        self.encoder_layers.append(EncoderLayer(in_dim=input_dim, edge_dim=edge_dim, out_dim=hidden_dim,
                                                  aggr=aggr, dropout_rate=dropout_rate))
        # Intermediate encoder layers: hidden_dim -> hidden_dim.
        for _ in range(num_encoder_layers - 2):
            self.encoder_layers.append(EncoderLayer(in_dim=hidden_dim, edge_dim=edge_dim, out_dim=hidden_dim,
                                                      aggr=aggr, dropout_rate=dropout_rate))
        # Final encoder layer: from hidden_dim to latent_dim.
        self.encoder_layers.append(EncoderLayer(in_dim=hidden_dim, edge_dim=edge_dim, out_dim=latent_dim,
                                                  aggr=aggr, dropout_rate=dropout_rate))

        ##############################################
        # Build Decoder: Stack multiple decoder layers
        ##############################################
        self.decoder_layers = nn.ModuleList()
        # First decoder layer: from latent_dim to hidden_dim.
        self.decoder_layers.append(DecoderLayer(in_dim=latent_dim, edge_dim=edge_dim, out_dim=hidden_dim,
                                                  aggr=aggr, dropout_rate=dropout_rate))
        # Intermediate decoder layers: hidden_dim -> hidden_dim.
        for _ in range(num_decoder_layers - 2):
            self.decoder_layers.append(DecoderLayer(in_dim=hidden_dim, edge_dim=edge_dim, out_dim=hidden_dim,
                                                      aggr=aggr, dropout_rate=dropout_rate))
        # Final decoder: a simple linear layer (without message passing) to produce output.
        self.final_decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        """
        Args:
            data: A PyG data object with attributes:
                - x: Node features of shape [N, input_dim].
                - edge_index: Graph connectivity [2, E].
                - edge_attr: Edge features of shape [E, edge_dim].
        Returns:
            Reconstructed node features of shape [N, output_dim].
        """
        if self.input_dim == 3:
            x = data.x  # [N, input_dim]
        elif self.input_dim==1: 
            x = data.x[:,0].reshape(-1,1)
        x = self.batchnorm(x)
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # Encoder: apply each encoder layer sequentially.
        for layer in self.encoder_layers:
            x = layer(x, edge_index, edge_attr)
        latent = x  # latent representation: shape [N, latent_dim]

        # Decoder: apply each decoder layer sequentially.
        for layer in self.decoder_layers:
            x = layer(x, edge_index, edge_attr)
        # Final reconstruction (e.g., to 3D momentum)
        out = self.final_decoder(x)
        return out

##############################################
# Custom EdgeConv Layer with Edge Features
##############################################
class EdgeConvWithEdgeFeatures(MessagePassing):
    def __init__(self, nn_module, aggr='mean'):
        super(EdgeConvWithEdgeFeatures, self).__init__(aggr=aggr)
        self.nn = nn_module

    def forward(self, x, edge_index, edge_attr):
        # Propagate messages using the provided edge_index and edge attributes.
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, edge_attr):
        # Concatenate the target node feature with the edge attribute.
        msg_input = torch.cat([x_i, edge_attr], dim=-1)
        return self.nn(msg_input)

    def update(self, aggr_out):
        return aggr_out


##############################################
# GNN Encoder Branch: Stacked Encoder Layers with Residuals
##############################################
class EncoderLayer_gnn(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim, out_dim, aggr='mean', dropout_rate=0.1):
        super(EncoderLayer_gnn, self).__init__()
        enc_in_dim = in_dim + edge_dim
        
        encoder_nn = nn.Sequential(
            nn.Linear(enc_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate), 
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate), 
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )
        self.encoder = EdgeConvWithEdgeFeaturesold(nn_module=encoder_nn, aggr=aggr)

    def forward(self, x, edge_index, edge_attr):
        out = self.encoder(x, edge_index, edge_attr, dec=False)

        return out  
    
##############################################
# Global MLP Encoder: Produces a Single Global Vector
##############################################
class MLPGlobalEncoder(nn.Module):
    def __init__(self, in_dim, latent_dim, dropout_rate=0.1):
        super(MLPGlobalEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        # x is expected to be a single vector representing the entire graph.
        return self.mlp(x)

##############################################
# Transformer Global Encoder with [CLS] Token (Batched per Graph)
##############################################
class TransformerGlobalEncoder(nn.Module):
    def __init__(self, in_dim, d_model, nhead=2, num_layers=2, dropout_rate=0.1):
        """
        in_dim: Dimension of the raw node features (e.g., 3)
        d_model: The higher dimension for the Transformer (and for the [CLS] token)
        nhead: Number of attention heads (ensure d_model is divisible by nhead)
        num_layers: Number of Transformer encoder layers.
        """
        super(TransformerGlobalEncoder, self).__init__()
        self.embedding = nn.Linear(in_dim, d_model)
        # Learnable [CLS] token (will be expanded to each graph in the batch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                  dropout=dropout_rate, dim_feedforward=32)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        x: Node features for each graph, shape: [batch_size, num_nodes, in_dim]
        Returns:
            Global latent vector for each graph of shape: [batch_size, d_model]
        """
        batch_size, num_nodes, _ = x.shape
        x = self.embedding(x)  # [batch_size, num_nodes, d_model]
        
        # Expand the CLS token for each graph.
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, d_model]
        x = torch.cat([cls_tokens, x], dim=1)  # [batch_size, num_nodes+1, d_model]
        
        # Transformer expects [sequence_length, batch_size, d_model]
        x = x.transpose(0, 1)  # [num_nodes+1, batch_size, d_model]
        x = self.transformer_encoder(x)  # [num_nodes+1, batch_size, d_model]
        x = x.transpose(0, 1)  # [batch_size, num_nodes+1, d_model]
        
        # Return the CLS token for each graph.
        global_feature = x[:, 0, :]  # [batch_size, d_model]
        return global_feature


##############################################
# Global Cross-Attention Module (Batched per Graph)
##############################################
class GlobalCrossAttention(nn.Module):
    def __init__(self, node_dim, cls_dim, nhead=1, dropout=0.1):
        """
        node_dim: Dimension of node features from the GNN branch (e.g., 2)
        cls_dim: Dimension of the global [CLS] token (d_model, e.g., 10)
        """
        super(GlobalCrossAttention, self).__init__()
        # Project node features up to the high-dimensional (CLS) space.
        self.node_to_cls = None 
        if node_dim != cls_dim:
            self.node_to_cls = nn.Linear(node_dim, cls_dim)
        
        self.cross_attn = nn.MultiheadAttention(embed_dim=cls_dim, num_heads=nhead, dropout=dropout)
        # Project the refined global feature back to node_dim.
        self.cls_to_node = nn.Linear(cls_dim, node_dim)
      
    def forward(self, node_features, global_feature):
        """
        node_features: Batched node features, shape: [batch_size, num_nodes, node_dim]
        global_feature: Batched global features, shape: [batch_size, cls_dim]
        Returns:
            Refined global features for each graph, shape: [batch_size, node_dim]
        """
        batch_size, num_nodes, _ = node_features.shape
        if self.node_to_cls is not None:
            node_proj = self.node_to_cls(node_features)  # [batch_size, num_nodes, cls_dim]
        else: node_proj = node_features
        # Transpose for multihead attention: [num_nodes, batch_size, cls_dim]
        node_proj = node_proj.transpose(0, 1)
        # Prepare query: [1, batch_size, cls_dim]
        query = global_feature.unsqueeze(0)
        attn_output, _ = self.cross_attn(query=query, key=node_proj, value=node_proj)
        attn_output = attn_output.squeeze(0)  # [batch_size, cls_dim]
        refined_global = self.cls_to_node(attn_output)  # [batch_size, node_dim]
        return refined_global



##############################################
# Decoder: EdgeConv-Based Decoder for Reconstruction
##############################################
class EdgeConvDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, output_dim, edge_dim, dropout_rate=0.1, aggr='mean'):
        """
        in_dim: Dimension of the input per node (concatenated node & global latent).
        hidden_dim: Hidden dimension for the EdgeConv layers.
        output_dim: Dimension of the reconstruction (per node).
        edge_dim: Dimensionality of edge features.
        """
        super(EdgeConvDecoder, self).__init__()
        dec_in_dim =  in_dim * 2
        
        decoder_nn = nn.Sequential(
            nn.Linear(dec_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate), 
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate), 
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate), 
            nn.Linear(hidden_dim, output_dim)
        )

        self.decoder = EdgeConvWithEdgeFeaturesold(nn_module=decoder_nn, aggr=aggr)

    def forward(self, x, edge_index, edge_attr):
        x = self.decoder(x, edge_index, None, dec=True)
        return x

##############################################
# Hybrid Graph Autoencoder with Independent Encoder Branches
# and an EdgeConv-based Decoder
##############################################
class HybridEdgeNet(nn.Module):
    def __init__(self,
                 input_dim=1,    # e.g. only pT on the nodes
                 edge_dim=3,     # e.g. ΔR (and other relational features)
                 hidden_dim=16,  # hidden dimension for GNN branch and decoder
                 latent_dim=2,   # latent dimension for nodes and global representation
                 d_model = 2,
                 output_dim=3,   # reconstruct 3-momentum (pt, eta, phi)
                 num_encoder_layers=1,
                 aggr='mean',
                 dropout_rate=0.1):
        """
        - The EdgeConv branch encodes per-node latent representations.
        - The MLP branch produces a global latent vector independently.
        - The decoder uses concatenated (per-node + replicated global) representations,
          and further refines them using EdgeConv layers.
        """
        super(HybridEdgeNet, self).__init__()
        self.input_dim = input_dim
        # Batch normalization for the node input.
        self.batchnorm = nn.BatchNorm1d(input_dim)

        ##############################################
        # GNN Encoder Branch: Stacked Encoder Layers
        ##############################################
        self.gnn_encoder_layers = nn.ModuleList()
        self.gnn_encoder_layers.append(EncoderLayer_gnn(in_dim=input_dim, edge_dim=edge_dim, hidden_dim=hidden_dim, out_dim=latent_dim,
                                                        aggr=aggr, dropout_rate=dropout_rate))
        
        # Projection: from latent_dim to d_model (for global fusion)
        self.node_proj = nn.Linear(latent_dim, d_model)

        
        ##############################################
        # Transformer Global Encoder Branch: Produces a Global Vector
        ##############################################
        self.global_transformer_encoder = TransformerGlobalEncoder(in_dim=input_dim, d_model = d_model, dropout_rate=dropout_rate)
        
        self.global_cross_attention = GlobalCrossAttention(node_dim=d_model, cls_dim=d_model, nhead=1, dropout=dropout_rate)

        ##############################################
        # Decoder: Uses EdgeConv layers for a more expressive mapping.
        # Input dimension is (latent_dim (node) + latent_dim (global)) = 2*latent_dim.
        ##############################################
        self.decoder = EdgeConvDecoder(in_dim=d_model , hidden_dim=hidden_dim, output_dim=output_dim,
                                       edge_dim=edge_dim,  dropout_rate=dropout_rate, aggr=aggr)


    def forward(self, data):
        """
        Args:
            data: A PyG data object with attributes:
                  - x: Node features of shape [N, input_dim].
                  - edge_index: Graph connectivity [2, E].
                  - edge_attr: Edge features of shape [E, edge_dim].
        Returns:
            Reconstructed node features of shape [N, output_dim].
        """
        # Process node features.
        if self.input_dim == 3:
            x = data.x  # [N, input_dim]
        elif self.input_dim==1:
            x = data.x[:,0].reshape(-1,1)

        x = self.batchnorm(x)
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # 1. GNN branch: compute per-node latent representations (shape: [N, latent_dim]).
        z_nodes = x
        for layer in self.gnn_encoder_layers:
            z_nodes = layer(z_nodes, edge_index, edge_attr)
        
        # 2. Project GNN output from latent_dim to d_model.
        z_nodes_proj = self.node_proj(z_nodes)  # shape: [N, d_model]

        # 3. Get batch information (assumes data.batch is provided).
        batch = data.batch  # shape: [N]
        batch_size = int(batch.max().item() + 1)
        num_nodes_per_graph = z_nodes_proj.shape[0] // batch_size

        # 4. Reshape to batched form.
        # For the transformer branch, we use the raw input features.
        x_reshaped = x.view(batch_size, num_nodes_per_graph, -1)  # [batch_size, num_nodes, input_dim]
        # And reshape the projected node features.
        z_nodes_batched = z_nodes_proj.view(batch_size, num_nodes_per_graph, -1)  # [batch_size, num_nodes, d_model]

        # 5. Transformer branch: compute a per-graph global latent vector (in d_model).
        z_global = self.global_transformer_encoder(x_reshaped)  # [batch_size, d_model]

        # 6. Global Cross-Attention: refine the global feature using the per-graph node features.
        global_refined = self.global_cross_attention(z_nodes_batched, z_global)  # [batch_size, d_model]

        # 7. Fuse the projected node features with the refined global feature.
        # (Broadcast the global feature to each node in its graph.)
        z_final_batched = z_nodes_batched + global_refined.unsqueeze(1)  # [batch_size, num_nodes, d_model]

        # 8. Flatten back to [N, d_model] and pass to the decoder.
        z_final = z_final_batched.view(-1, z_final_batched.size(-1))

        out = self.decoder(z_final, edge_index, edge_attr)

        return out









# Custom EdgeConv that incorporates edge features.
class EdgeConvWithEdgeFeaturesold(MessagePassing):
    def __init__(self, nn_module, aggr='mean'):
        super(EdgeConvWithEdgeFeaturesold, self).__init__(aggr=aggr)
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
        # You can change the order or combine them in another way if desired.
        #edge_attr = x_j - x_i
        #print(f'x_i: {x_i.shape}, x_j: {x_j.shape}, edge_attr: {edge_attr.shape}')
        if dec: 
            edge_attr = x_j - x_i

        if edge_attr is None:
            msg_input = torch.cat([x_i], dim=-1)
        else:
            msg_input = torch.cat([x_i, edge_attr], dim=-1)
        #print(f'msg_input: {msg_input.shape}')
        return self.nn(msg_input)

    def update(self, aggr_out):
        # In this simple example, we directly output the aggregated messages.
        return aggr_out
    
    def __repr__(self):
        return f"EdgeConvWithEdgeFeaturesold(\n  {self.nn}\n)"


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
        # First layer: from concatenated features (2*latent_dim) to hidden_dim.
        self.fc1 = nn.Linear(2 * latent_dim, hidden_dim)
        # Second layer: further processing with a residual connection.
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Output layer: map to edge_dim.
        #self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, edge_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.bn1 = nn.BatchNorm1d(2*latent_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

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
        # Concatenate the latent features for the two endpoints.
        x_cat = torch.cat([x[src], x[tgt]], dim=-1)  # [E, 2 * latent_dim]
        x_cat = self.bn1(x_cat)
        h = self.relu(self.fc1(x_cat))
        h = self.dropout(h)
        h = self.bn2(h)
        # Residual block:
        h_res = self.fc2(h)
        h = h + self.bn3(self.relu(h_res))
        h = self.dropout(h)
        #h_res = self.fc3(h)
        #h = h + self.relu(h_res)
        #h = self.dropout(h)
        edge_pred = self.fc4(h)
        return edge_pred



class Block(nn.Module):
    def __init__(self, node_in_dim, node_out_dim, edge_dim, hidden_dim = 32, aggr='mean', dropout_rate=0.1, dec=False, final_block=False):
        """
        An encoder block that first applies a linear transformation followed by ReLU and dropout,
        and then applies an EdgeConvWithEdgeFeaturesold operation.

        Args:
            node_in_dim (int): Input dimension of node features.
            node_out_dim (int): Output dimension for node features.
            edge_dim (int): Dimension of edge features.
            aggr (str): Aggregation method for EdgeConv (e.g., 'mean' or 'max').
            dropout_rate (float): Dropout probability.
        """
        super(Block, self).__init__()
        # Linear transformation followed by non-linearity and dropout.

        #self.batchnorm = nn.BatchNorm1d(node_in_dim)
        self.dec = dec
        if dec: enc_in_dim = 2*node_in_dim  
        else: enc_in_dim = node_in_dim + edge_dim
        
        encoder_nn = nn.Sequential(
            nn.Linear(enc_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate), 
            nn.Linear(hidden_dim, node_out_dim),
            nn.ReLU() if not (dec and final_block) else nn.Identity(),
        )
        
        self.edgeconv = EdgeConvWithEdgeFeaturesold(nn_module=encoder_nn, aggr=aggr)
        
        
        # Define an MLP to be used inside EdgeConv.
        # It takes concatenated input: transformed node features (node_out_dim)
        # and edge features (edge_dim) and outputs a feature vector of size node_out_dim.
        #edge_mlp = nn.Sequential(
        #    nn.Linear(node_out_dim + edge_dim, node_out_dim),
        #    nn.ReLU(),
        #    nn.Dropout(p=dropout_rate)
        #)
        # Create the EdgeConv layer.
        #self.edgeconv = EdgeConvWithEdgeFeaturesold(nn_module=edge_mlp, aggr=aggr)
        
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
        #x = self.batchnorm(x)
        x_out = self.edgeconv(x, edge_index, edge_attr, dec=self.dec)
        return x_out


class EdgeNet_edge(nn.Module):
    def __init__(self,
                 input_dim=3,
                 edge_dim=3,
                 big_dim=32,
                 latent_dim=2,
                 output_dim=3,
                 encoder_layers=[32,],
                 decoder_layers=[32,],
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

        super(EdgeNet_edge, self).__init__()
        encoder_layers.append(latent_dim)
        decoder_layers.append(output_dim)

        # Check that the provided layer lists end with the proper dimensions.
        assert encoder_layers[-1] == latent_dim, "Last element of encoder_layers must equal latent_dim."
        assert decoder_layers[-1] == output_dim, "Last element of decoder_layers must equal output_dim."
        
        self.input_dim = input_dim
        
        # Build encoder blocks.
        # For the first encoder block, the input dimension is the raw input_dim.
        # For subsequent blocks, the input dimension is the previous block's output.
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
        
        # Build decoder blocks.
        # Here we use edge_dim=0 since the decoder will compute edge features internally (e.g., using differences).
        self.decoder_bn = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        current_dim = latent_dim
        for index, out_dim in enumerate(decoder_layers):
            self.decoder_bn.append(nn.BatchNorm1d(current_dim))
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
        # Process node features.
        if self.input_dim == 3:
            x = data.x  # [N, input_dim]
        elif self.input_dim == 1:
            x = data.x[:,0].reshape(-1,1)
            
        # Pass through the encoder blocks.
        for bn_layer, block in zip(self.encoder_bn, self.encoder_blocks):
            x = bn_layer(x)
            x = block(x, data.edge_index, data.edge_attr)
        # Save the latent representation for edge prediction.
        x_encoded = x

        # Pass through the decoder blocks.
        x_recon = x_encoded
        for bn_layer, block in zip(self.decoder_bn, self.decoder_blocks):
            x_recon = bn_layer(x_recon)
            x_recon = block(x_recon, data.edge_index, None)

        # Predict edge attributes using the latent node representations.
        pred_edge_attr = self.edge_predictor(x_encoded, data.edge_index)

        return x_recon, pred_edge_attr



# Example GNN Autoencoder using the modified EdgeConv.
class EdgeNet_edge_old(nn.Module):
    def __init__(self, input_dim=3, edge_dim=3, big_dim=32, latent_dim=2, output_dim=3, aggr='mean', dropout_rate=0.1):
        """
        Args:
            input_dim (int): Dimensionality of node features.
            edge_dim (int): Dimensionality of edge features.
            big_dim (int): Intermediate hidden layer size.
            hidden_dim (int): Output size of encoder (latent dim).
            aggr (str): Aggregation method ('mean', 'max', etc.).
            dropout_rate (float): Dropout probability.
        """
        super(EdgeNet_edge_old, self).__init__()
        
        # Note: our MLPs now need to account for the edge features.
        # The input to the encoder MLP is a concatenation of:
        #   - target node features (input_dim)
        #   - source node features (input_dim)
        #   - edge features (edge_dim)
        self.input_dim = input_dim
        enc_in_dim = input_dim + edge_dim
        
        encoder_nn = nn.Sequential(
            nn.Linear(enc_in_dim, big_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate), 
            nn.Linear(big_dim, 2 * big_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate), 
            nn.Linear(2 * big_dim, big_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(big_dim, latent_dim),
            nn.ReLU(),
        )
        
        
        dec_in_dim =  latent_dim * 2
        
        decoder_nn = nn.Sequential(
            nn.Linear(dec_in_dim, big_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate), 
            nn.Linear(big_dim, 2 * big_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate), 
            nn.Linear(2 * big_dim, big_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate), 
            nn.Linear(big_dim, output_dim)
        )

        # Use batch normalization on the input node features.
        self.batchnorm = nn.BatchNorm1d(input_dim)
        
        self.batchnorm2 = nn.BatchNorm1d(latent_dim)
        # Replace the standard EdgeConv with our custom version.
        self.encoder = EdgeConvWithEdgeFeaturesold(nn_module=encoder_nn, aggr=aggr)
        self.decoder = EdgeConvWithEdgeFeaturesold(nn_module=decoder_nn, aggr=aggr)

        # --- New: Edge Predictor Head ---
        # Predict edge attributes using the encoder's latent node features.
        self.edge_predictor = EdgeAttrPredictor(latent_dim, edge_dim, hidden_dim=32, dropout_rate=dropout_rate)


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
        # Process node features.
        if self.input_dim == 3:
            x = data.x  # [N, input_dim]
        elif self.input_dim==1:
            x = data.x[:,0].reshape(-1,1)
            
        # Process node features.
        x = self.batchnorm(x)
        # Compute latent node representations from the encoder.
        x_encoded = self.encoder(x, data.edge_index, data.edge_attr, dec=False)
        
        x_encoded = self.batchnorm2(x_encoded)

        # Reconstruct node features using the decoder.
        x_recon = self.decoder(x_encoded, data.edge_index, None, dec=True)
        # Predict edge attributes using the encoder's latent node features.
        pred_edge_attr = self.edge_predictor(x_encoded, data.edge_index)

        return x_recon, pred_edge_attr






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
        
        self.batchnorm = nn.BatchNorm1d(input_dim)
        self.encoder = EdgeConv(nn=encoder_nn,aggr=aggr)
        self.decoder = EdgeConv(nn=decoder_nn,aggr=aggr)

    def forward(self, data):
        x = self.batchnorm(data.x)
        x = self.encoder(x,data.edge_index)
        x = self.decoder(x,data.edge_index)
        return x







# GNN AE using EdgeConv (mean aggregation graph operation). Basic GAE model.
class EdgeNet_ext(nn.Module):
    def __init__(self, input_dim=4, big_dim=32, latent_dim=2, aggr='mean', dropout_rate=0.1):
        super(EdgeNet, self).__init__()
        encoder_nn = nn.Sequential(nn.Linear(2*(input_dim), big_dim),
                               nn.ReLU(),
                               nn.Dropout(p=dropout_rate), 
                               nn.Linear(big_dim, big_dim),
                               nn.ReLU(),
                               nn.Dropout(p=dropout_rate), 
                               nn.Linear(big_dim, latent_dim),
                               nn.ReLU(),
        )
        
        decoder_nn = nn.Sequential(nn.Linear(2*(latent_dim), big_dim),
                               nn.ReLU(),
                               nn.Dropout(p=dropout_rate), 
                               nn.Linear(big_dim, big_dim),
                               nn.ReLU(),
                               nn.Dropout(p=dropout_rate), 
                               nn.Linear(big_dim, input_dim)
        )
        
        self.batchnorm = nn.BatchNorm1d(input_dim)

        self.encoder = EdgeConv(nn=encoder_nn,aggr=aggr)
        self.decoder = EdgeConv(nn=decoder_nn,aggr=aggr)

    def forward(self, data):
        x = self.batchnorm(data.x)
        x = self.encoder(x,data.edge_index)
        x = self.decoder(x,data.edge_index)
        return x


