"""
    Model definitions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_scatter import scatter_mean, scatter
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn import MetaLayer, EdgeConv, global_mean_pool, DynamicEdgeConv, GATConv
from models.layers import GraphConvolution


# GNN AE using EdgeConv (mean aggregation graph operation). Basic GAE model.
class EdgeNet(nn.Module):
    def __init__(self, input_dim=4, big_dim=32, hidden_dim=2, aggr='mean', dropout_rate=0.1):
        super(EdgeNet, self).__init__()
        encoder_nn = nn.Sequential(nn.Linear(2*(input_dim), big_dim),
                               nn.ReLU(),
                               nn.Dropout(p=dropout_rate), 
                               nn.Linear(big_dim, big_dim),
                               nn.ReLU(),
                               nn.Dropout(p=dropout_rate), 
                               nn.Linear(big_dim, hidden_dim),
                               nn.ReLU(),
        )
        
        decoder_nn = nn.Sequential(nn.Linear(2*(hidden_dim), big_dim),
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



class GATAE(nn.Module):
    """
    Simple Graph Autoencoder using GATConv for 'attention'.
    It has:
      - An encoder: [BatchNorm -> GATConv -> (ReLU) -> GATConv -> ...]
      - A decoder: [GATConv -> (ReLU) -> GATConv -> ...]
      - Optional skip connections.
    """

    def __init__(
        self, 
        input_dim=4, 
        hidden_dim=32, 
        latent_dim=2, 
        heads=4,            # number of attention heads
        dropout=0.0,        # dropout inside GATConv
        add_skip=False       # whether to include skip connections
    ):
        super().__init__()

        # ---- 1) Normalization of inputs ----
        self.input_bn = nn.BatchNorm1d(input_dim)

        # ---- 2) Encoder layers ----
        # First GATConv: from input_dim -> hidden_dim, multiplied by 'heads' if concat=True
        self.gat_enc1 = GATConv(
            in_channels=input_dim, 
            out_channels=hidden_dim, 
            heads=heads, 
            concat=True,        # output size is hidden_dim * heads
            dropout=dropout
        )

        #self.gat_enc2 = GATConv(
        #            in_channels=hidden_dim*heads, 
        #            out_channels=hidden_dim, 
        #            heads=heads, 
        #            concat=True,        # output size is hidden_dim * heads
        #            dropout=dropout        )


        # Second GATConv: reduce from hidden_dim*heads -> latent_dim
        self.gat_enc3 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=latent_dim,
            heads=1,            # single head so output size = latent_dim
            concat=False,
            dropout=dropout
        )

        # ---- 3) Decoder layers ----
        # We basically reverse the process. 
        # from latent_dim -> hidden_dim*heads
        self.gat_dec1 = GATConv(
            in_channels=latent_dim, 
            out_channels=hidden_dim, 
            heads=heads, 
            concat=True,
            dropout=dropout
        )


        # from hidden_dim*heads -> hidden_dim*heads
        #self.gat_dec2 = GATConv(
        #    in_channels=hidden_dim * heads, 
        #    out_channels=hidden_dim, 
        #    heads=heads, 
        #    concat=True,
        #    dropout=dropout)


        # from hidden_dim*heads -> input_dim
        self.gat_dec3 = GATConv(
            in_channels=hidden_dim * heads, 
            out_channels=input_dim, 
            heads=1, 
            concat=False,
            dropout=dropout
        )

        self.add_skip = add_skip


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # ----- ENCODER -----
        x_in = self.input_bn(x)   # (Optional) batchnorm on input features

        # GAT encoder layer 1
        enc1 = self.gat_enc1(x_in, edge_index)   # shape ~ [num_nodes, hidden_dim * heads]
        enc1 = F.relu(enc1)

        # GAT encoder layer 2
        #enc2 = self.gat_enc2(enc1, edge_index)   # shape ~ [num_nodes, hidden_dim * heads]
        #enc2 = F.relu(enc2)

        # GAT encoder layer 2 => "latent representation"
        latent = self.gat_enc3(enc1, edge_index) # shape ~ [num_nodes, latent_dim]

        # ----- DECODER -----
        # GAT decoder layer 1
        dec1 = self.gat_dec1(latent, edge_index) # shape ~ [num_nodes, hidden_dim*heads]
        dec1 = F.relu(dec1)

        # GAT decoder layer 2
        #dec2 = self.gat_dec2(dec1, edge_index)   # shape ~ [num_nodes, hidden_dim*heads]
        #dec2 = F.relu(dec2)

        # GAT decoder layer 2 => reconstruct back to input_dim
        x_out = self.gat_dec3(dec1, edge_index)  # shape ~ [num_nodes, input_dim]

        # (Optional) skip from input straight to the output
        if self.add_skip:
            x_out = x_out + x_in

        return x_out



# GVAE based on EdgeNet model above.
class EdgeNetVAE(nn.Module):
    def __init__(self, input_dim=4, big_dim=32, hidden_dim=2, aggr='mean'):
        super(EdgeNetVAE, self).__init__()
        encoder_nn = nn.Sequential(nn.Linear(2*(input_dim), big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, big_dim),
                               nn.ReLU()
        )
        
        self.mu_layer = nn.Linear(big_dim, hidden_dim)
        self.var_layer = nn.Linear(big_dim, hidden_dim)
        
        decoder_nn = nn.Sequential(nn.Linear(2*(hidden_dim), big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, input_dim)
        )
        
        self.batchnorm = nn.BatchNorm1d(input_dim)

        self.encoder = EdgeConv(nn=encoder_nn,aggr=aggr)
        self.decoder = EdgeConv(nn=decoder_nn,aggr=aggr)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, data):
        x = self.batchnorm(data.x)
        x = self.encoder(x,data.edge_index)
        mu = self.mu_layer(x)
        log_var = self.var_layer(x)
        z = self.reparameterize(mu, log_var)
        x = self.decoder(z,data.edge_index)
        return x, mu, log_var

# 2 EdgeConv Wider
class EdgeNetDeeper(nn.Module):
    def __init__(self, input_dim=4, big_dim=32, hidden_dim=2, aggr='mean'):
        super(EdgeNetDeeper, self).__init__()

        encoder_nn_1 = nn.Sequential(nn.Linear(2*(input_dim), big_dim*2),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim*2),
                                   nn.Linear(big_dim*2, big_dim*2),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim*2),
                                   nn.Linear(big_dim*2, big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim)
        )
        encoder_nn_2 = nn.Sequential(nn.Linear(2*(big_dim), big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim),
                                   nn.Linear(big_dim, big_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim),
                                   nn.Linear(big_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim)
        )
        decoder_nn_1 = nn.Sequential(nn.Linear(2*(hidden_dim), big_dim),
                                   nn.ReLU(),
                                   nn.Linear(big_dim, big_dim),
                                   nn.BatchNorm1d(big_dim),
                                   nn.ReLU(),
                                   nn.Linear(big_dim, big_dim*2),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim*2)
        )
        decoder_nn_2 = nn.Sequential(nn.Linear(2*(big_dim*2), big_dim*2),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim*2),
                                   nn.Linear(big_dim*2, big_dim*2),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(big_dim*2),
                                   nn.Linear(big_dim*2, input_dim)
        )

        self.batchnorm = nn.BatchNorm1d(input_dim)

        self.encoder_1 = EdgeConv(nn=encoder_nn_1,aggr=aggr)
        self.encoder_2 = EdgeConv(nn=encoder_nn_2,aggr=aggr)
        self.decoder_1 = EdgeConv(nn=decoder_nn_1,aggr=aggr)
        self.decoder_2 = EdgeConv(nn=decoder_nn_2,aggr=aggr)

    def forward(self, data):
        x = self.batchnorm(data.x)
        x = self.encoder_1(x,data.edge_index)
        x = self.encoder_2(x,data.edge_index)
        x = self.decoder_1(x,data.edge_index)
        x = self.decoder_2(x,data.edge_index)
        return x

# GNN AE using EdgeConv (mean aggregation graph operation). Basic GAE model.
class EdgeNet2(nn.Module):
    def __init__(self, input_dim=4, big_dim=32, hidden_dim=2, aggr='mean'):
        super(EdgeNet2, self).__init__()
        encoder_nn = nn.Sequential(nn.Linear(2*(input_dim), big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, hidden_dim),
                               nn.ReLU(),
        )
        
        decoder_nn = nn.Sequential(nn.Linear(2*(hidden_dim), big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, input_dim)
        )
        
        self.batchnorm = nn.BatchNorm1d(input_dim)

        self.encoder = EdgeConv(nn=encoder_nn,aggr=aggr)
        self.decoder = EdgeConv(nn=decoder_nn,aggr=aggr)

    def forward(self, data):
        x = self.batchnorm(data.x)
        x = self.encoder(x,data.edge_index)
        x = self.decoder(x,data.edge_index)
        F.relu(x[:,0], inplace=True)
        return x

