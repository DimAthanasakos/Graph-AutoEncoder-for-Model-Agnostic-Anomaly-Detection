''' Particle Transformer (ParT)

Paper: "Particle Transformer for Jet Tagging" - https://arxiv.org/abs/2202.03772
Link to the full model implementation:
https://github.com/hqucms/weaver-core/blob/main/weaver/nn/model/ParticleTransformer.py

In this version I have deleted the two extra classes: ParticleTransformerTagger, ParticleTransformerTaggerWithExtraPairFeatures
since we only care about the kinematic features of the particles.

'''
import math
import random
import warnings
import copy
import torch
import torch.nn as nn
from functools import partial
import numpy as np
import time
import random 

import warnings


@torch.jit.script
def delta_phi(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


@torch.jit.script
def delta_r2(eta1, phi1, eta2, phi2):
    return (eta1 - eta2)**2 + delta_phi(phi1, phi2)**2


def p4s_from_ptyphims(ptyphims):
    """
    Calculate Cartesian four-vectors from transverse momenta (pt),
    rapidities (y), azimuthal angles (phi), and (optionally) masses (m).

    Arguments:
        ptyphims : torch.Tensor or array-like
          A shape (...,3) or (...,4) tensor storing [pt, y, phi, (m)].

    Returns:
        torch.Tensor
          An array (...,4) of Cartesian four-vectors [E, px, py, pz].
    """
    # Convert input to a torch Tensor of float type
    ptyphims = torch.as_tensor(ptyphims, dtype=torch.float)

    # Slice out pt, y, phi
    pts  = ptyphims[..., 0:1]  # (...,1)
    ys   = ptyphims[..., 1:2]  # (...,1)
    phis = ptyphims[..., 2:3]  # (...,1)

    # If a mass is present, slice that out; otherwise fill with zeros
    if ptyphims.shape[-1] == 4:
        ms = ptyphims[..., 3:4]
    else:
        ms = torch.zeros_like(pts)

    # Compute transverse energy
    Ets = torch.sqrt(pts**2 + ms**2)

    # Build the four-vector: [E, px, py, pz]
    p4s = torch.cat([
        Ets * torch.cosh(ys),   # E
        pts * torch.cos(phis),  # px
        pts * torch.sin(phis),  # py
        Ets * torch.sinh(ys),   # pz
    ], dim=-1)

    return p4s


def to_pt2(x, eps=1e-8):
    pt2 = x[:, :2].square().sum(dim=1, keepdim=True)
    if eps is not None:
        pt2 = pt2.clamp(min=eps)
    return pt2


def to_m2(x, eps=1e-8):
    m2 = x[:, 3:4].square() - x[:, :3].square().sum(dim=1, keepdim=True)
    if eps is not None:
        m2 = m2.clamp(min=eps)
    return m2


def atan2(y, x):
    sx = torch.sign(x)
    sy = torch.sign(y)
    pi_part = (sy + sx * (sy ** 2 - 1)) * (sx - 1) * (-math.pi / 2)
    atan_part = torch.arctan(y / (x + (1 - sx ** 2))) * sx ** 2
    return atan_part + pi_part

# Transform the 4-momentum vector to the (pt, rapidity, phi, mass) representation
def to_ptrapphim(x, return_mass=True, eps=1e-8): 
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)
    pt = torch.sqrt(to_pt2(x, eps=eps))
    # rapidity = 0.5 * torch.log((energy + pz) / (energy - pz))
    rapidity = 0.5 * torch.log(1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    phi = torch.atan2(py, px)
    
    if not return_mass:
        return torch.cat((pt, rapidity, phi), dim=1)
    else:
        m = torch.sqrt(to_m2(x, eps=eps))
        return torch.cat((pt, rapidity, phi, m), dim=1)


def boost(x, boostp4, eps=1e-8):
    # boost x to the rest frame of boostp4
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    p3 = -boostp4[:, :3] / boostp4[:, 3:].clamp(min=eps)
    b2 = p3.square().sum(dim=1, keepdim=True)
    gamma = (1 - b2).clamp(min=eps)**(-0.5)
    gamma2 = (gamma - 1) / b2
    gamma2.masked_fill_(b2 == 0, 0)
    bp = (x[:, :3] * p3).sum(dim=1, keepdim=True)
    v = x[:, :3] + gamma2 * bp * p3 + x[:, 3:] * gamma * p3
    return v


def p3_norm(p, eps=1e-8):
    return p[:, :3] / p[:, :3].norm(dim=1, keepdim=True).clamp(min=eps)

# TODO:
# The current input to this functions requires x to be [px, py, pz, E] and then transformed to [pt, eta, phi]
# Our datasets by default produce [pt, eta, phi] so currently we do the redundant transformation: [pt, eta, phi] -> [px, py, pz, E] in models.ParticleTransformer 
# and then -> [pt, eta, phi] here: 
def pairwise_lv_fts(xi, xj, num_outputs=4, eps=1e-8):
    #pti, rapi, phii = to_ptrapphim(xi, False, eps=None).split((1, 1, 1), dim=1)
    #ptj, rapj, phij = to_ptrapphim(xj, False, eps=None).split((1, 1, 1), dim=1)
    
    pti, rapi, phii = xi.split((1, 1, 1), dim=1)
    ptj, rapj, phij = xj.split((1, 1, 1), dim=1)
    #print(f'xi.shape: {xi.shape}')
    #print(f'xi[0]: {xi[0]}')
    #print(f'num_outputs: {num_outputs}')

    delta = delta_r2(rapi, phii, rapj, phij).sqrt()
    lndelta = torch.log(delta.clamp(min=eps))
    if num_outputs == 1:
        return lndelta

    if num_outputs > 1:
        ptmin = torch.minimum(pti, ptj)
        lnkt = torch.log((ptmin * delta).clamp(min=eps))
        lnz = torch.log((ptmin / (pti + ptj).clamp(min=eps)).clamp(min=eps))
        outputs = [lnkt, lnz, lndelta]

    if num_outputs > 3:
        xi = p4s_from_ptyphims(xi)
        xj = p4s_from_ptyphims(xj)
        print(f'xi.shape: {xi.shape}')
        print(f'xi[0]: {xi[0]}')
        xij = xi + xj
        lnm2 = torch.log(to_m2(xij, eps=eps))
        print(f'lnm2.shape: {lnm2.shape}')
        print(f'lnm2[:5]: {lnm2[:5]}')
        outputs.append(lnm2)

    if num_outputs > 4:
        lnds2 = torch.log(torch.clamp(-to_m2(xi - xj, eps=None), min=eps))
        outputs.append(lnds2)

    # the following features are not symmetric for (i, j)
    if num_outputs > 5:
        xj_boost = boost(xj, xij)
        costheta = (p3_norm(xj_boost, eps=eps) * p3_norm(xij, eps=eps)).sum(dim=1, keepdim=True)
        outputs.append(costheta)

    if num_outputs > 6:
        deltarap = rapi - rapj
        deltaphi = delta_phi(phii, phij)
        outputs += [deltarap, deltaphi]

    assert (len(outputs) == num_outputs)
    #print(f'len(outputs): {len(outputs)}')
    #print(f'outputs[0].shape: {outputs[0].shape}')
    aux = torch.cat(outputs, dim=1)
    #print(f'aux.shape: {aux.shape}')
    return torch.cat(outputs, dim=1)


def build_sparse_tensor(uu, idx, seq_len):
    # inputs: uu (N, C, num_pairs), idx (N, 2, num_pairs). uu holds the values of the pairs whose indices are in idx. 
    # return: (N, C, seq_len, seq_len)
    # N = batch_size, C = num_fts 
     
    batch_size, num_fts, num_pairs = uu.size()
    idx = torch.min(idx, torch.ones_like(idx) * seq_len) # ensures that all indices in idx refer to particles with index < seq_len  
    i = torch.cat((
        torch.arange(0, batch_size, device=uu.device).repeat_interleave(num_fts * num_pairs).unsqueeze(0), # tensor with size [1, batch_size * num_fts * num_pairs]
        torch.arange(0, num_fts, device=uu.device).repeat_interleave(num_pairs).repeat(batch_size).unsqueeze(0), # tensor with size [1, batch_size * num_fts * num_pairs]
        idx[:, :1, :].expand_as(uu).flatten().unsqueeze(0), # row indices of the pairs in the sparse tensor rep 
        idx[:, 1:, :].expand_as(uu).flatten().unsqueeze(0), # col indices of the pairs in the sparse tensor rep
    ), dim=0) # tensor with size [4, batch_size * num_fts * num_pairs]
    
    return torch.sparse_coo_tensor(
        i, uu.flatten(),
        size=(batch_size, num_fts, seq_len + 1, seq_len + 1),
        device=uu.device).to_dense()[:, :, :seq_len, :seq_len] # tensor with size [batch_size, num_fts, seq_len, seq_len]. It is a dense rep of the sparse tensor.



class SequenceTrimmer(nn.Module):

    def __init__(self, enabled=False, target=(0.9, 1.02), **kwargs) -> None:
        super().__init__(**kwargs)
        self.enabled = enabled
        self.target = target
        self._counter = 0

    def forward(self, x, v=None, mask=None, uu=None, graph=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # uu: (N, C', P, P)
        if mask is None:
            mask = torch.ones_like(x[:, :1])
        mask = mask.bool()

        if self.enabled:
            if self._counter == 0: 
                print()
                print(f"Trimmer enabled")
                print()

            if self._counter < 5:
                self._counter += 1
            else:
                if self.training: # An attribute of the nn.Module class. It is set to True when the model is in training mode, i.e. model.train() is called.
                    #print()
                    q = min(1, random.uniform(*self.target)) 
                    #print(f"q = {q}")
                    maxlen = torch.quantile(mask.type_as(x).sum(dim=-1), q).long() # maxlen is always 139 since we dont provide a mask
                    #print(f"maxlen = {maxlen}")
                    rand = torch.rand_like(mask.type_as(x))
                    rand = rand.masked_fill(~mask, -1)
                    #print(f"rand.shape = {rand.shape}")
                    #print(f"rand = {rand}")
                    perm = rand.argsort(dim=-1, descending=True)  # (N, 1, P). This returns the indices that would sort the rand tensor 
                    mask = torch.gather(mask, -1, perm)
                    #print(f"x[0] = {x[0]}")
                    x = torch.gather(x, -1, perm.expand_as(x))    # Permutes the elements of the tensor x according to the indices in perm.
                    if v is not None:
                        v = torch.gather(v, -1, perm.expand_as(v))
                    if uu is not None:
                        uu = torch.gather(uu, -2, perm.unsqueeze(-1).expand_as(uu))
                        uu = torch.gather(uu, -1, perm.unsqueeze(-2).expand_as(uu))
                    if graph is not None: 
                        graph = torch.gather(graph, -1, perm.expand_as(graph))
                        graph = torch.gather(graph, -2, perm.expand_as(graph))
                else:
                    maxlen = mask.sum(dim=-1).max()
                maxlen = max(maxlen, 1)
                if maxlen < mask.size(-1):
                    mask = mask[:, :, :maxlen]
                    x = x[:, :, :maxlen]
                    if v is not None:
                        v = v[:, :, :maxlen]
                    if uu is not None:
                        uu = uu[:, :, :maxlen, :maxlen]
                    if graph is not None:
                        graph = graph[:, :, :maxlen, :maxlen]
                

        return x, v, mask, uu, graph


class Embed(nn.Module):
    def __init__(self, input_dim, dims, normalize_input=False, activation='gelu'):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim) if normalize_input else None
        module_list = []
        starting_dim = input_dim
        for index, dim in enumerate(dims):
            module_list.extend([
                # WARNING THE >2 WAS >1
                #nn.LayerNorm(input_dim) if starting_dim > 2  else nn.Identity(), # LayerNorm averages across the feature space for the same particle. For 1d input space this leads to a random classifier.
                                                                                 # Surprisingly, for 1d input space, although we can use LayerNorm for later layers, this decreases the performance. 
                nn.Linear(input_dim, dim),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
            ])
            input_dim = dim
        self.embed = nn.Sequential(*module_list)

    def forward(self, x):
        #print(f'forward  Embed: x.shape: {x.shape}')
        if self.input_bn is not None:
            # x: (batch, embed_dim, seq_len)
            x = self.input_bn(x)
        x = x.permute(2, 0, 1).contiguous()
        #print(f'forward  Embed: x.shape: {x.shape}')
        # x: (seq_len, batch, embed_dim)
        x= self.embed(x)
        #print(f'forward  Embed: x.shape: {x.shape}')
        #time.sleep(10)
        return x


class PairEmbed(nn.Module):
    def __init__(
            self, pairwise_lv_dim, pairwise_input_dim, dims,
            remove_self_pair=False, use_pre_activation_pair=True, mode='sum',
            normalize_input=False, activation='gelu', eps=1e-8,):
        super().__init__()

        self.pairwise_lv_dim = pairwise_lv_dim # the number of features for the pairwise interaction terms. The default is 4 for [lnΔ, lnk_t, lnz, lnm^2]
        self.pairwise_input_dim = pairwise_input_dim
        self.is_symmetric = (pairwise_lv_dim <= 5) and (pairwise_input_dim == 0)
        self.remove_self_pair = remove_self_pair
        self.mode = mode
        self.pairwise_lv_fts = partial(pairwise_lv_fts, num_outputs=pairwise_lv_dim, eps=eps) # partial: freezes the arguments of the function pairwise_lv_fts 
                                                                                              # to num_outputs=pairwise_lv_dim, eps=eps, 
        self.out_dim = dims[-1] 

        if self.mode == 'concat':
            input_dim = pairwise_lv_dim + pairwise_input_dim
            module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
            for dim in dims:
                module_list.extend([
                    nn.Conv1d(input_dim, dim, 1),
                    nn.BatchNorm1d(dim),
                    nn.GELU() if activation == 'gelu' else nn.ReLU(),
                ])
                input_dim = dim
            if use_pre_activation_pair:
                module_list = module_list[:-1]
            self.embed = nn.Sequential(*module_list)
        elif self.mode == 'sum':
            if pairwise_lv_dim > 0:
                input_dim = pairwise_lv_dim
                module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
                for dim in dims:
                    module_list.extend([
                        nn.Conv1d(input_dim, dim, 1),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == 'gelu' else nn.ReLU(),
                    ])
                    input_dim = dim
                if use_pre_activation_pair:
                    module_list = module_list[:-1]
                self.embed = nn.Sequential(*module_list)
            if pairwise_input_dim > 0:
                input_dim = pairwise_input_dim
                module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
                for dim in dims:
                    module_list.extend([
                        nn.Conv1d(input_dim, dim, 1),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == 'gelu' else nn.ReLU(),
                    ])
                    input_dim = dim
                if use_pre_activation_pair:
                    module_list = module_list[:-1]
                self.fts_embed = nn.Sequential(*module_list)
        else:
            raise RuntimeError('`mode` can only be `sum` or `concat`')


    def forward(self, x, uu=None):
        # x: (batch, v_dim, seq_len) with v_dim = num_features for the input to the interaction terms. The default is 4 for [px, py, pz, E]
        # uu: (batch, v_dim, seq_len, seq_len)
        assert (x is not None or uu is not None)
        #print(f'forward  PairEmbed: x.shape: {x.shape}')
        with torch.no_grad():
            if x is not None:
                batch_size, _, seq_len = x.size()
            else: # at least one of x, uu is not None
                batch_size, _, seq_len, _ = uu.size()
            if self.is_symmetric:
                # create the pairs of indices for the graph
                # For Transformer, we have a fully connected graph -> we need to create all pairs of indices
                # Careful to not double-count. 
                i, j = torch.tril_indices(seq_len, seq_len, offset=-1 if self.remove_self_pair else 0,
                                          device=(x if x is not None else uu).device)
         #       print(f'i.shape: {i.shape}')
          #      print(f'j.shape: {j.shape}')
                if x is not None:
                    x = x.unsqueeze(-1).repeat(1, 1, 1, seq_len) # (batch, v_dim, seq_len, seq_len) 
           #         print(f'x.shape: {x.shape}')
                    # print some values for x to see what it looks like
                    xi = x[:, :, i, j]  # (batch, v_dim, seq_len*(seq_len+-1)/2) +: if we include self-pairs, -: otherwise
                    xj = x[:, :, j, i]
                    x = self.pairwise_lv_fts(xi, xj)
            #        print(f'xi.shape: {xi.shape}')
             #       print(f'xj.shape: {xj.shape}')
              #      print(f'x.shape: {x.shape}')
                if uu is not None:  
                    # (batch, v_dim, seq_len*(seq_len+1)/2)
                    uu = uu[:, :, i, j]
            else:
                if x is not None:
                    x = self.pairwise_lv_fts(x.unsqueeze(-1), x.unsqueeze(-2))
                    if self.remove_self_pair:
                        i = torch.arange(0, seq_len, device=x.device)
                        x_new = x.clone()
                        x_new[:, :, i, i] = 0
                        x = x_new 
                        #x[:, :, i, i] = 0
                    x = x.view(-1, self.pairwise_lv_dim, seq_len * seq_len)
                if uu is not None:
                    uu = uu.view(-1, self.pairwise_input_dim, seq_len * seq_len)
            if self.mode == 'concat':
                if x is None:
                    pair_fts = uu
                elif uu is None:
                    pair_fts = x
                else:
                    pair_fts = torch.cat((x, uu), dim=1)
        #print(f'forward  PairEmbed: x.shape: {x.shape}')
        if self.mode == 'concat':
            elements = self.embed(pair_fts)  # (batch, embed_dim, num_elements)
        elif self.mode == 'sum':
            if x is None:
                elements = self.fts_embed(uu)
            elif uu is None:
                elements = self.embed(x)
            else:
                elements = self.embed(x) + self.fts_embed(uu)

        if self.is_symmetric:
            y = torch.zeros(batch_size, self.out_dim, seq_len, seq_len, dtype=elements.dtype, device=elements.device)
            y[:, :, i, j] = elements
            y[:, :, j, i] = elements
        else:
            y = elements.view(-1, self.out_dim, seq_len, seq_len)
        return y


class Block(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, ffn_ratio=4,
                 dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                 add_bias_kv=False, activation='gelu',
                 scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True, need_weights=False):
        super().__init__()
        
        self.need_weights = need_weights
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ffn_dim = embed_dim * ffn_ratio

        self.pre_attn_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attn_dropout,
            add_bias_kv=add_bias_kv,
        )
        self.post_attn_norm = nn.LayerNorm(embed_dim) if scale_attn else None
        self.dropout = nn.Dropout(dropout)

        self.pre_fc_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, self.ffn_dim)
        self.act = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.act_dropout = nn.Dropout(activation_dropout)
        self.post_fc_norm = nn.LayerNorm(self.ffn_dim) if scale_fc else None
        self.fc2 = nn.Linear(self.ffn_dim, embed_dim)

        self.c_attn = nn.Parameter(torch.ones(num_heads), requires_grad=True) if scale_heads else None
        self.w_resid = nn.Parameter(torch.ones(embed_dim), requires_grad=True) if scale_resids else None

    def forward(self, x, padding_mask=None, attn_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, seq_len)` where padding
                elements are indicated by ``1``.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """


        residual = x
        x = self.pre_attn_norm(x)
        x = self.attn(x, x, x, key_padding_mask=padding_mask, attn_mask=attn_mask, need_weights=self.need_weights)  # (seq_len, batch, embed_dim). By default it returnes the attention weights as well 
                                                                                                        # with need_weights=False, we only get the output of the attention layer, major speedup
                                                                                                        # with need_weights=True, we get the attention weights as well as a 2nd output
        # DA: Pytorch throws a warning here which I suspect is relevant to: https://github.com/pytorch/pytorch/issues/95702 
        # It looks like a bug of pytorch 2.x 
        # TODO: Address this or hope that it's resolved in the next version of pytorch
        warnings.filterwarnings("ignore", message="Support for mismatched key_padding_mask and attn_mask is deprecated.*")

        # Ensure `x` is a tensor (handle tuple output)
        if isinstance(x, tuple):
            x = x[0]
            
        if self.c_attn is not None:
            tgt_len = x.size(0)
            x = x.view(tgt_len, -1, self.num_heads, self.head_dim)
            x = torch.einsum('tbhd,h->tbdh', x, self.c_attn)
            x = x.reshape(tgt_len, -1, self.embed_dim)
        if self.post_attn_norm is not None:
            x = self.post_attn_norm(x)
        x = self.dropout(x)
        #x += residual
        x = x + residual

        residual = x
        x = self.pre_fc_norm(x)
        x = self.act(self.fc1(x))
        x = self.act_dropout(x)
        if self.post_fc_norm is not None:
            x = self.post_fc_norm(x)
        x = self.fc2(x)
        x = self.dropout(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        #x += residual
        x = x + residual

        return x


class Encoder(nn.Module):

    def __init__(self,
                 input_dim,
                 # network configurations
                 pair_input_dim=4, # the default is [lnΔ, lnk_t, lnz, lnm^2] for each pair of particles 
                 pair_extra_dim=0, # ?
                 remove_self_pair=False,
                 use_pre_activation_pair=True,
                 embed_dims=[32, 32],# 64, #[8,], #[128, 512, 128],   # the MLP for transforming the particle features input 
                 pair_embed_dims=[32, 32,], #64], # the MPL for transforming the pairwise features input, i.e. interactions. Note that later we add
                                               # one more layers to this to match the number of heads in the attention layer.
                 num_heads=2,  # how many attention heads in each particle attention block
                 num_layers=2, # how many particle attention blocks
                 block_params=None,
                 activation='gelu',
                 # misc
                 trim=False,  
                 for_inference=False,
                 use_amp=False,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference) # Since we do not provide a mask, what this does is to permute the input (x, v, graph)
                                                                           # which in principle leads to better generalization.
        self.for_inference = for_inference
        self.use_amp = use_amp
        self.num_heads = num_heads

        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else input_dim
        default_cfg = dict(embed_dim=embed_dim, num_heads=num_heads, ffn_ratio=4,
                           dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                           add_bias_kv=False, activation=activation,
                           scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True)

        cfg_block = copy.deepcopy(default_cfg)
        if block_params is not None:
            cfg_block.update(block_params)

        self.pair_extra_dim = pair_extra_dim

        # Embed the particle features before passing them to the attention layers
        self.embed = Embed(input_dim, embed_dims, activation=activation) if len(embed_dims) > 0 else nn.Identity()
        
        # self.pair_embed is only used if we want "interaction terms" between pairs of particles. These act as bias in the attention layer.
        # It is the MLP that transforms the interactions before passing it to the attention layers 
        # The final embedding dim for the pair_embed is the same as the number of heads in the attention layer. 
        # Each head has only one bias feature for all pairs of particles.
        self.pair_embed = PairEmbed( 
            pair_input_dim, pair_extra_dim, pair_embed_dims + [cfg_block['num_heads']],
            remove_self_pair=remove_self_pair, use_pre_activation_pair=use_pre_activation_pair,) if pair_embed_dims is not None and pair_input_dim + pair_extra_dim > 0 else None
        
        self.pair_embed = None

        self.blocks = nn.ModuleList([Block(**cfg_block) for _ in range(num_layers)])
        #self.blocks = None 
        self.proj_to_2d = nn.Linear(embed_dim, 2)
        #self.proj_to_3d = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 3))
        #self.proj_to_2d = None

    def forward(self, x, v=None, mask=None, uu=None, uu_idx=None, graph = None, pr=False):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy] from which we construct the interaction terms between pairs of particles
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # for pytorch: uu (N, C', num_pairs), uu_idx (N, 2, num_pairs): Sparce format for indexing the pairs  
        st = x # save the input for later use
        batch_size, _, num_particles = x.size()
        if pr:
            print(f'Encoder input shape: {x.shape}')
            print(f'x[:2,:,:5]: {x[:2,:,:5]}')
            print()
        
        with torch.no_grad():
            if not self.for_inference: # if training 
                if uu_idx is not None:
                    uu = build_sparse_tensor(uu, uu_idx, x.size(-1)) # returns: (N, C', P, P)
            x, v, mask, uu, graph = self.trimmer(x, v, mask, uu, graph)            # 
            padding_mask = ~mask.squeeze(1)                          # (N, P) and padded = 1, real particle = 0 now due to ~ (bitwise not)
            padding_mask = padding_mask.float().masked_fill(padding_mask, float('-inf')) # (N, P) and padded = -inf, real particle = 0
        if pr: 
            print(f'Trimmed input')
            print(f'x.shape: {x.shape}')
            print(f'v.shape: {v.shape}')
            print(f'mask.shape: {mask.shape}')
            print(f'uu.shape: {uu.shape if uu is not None else None}')
            print(f'x[:2,:,:5]: {x[:2,:,:5]}')
            print()

        with torch.cuda.amp.autocast(enabled=self.use_amp): # if true it lowers the precision of some computations to half precision for faster computation
                                                            # The default for self.use_amp = False

            # input embedding. Will change x.shape to x: (seq_len=num_particles in a jet, batch_size, embed_dim)
            if isinstance(self.embed, nn.Identity):
                #print(f'No embedding')
                #print(f'x.shape: {x.shape}')
                x = x.permute(2, 0, 1)
                #print(f'x.shape: {x.shape}')
            else:
                x = self.embed(x)#.masked_fill(~mask.permute(2, 0, 1), 0)  # (P, N, C)  # masked_fill: fill the elements of x with 0 where mask is False
                                                                      # mask.permute(2, 0, 1) -> (P, N, 1)

            #print(f'mask.shape: {mask.shape}')
            m_aux = mask.permute(2, 0, 1)
            #print(f'm_aux.shape: {m_aux.shape}')
            x = x.masked_fill(~mask.permute(2, 0, 1), 0)  # (P, N, C)
            # pair embedding to get the interaction terms between pairs of particles -> Acts as the bias in the attention layer of the particle attn block.
            if pr:
                print(f'Embedded x')
                print(f'mask.shape: {mask.shape}')
                print(f'mask[:3, :5, :5]: {mask[:3, :5, :5]}')
                print()
                print(f'x.shape: {x.shape}')
                print(f'x[:4,:2,:5]: {x[:4,2,:5]}')
                print()

            attn_mask = None
            if  (v is not None or uu is not None) and self.pair_embed is not None:
                attn_mask = self.pair_embed(v, uu).view(-1, v.size(-1), v.size(-1))  # (N*num_heads, P, P)
                if pr:
                    print(f'attn_mask.shape: {attn_mask.shape}')
                    print(f'attn_mask[0, :5, :5]: {attn_mask[0, :5, :5]}')
                    print()

            # filter the attn_mask with the graph that was constructed in models.ParticleTransformer. Otherwise, full transformer is used.
            if graph is not None and self.pair_embed is not None:
                bool_mask =  graph.unsqueeze(1).repeat(1, self.num_heads, 1, 1).reshape(batch_size*self.num_heads, 
                                                                                        num_particles, num_particles).to(attn_mask.device)
                   
                attn_mask = torch.where(bool_mask, attn_mask, torch.tensor(0).to(attn_mask.dtype).to(attn_mask.device))


            # transform
            if self.blocks is not None:
                for block in self.blocks:
                    if pr:
                        print(f'x.shape: {x.shape}')
                        print(f'x[:2, :5, :5]: {x[0, :5, :5]}')
                        print()
                    x = block(x, padding_mask=padding_mask, attn_mask = attn_mask)
                
            if pr:
                print(f'x.shape: {x.shape}')
                print(f'x[0, :5, :5]: {x[0, :5, :5]}')
                print()

            x = self.proj_to_2d(x) # should we permute the dimensions here? check the dims. Check the padded particles if they are zeroed out.

            #x = self.proj_to_3d(x)

            # reshape x to (batch_size, 2, num_particles)
            x = x.permute(1, 2, 0) 
            if pr:
                print(f'x.shape: {x.shape}')
                print(f'x[0, :, :5]: {x[0, :, :5]}')
                print()
                time.sleep(3)
            return x


class Decoder(nn.Module):
    def __init__(self, 
                 embed_dims=[32, 32],# 64], #[128, 512, 128],   # the MLP for transforming the particle features from 2D (the output of the encoder)  
                 num_heads=2,  # how many attention heads in each particle attention block
                 num_layers=2, # how many particle attention blocks
                 activation='gelu',
                 use_amp=False,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_amp = use_amp

        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else 2
        default_cfg = dict(embed_dim=embed_dim, num_heads=num_heads, ffn_ratio=4,
                           dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                           add_bias_kv=False, activation=activation,
                           scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True)

        self.embed = Embed(2, embed_dims, activation=activation) if len(embed_dims) > 0 else nn.Identity() 
        self.blocks = nn.ModuleList([Block(**default_cfg) for _ in range(num_layers)])
        #self.blocks = None
        self.proj_to_3d = nn.Linear(embed_dim, 3)


    def forward(self, x, pr=False):
        # reshape x to (batch_size, 2, num_particles)
        #x = x.permute(1, 2, 0) 
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # check if we need to mask the padded particles here.
            x = self.embed(x)
            if self.blocks is not None:
                for block in self.blocks:
                    x = block(x)
            x = self.proj_to_3d(x)
            x = x.permute(1, 0, 2)
            return x


class TAE(nn.Module):
    def __init__(self, encoder_cfg, decoder_cfg, **kwargs):
        super().__init__(**kwargs)
        self.encoder = Encoder(**encoder_cfg)
        self.decoder = Decoder(**decoder_cfg)

    def forward(self, x, v=None, mask=None, uu=None, uu_idx=None, graph=None, pr=False):
        x = self.encoder(x, v, mask, uu, uu_idx, graph, pr)
        #print(f'after encoder')
        #print(f'x.shape: {x.shape}')
        #x = x.permute(0, 2, 1)
        x = self.decoder(x, pr)
        #print(f'after decoder')
        #print(f'x.shape: {x.shape}')
        #time.sleep(2)
        return x