import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, radius
from torch_geometric.utils import remove_self_loops
from torch_scatter import scatter
from torch_sparse import SparseTensor

from layers import AuxiliaryLayer, Global_MP, Local_MP
from utils import DAGNN, MLP, BesselBasisLayer, SphericalBasisLayer


class Config(object):
    def __init__(self, dim, n_layer, cutoff, virtual_node, auxiliary_layer, dagnn):
        self.dim = dim
        self.n_layer = n_layer
        self.cutoff = cutoff
        self.virtual_node = virtual_node
        self.auxiliary_layer = auxiliary_layer
        self.dagnn = dagnn


class MXMNet(nn.Module):
    def __init__(
        self, config: Config, num_spherical=7, num_radial=6, envelope_exponent=5
    ):
        super(MXMNet, self).__init__()

        self.dim = config.dim
        self.n_layer = config.n_layer
        self.cutoff = config.cutoff
        self.virtual_node_enabled = config.virtual_node
        self.auxiliary_layer_enabled = config.auxiliary_layer
        self.dagnn_enabled = config.dagnn

        self.embeddings = nn.Parameter(torch.ones((5, self.dim)))

        self.rbf_l = BesselBasisLayer(16, 5, envelope_exponent)
        self.rbf_g = BesselBasisLayer(16, self.cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, 5, envelope_exponent)

        self.rbf_g_mlp = MLP([16, self.dim])
        self.rbf_l_mlp = MLP([16, self.dim])

        self.sbf_1_mlp = MLP([num_spherical * num_radial, self.dim])
        self.sbf_2_mlp = MLP([num_spherical * num_radial, self.dim])

        if not self.virtual_node_enabled:
            self.global_layers = nn.ModuleList()
            for layer in range(config.n_layer):
                self.global_layers.append(Global_MP(config))

        self.local_layers = nn.ModuleList()
        for layer in range(config.n_layer):
            self.local_layers.append(Local_MP(config))

        if self.virtual_node_enabled:
            self.virtualnode_embedding = nn.Embedding(1, self.dim)
            self.mlp_virtualnode_layers = nn.ModuleList()
            for layer in range(config.n_layer):
                self.mlp_virtualnode_layers.append(
                    nn.Sequential(
                        nn.Linear(self.dim, self.dim),
                        nn.Sigmoid(),
                        nn.Linear(self.dim, self.dim),
                        nn.Sigmoid(),
                    )
                )

        if self.auxiliary_layer_enabled:
            self.auxiliary_layer = AuxiliaryLayer(self.dim)
        if self.dagnn_enabled:
            self.dagnn = DAGNN(5, self.dim)
        if self.auxiliary_layer_enabled or self.dagnn_enabled:
            self.graph_pred_linear = torch.nn.Linear(self.dim, 1)
        self.pool = global_add_pool
        self.init()

    def init(self):
        stdv = math.sqrt(3)
        self.embeddings.data.uniform_(-stdv, stdv)
        if self.virtual_node_enabled:
            nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

    def indices(self, edge_index, num_nodes):
        row, col = edge_index

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(
            row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes)
        )

        # Compute the node indices for two-hop angles
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k
        idx_i_1, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji_1 = adj_t_row.storage.row()[mask]

        # Compute the node indices for one-hop angles
        adj_t_col = adj_t[col]

        num_pairs = adj_t_col.set_value(None).sum(dim=1).to(torch.long)
        idx_i_2 = row.repeat_interleave(num_pairs)
        idx_j1 = col.repeat_interleave(num_pairs)
        idx_j2 = adj_t_col.storage.col()

        idx_ji_2 = adj_t_col.storage.row()
        idx_jj = adj_t_col.storage.value()

        return (
            idx_i_1,
            idx_j,
            idx_k,
            idx_kj,
            idx_ji_1,
            idx_i_2,
            idx_j1,
            idx_j2,
            idx_jj,
            idx_ji_2,
        )

    def forward(self, data):
        # x = data.x
        x = torch.argmax(data.x[:, :5], dim=1)
        edge_index = data.edge_index
        pos = data.pos
        batch = data.batch
        # Initialize node embeddings
        h = torch.index_select(self.embeddings, 0, x.long())

        # Get the edges and pairwise distances in the local layer
        edge_index_l, _ = remove_self_loops(edge_index)
        j_l, i_l = edge_index_l
        dist_l = (pos[i_l] - pos[j_l]).pow(2).sum(dim=-1).sqrt()

        # Get the edges pairwise distances in the global layer
        row, col = radius(pos, pos, self.cutoff, batch, batch, max_num_neighbors=500)
        edge_index_g = torch.stack([row, col], dim=0)
        edge_index_g, _ = remove_self_loops(edge_index_g)
        j_g, i_g = edge_index_g
        dist_g = (pos[i_g] - pos[j_g]).pow(2).sum(dim=-1).sqrt()

        # Compute the node indices for defining the angles
        (
            idx_i_1,
            idx_j,
            idx_k,
            idx_kj,
            idx_ji,
            idx_i_2,
            idx_j1,
            idx_j2,
            idx_jj,
            idx_ji_2,
        ) = self.indices(edge_index_l, num_nodes=h.size(0))

        # Compute the two-hop angles
        pos_ji_1, pos_kj = pos[idx_j] - pos[idx_i_1], pos[idx_k] - pos[idx_j]
        a = (pos_ji_1 * pos_kj).sum(dim=-1)
        b = torch.cross(pos_ji_1, pos_kj).norm(dim=-1)
        angle_1 = torch.atan2(b, a)

        # Compute the one-hop angles
        pos_ji_2, pos_jj = pos[idx_j1] - pos[idx_i_2], pos[idx_j2] - pos[idx_j1]
        a = (pos_ji_2 * pos_jj).sum(dim=-1)
        b = torch.cross(pos_ji_2, pos_jj).norm(dim=-1)
        angle_2 = torch.atan2(b, a)

        # Get the RBF and SBF embeddings
        rbf_g = self.rbf_g(dist_g)
        rbf_l = self.rbf_l(dist_l)
        sbf_1 = self.sbf(dist_l, angle_1, idx_kj)
        sbf_2 = self.sbf(dist_l, angle_2, idx_jj)

        rbf_g = self.rbf_g_mlp(rbf_g)
        rbf_l = self.rbf_l_mlp(rbf_l)
        sbf_1 = self.sbf_1_mlp(sbf_1)
        sbf_2 = self.sbf_2_mlp(sbf_2)

        # Perform the message passing schemes
        if self.virtual_node_enabled:
            virtualnode_embedding = self.virtualnode_embedding(
                torch.zeros(batch[-1].item() + 1)
                .to(edge_index.dtype)
                .to(edge_index.device)
            )
        node_sum = 0
        for layer in range(self.n_layer):
            # Do not add global layer if Virtual Node is present
            if not self.virtual_node_enabled:
                # Message passing through global layers
                h = self.global_layers[layer](h, rbf_g, edge_index_g)
            # Message passing through local layers
            h, t = self.local_layers[layer](
                h, rbf_l, sbf_1, sbf_2, idx_kj, idx_ji, idx_jj, idx_ji_2, edge_index_l
            )
            node_sum += t
            # Pool information from graph to virtual node and transform
            if self.virtual_node_enabled:
                if layer < self.n_layer:
                    virtualnode_embedding_temp = (
                        global_add_pool(h, batch) + virtualnode_embedding
                    )
                    virtualnode_embedding = self.mlp_virtualnode_layers[layer](
                        virtualnode_embedding_temp
                    )
                    h = h + virtualnode_embedding[batch]
        # Add Auxiliary Information
        if self.auxiliary_layer_enabled:
            a_h = self.auxiliary_layer(data)
            a_h = self.pool(a_h, batch)
        # Readout
        if self.dagnn_enabled:
            h = self.dagnn(h, edge_index)
            h = self.pool(h, data.batch)
            if self.auxiliary_layer_enabled:
                h = h + a_h
            output = self.graph_pred_linear(h)
        else:
            if self.auxiliary_layer_enabled:
                h = self.pool(h, batch)
                h = h + a_h
                output = self.graph_pred_linear(h)
            else:
                output = self.pool(node_sum, batch)
        return output.view(-1)
