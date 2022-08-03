import argparse
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import NNConv, global_add_pool

NUM_NODE_FEAT = 11
NUM_EDGE_FEAT = 4


class MPNN(torch.nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        num_node_features = NUM_NODE_FEAT
        num_edge_features = NUM_EDGE_FEAT
        hidden_dim = dim // 2
        conv1_net = nn.Sequential(
            nn.Linear(num_edge_features, dim),
            nn.ReLU(),
            nn.Linear(dim, num_node_features * dim),
        )
        conv2_net = nn.Sequential(
            nn.Linear(num_edge_features, dim), nn.ReLU(), nn.Linear(dim, dim * dim)
        )
        self.conv1 = NNConv(num_node_features, dim, conv1_net)
        self.conv2 = NNConv(dim, dim, conv2_net)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Parameters
        ----------
        data
            Torch Geometric Data object with attributes: batch, x, edge_index, edge_attr

        Returns
        -------
        torch.Tensor
            of dimensions (B, 1)
        """
        batch, x, edge_index, edge_attr = (
            data.batch,
            data.x,
            data.edge_index,
            data.edge_attr,
        )
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        # x = global_add_pool(x, batch)
        # x = F.relu(self.fc_1(x))
        return x
