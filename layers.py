import torch
import torch.nn as nn
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter

from utils import MLP, Res, MessagePassing

class Global_MP(MessagePassing):

    def __init__(self, config):
        super(Global_MP, self).__init__()
        self.dim = config.dim

        self.h_mlp = MLP([self.dim, self.dim])

        self.res1 = Res(self.dim)
        self.res2 = Res(self.dim)
        self.res3 = Res(self.dim)
        self.mlp = MLP([self.dim, self.dim])

        self.x_edge_mlp = MLP([self.dim * 3, self.dim])
        self.linear = nn.Linear(self.dim, self.dim, bias=False)

    def forward(self, h, edge_attr, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=h.size(0))

        res_h = h
        
        # Integrate the Cross Layer Mapping inside the Global Message Passing
        h = self.h_mlp(h)
        
        # Message Passing operation
        h = self.propagate(edge_index, x=h, num_nodes=h.size(0), edge_attr=edge_attr)

        # Update function f_u
        h = self.res1(h)
        h = self.mlp(h) + res_h
        h = self.res2(h)
        h = self.res3(h)
        
        # Message Passing operation
        h = self.propagate(edge_index, x=h, num_nodes=h.size(0), edge_attr=edge_attr)

        return h

    def message(self, x_i, x_j, edge_attr, edge_index, num_nodes):
        num_edge = edge_attr.size()[0]

        x_edge = torch.cat((x_i[:num_edge], x_j[:num_edge], edge_attr), -1)
        x_edge = self.x_edge_mlp(x_edge)

        x_j = torch.cat((self.linear(edge_attr) * x_edge, x_j[num_edge:]), dim=0)

        return x_j

    def update(self, aggr_out):

        return aggr_out


class Local_MP(torch.nn.Module):
    def __init__(self, config):
        super(Local_MP, self).__init__()
        self.dim = config.dim

        self.h_mlp = MLP([self.dim, self.dim])

        self.mlp_kj = MLP([3 * self.dim, self.dim])
        self.mlp_ji_1 = MLP([3 * self.dim, self.dim])
        self.mlp_ji_2 = MLP([self.dim, self.dim])
        self.mlp_jj = MLP([self.dim, self.dim])

        self.mlp_sbf1 = MLP([self.dim, self.dim, self.dim])
        self.mlp_sbf2 = MLP([self.dim, self.dim, self.dim])
        self.lin_rbf1 = nn.Linear(self.dim, self.dim, bias=False)
        self.lin_rbf2 = nn.Linear(self.dim, self.dim, bias=False)

        self.res1 = Res(self.dim)
        self.res2 = Res(self.dim)
        self.res3 = Res(self.dim)

        self.lin_rbf_out = nn.Linear(self.dim, self.dim, bias=False)

        self.h_mlp = MLP([self.dim, self.dim])

        self.y_mlp = MLP([self.dim, self.dim, self.dim, self.dim])
        self.y_W = nn.Linear(self.dim, 1)

    def forward(self, h, rbf, sbf1, sbf2, idx_kj, idx_ji_1, idx_jj, idx_ji_2, edge_index, num_nodes=None):
        res_h = h
        
        # Integrate the Cross Layer Mapping inside the Local Message Passing
        h = self.h_mlp(h)
        
        # Message Passing 1
        j, i = edge_index
        m = torch.cat([h[i], h[j], rbf], dim=-1)

        m_kj = self.mlp_kj(m)
        m_kj = m_kj * self.lin_rbf1(rbf)
        m_kj = m_kj[idx_kj] * self.mlp_sbf1(sbf1)
        m_kj = scatter(m_kj, idx_ji_1, dim=0, dim_size=m.size(0), reduce='add')
        
        m_ji_1 = self.mlp_ji_1(m)

        m = m_ji_1 + m_kj

        # Message Passing 2       (index jj denotes j'i in the main paper)
        m_jj = self.mlp_jj(m)
        m_jj = m_jj * self.lin_rbf2(rbf)
        m_jj = m_jj[idx_jj] * self.mlp_sbf2(sbf2)
        m_jj = scatter(m_jj, idx_ji_2, dim=0, dim_size=m.size(0), reduce='add')
        
        m_ji_2 = self.mlp_ji_2(m)

        m = m_ji_2 + m_jj

        # Aggregation
        m = self.lin_rbf_out(rbf) * m
        h = scatter(m, i, dim=0, dim_size=h.size(0), reduce='add')
        
        # Update function f_u
        h = self.res1(h)
        h = self.h_mlp(h) + res_h
        h = self.res2(h)
        h = self.res3(h)

        # Output Module
        y = self.y_mlp(h)
        y = self.y_W(y)

        return h, y
