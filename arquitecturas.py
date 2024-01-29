
import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN_global(nn.Module):
    def __init__(self, dim, edge_index, num_layers, K, num_nodes, num_gens):
        super(GNN_global, self).__init__()

        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.num_nodes = num_nodes
        self.num_gens = num_gens
        self.dim = dim
        
        for layer in range(num_layers-1):
          self.convs.append(TAGConv(dim[layer],dim[layer+1],K[layer],bias=True))
        
        self.fcnn = torch.nn.Linear(dim[-2]*num_nodes,num_gens)

        self.edge_index = edge_index
        self.relu = LeakyReLU()

    def forward(self, x):

        # Apply the GNN to the node features
        num_layers = self.num_layers
        convs = self.convs
        relu = self.relu
        fcnn = self.fcnn
        out = x
        for layer in range(num_layers-1):
          out = convs[layer](out,edge_index)
          out = relu(out)
        out = out.reshape(-1,self.dim[-2]*self.num_nodes)
        out = fcnn(out)
        x = out.squeeze()

        return x

class FCNN_global(nn.Module):

    def __init__(self, dim, num_layers):
        super(FCNN_global, self).__init__()

        self.num_layers = num_layers
        self.linears = torch.nn.ModuleList()
        self.dim = dim
        
        for layer in range(num_layers):
          self.linears.append(nn.Linear(dim[layer],dim[layer+1]))
        self.relu = LeakyReLU()

    def forward(self, x):

        # Apply the GNN to the node features
        num_layers = self.num_layers
        linears = self.linears
        relu = self.relu

        out = x.reshape(-1,self.dim[0])

        for layer in range(num_layers-1):
          out = linears[layer](out)
          out = relu(out)
        
        out = linears[num_layers-1](out)
        return out
