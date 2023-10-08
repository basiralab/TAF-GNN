import torch
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from torch_geometric.nn import NNConv
from torch_geometric.nn import BatchNorm

N_SOURCE_NODES = 35

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")



class Aligner(torch.nn.Module):
    def __init__(self):
        super(Aligner, self).__init__()

        nn = Sequential(Linear(1, N_SOURCE_NODES*N_SOURCE_NODES), ReLU())
        self.conv1 = NNConv(N_SOURCE_NODES, N_SOURCE_NODES, nn, aggr='mean', root_weight=True, bias=True)
        self.conv11 = BatchNorm(N_SOURCE_NODES, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

        nn = Sequential(Linear(1, N_SOURCE_NODES), ReLU())
        self.conv2 = NNConv(N_SOURCE_NODES, 1, nn, aggr='mean', root_weight=True, bias=True)
        self.conv22 = BatchNorm(1, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

        nn = Sequential(Linear(1, N_SOURCE_NODES), ReLU())
        self.conv3 = NNConv(1, N_SOURCE_NODES, nn, aggr='mean', root_weight=True, bias=True)
        self.conv33 = BatchNorm(N_SOURCE_NODES, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.pos_edge_index, data.edge_attr

        x1 = F.sigmoid(self.conv11(self.conv1(x, edge_index, edge_attr)))
        x1 = F.dropout(x1, training=self.training)

        x2 = F.sigmoid(self.conv22(self.conv2(x1, edge_index, edge_attr)))
        x2 = F.dropout(x2, training=self.training)

        x3 = F.sigmoid(self.conv33(self.conv3(x2, edge_index, edge_attr)))

        symmetric_x6 = (x3 + x3.t()) / 2
        symmetric_x6 = symmetric_x6.fill_diagonal_(0)

        return symmetric_x6







