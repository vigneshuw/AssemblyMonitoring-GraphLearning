import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import Linear, HeteroConv, GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool


class RGCN(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, is_realtime=False):

        super(RGCN, self).__init__()
        self.convs1 = torch.nn.ModuleList()
        self.convs2 = torch.nn.ModuleList()
        self.convs3 = torch.nn.ModuleList()

        self.batch_norms1 = torch.nn.ModuleList()
        self.batch_norms2 = torch.nn.ModuleList()
        self.batch_norms3 = torch.nn.ModuleList()

        self.lin1 = Linear(hidden_channels * 3, hidden_channels * 3)
        self.lin2 = Linear(hidden_channels * 3, out_channels)

        self.dropout = nn.Dropout(p=0.5)

        for _ in range(num_layers):
            conv1 = HeteroConv({('frame_window', 'spatial', 'frame_window'): GCNConv(in_channels, hidden_channels),
                                ('frame_window', 'temporal', 'frame_window'): GCNConv(in_channels, hidden_channels)},
                               aggr='sum')
            conv2 = HeteroConv({('frame_window', 'spatial', 'frame_window'): GCNConv(hidden_channels, hidden_channels),
                                ('frame_window', 'temporal', 'frame_window'): GCNConv(hidden_channels,
                                                                                      hidden_channels)}, aggr='sum')
            conv3 = HeteroConv({('frame_window', 'spatial', 'frame_window'): GCNConv(hidden_channels, hidden_channels),
                                ('frame_window', 'temporal', 'frame_window'): GCNConv(hidden_channels,
                                                                                      hidden_channels)}, aggr='sum')
            bn1 = nn.BatchNorm1d(hidden_channels)
            bn2 = nn.BatchNorm1d(hidden_channels)
            bn3 = nn.BatchNorm1d(hidden_channels)
            self.convs1.append(conv1)
            self.convs2.append(conv2)
            self.convs3.append(conv3)
            self.batch_norms1.append(bn1)
            self.batch_norms2.append(bn2)
            self.batch_norms3.append(bn3)

        # Case of real-time inference
        self.is_realtime = is_realtime

    def forward(self, data):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        edge_weight_dict = data.edge_weight_dict
        if self.is_realtime:
            batch = torch.tensor([1], dtype=torch.long)
        else:
            batch = data['frame_window'].batch

        for conv1, conv2, conv3, bn1, bn2, bn3 in zip(self.convs1, self.convs2, self.convs3,
                                                      self.batch_norms1, self.batch_norms2, self.batch_norms3):
            gc1 = conv1(x_dict, edge_index_dict, edge_weight_dict)
            gc1 = {key: bn1(x.relu()) for key, x in gc1.items()}
            #             gc1 = {key: self.dropout(bn1(x.relu())) for key, x in gc1.items()}
            gc2 = conv2(gc1, edge_index_dict, edge_weight_dict)
            gc2 = {key: bn2(x.relu()) for key, x in gc2.items()}
            #             gc2 = {key: self.dropout(bn2(x.relu())) for key, x in gc2.items()}
            gc3 = conv3(gc2, edge_index_dict, edge_weight_dict)
            gc3 = {key: bn3(x.relu()) for key, x in gc3.items()}
        #             gc3 = {key: self.dropout(bn3(x.relu())) for key, x in gc3.items()}

        out1 = global_mean_pool(gc3['frame_window'], batch)
        out2 = global_add_pool(gc3['frame_window'], batch)
        out3 = global_max_pool(gc3['frame_window'], batch)
        out = torch.cat((out1, out2, out3), dim=1)
        out = F.relu(self.lin1(out))
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.lin2(out)
        return out
