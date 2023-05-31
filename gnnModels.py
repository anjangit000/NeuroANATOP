import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, ChebConv, SAGEConv
from torch_geometric.data import InMemoryDataset, Data
import networkx as nx
import numpy as np

class MyDataset(InMemoryDataset):
	def __init__(self, nx_G, vecLen, transform=None):
		super(MyDataset, self).__init__('.', transform, None, None)
		adj = nx.to_scipy_sparse_matrix(nx_G).tocoo()
		row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
		col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
		edge_index = torch.stack([row, col], dim=0)
		data = Data(edge_index=edge_index)
		data.num_nodes = edge_index.max().item() + 1
		#data.x = torch.eye(data.num_nodes, dtype=torch.float)
		data.x = torch.rand(data.num_nodes, 1000)
		y = torch.randint(0, vecLen, (data.num_nodes,))
		train_mask = [True for i in nx_G.nodes]	
		test_mask = [True for i in nx_G.nodes]
		data.y = torch.tensor(y)
		data.train_mask = torch.tensor(train_mask)
		data.test_mask = torch.tensor(test_mask)
		self.data, self.slices = self.collate([data])

	def _download(self):
		return

	def _process(self):
		return

	def __repr__(self):
		return '{}()'.format(self.__class__.__name__)

class GCN(torch.nn.Module):
    def __init__(self, dataset, args):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class SAGE_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(SAGE_Net, self).__init__()
        self.conv1 = SAGEConv(dataset.num_features, args.hidden)
        self.conv2 = SAGEConv(args.hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class ChebNet(torch.nn.Module):
    def __init__(self, dataset, args):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(dataset.num_features, 32, K=2)
        self.conv2 = ChebConv(32, dataset.num_classes, K=2)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GAT_Net, self).__init__()
        self.conv1 = GATConv(
            dataset.num_features,
            args.hidden,
            heads=args.heads,
            dropout=args.dropout)
        self.conv2 = GATConv(
            args.hidden * args.heads,
            dataset.num_classes,
            heads=args.output_heads,
            concat=False,
            dropout=args.dropout)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)