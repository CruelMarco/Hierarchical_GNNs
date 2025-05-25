import torch
from torch_geometric.datasets import TUDataset
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
import torch_geometric.transforms as T


# Dataset loading and preparation
#dataset = TUDataset(root='data/TUDataset', name='PROTEINS')
#torch.manual_seed(12345)
#dataset = dataset.shuffle()


max_nodes = 620

#path = "/Users/shaique/Desktop/BioInf_IMP/Bioinf_WS_2024/HO_GNN/data_sets"


dataset = TUDataset(
    root='data/TUDataset',
    name='PROTEINS',
    #transform=T.ToDense(max_nodes),
    #pre_filter=lambda data: data.num_nodes <= max_nodes,
)

filtered_dataset = [data for data in dataset if data.num_nodes <= max_nodes]

dataset = dataset.shuffle()
n = (len(dataset) + 9) // 10
test_dataset = dataset[:n]
val_dataset = dataset[n:2 * n]
train_dataset = dataset[2 * n:]
#test_loader = DenseDataLoader(test_dataset, batch_size=20)
#val_loader = DenseDataLoader(val_dataset, batch_size=20)
#train_loader = DenseDataLoader(train_dataset, batch_size=20)

# Check the distribution of node counts to decide on max_nodes
node_counts = [data.num_nodes for data in filtered_dataset]
print(f"Maximum nodes in any graph: {max(node_counts)}")
print(f"Minimum nodes in any graph: {min(node_counts)}")



# Splitting dataset into train, validation, and test
#train_dataset = dataset[150:-150]
#val_dataset = dataset[-150:-75]
#test_dataset = dataset[:150]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of validation graphs: {len(val_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

# Data loaders for batching
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = self.lin(x)
        return x

model = GCN(hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test(loader):
    model.eval()
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
            loss = criterion(out, data.y)
            total_loss += loss.item()
    return correct / len(loader.dataset), total_loss / len(loader)

train_loss_history = []
val_loss_history = []

for epoch in range(1, 151):
    train_loss = train()
    train_acc, _ = test(train_loader)
    val_acc, val_loss = test(val_loader)
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
    scheduler.step()

# Test set evaluation only after all training and validation is complete
test_acc, test_loss = test(test_loader)
print(f'Final Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}')

# Plotting training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()
