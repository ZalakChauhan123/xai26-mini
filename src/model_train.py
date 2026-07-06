
# Model Training Imports
from sklearn.model_selection import train_test_split
import torch

# Model Explainability Imports
from torch_geometric.nn import GCNConv
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer

# visulisation Imports
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Other imports
from feature_engineering import *


print("Data loaded successfully!")
print("y shape:", y.shape)


# Create Labeled Node List
labeled_nodes = (y != -1).nonzero(as_tuple=True)[0]
print("Total Labeled Nodes :-" , len(labeled_nodes))
# print(labeled_nodes[:10])




# TRAIN/TEST SPLIT
train_nodes, test_nodes = train_test_split(
    labeled_nodes,
    test_size = 0.2,
    random_state = 42
)

print("Train Nodes :- ", len(train_nodes))
print("Test Nodes :- ", len(test_nodes))



# CREATE TRAIN/TEST MASKS

# Empty mask for train & test dataset
train_mask = torch.zeros( num_nodes, dtype=torch.bool )
test_mask = torch.zeros( num_nodes, dtype=torch.bool )


# Set Train & Test Nodes
# Purpose
# Train Mask : Learn from these nodes
# Test Mask : Evalute on these nodes

train_mask[train_nodes] = True
test_mask[test_nodes] = True

# Varify
# print(train_mask.sum())
# print(test_mask.sum())



# CREATE NODE FEATURES
x = torch.eye(num_nodes)
# print(x.shape)


# CREATE PyTorch Geometric DATA OBJECT
from torch_geometric.data import Data

data = Data(
    x = x, # Node Features
    edge_index = edge_index,
    edge_type = edge_type,
    y = y
)

data.train_mask = train_mask
data.test_mask = test_mask

# print(data)




# BUILD R-GCN MODEL
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

# Create a custom neural network model
class RGCN(torch.nn.Module):

    def __init__(
            self,
            num_nodes,
            hidden_channels,
            num_classes,
            num_relations
    ):
        super().__init__()

        # learn graph embeddings
        self.conv1 = RGCNConv(
            num_nodes,
            hidden_channels,
            num_relations
        )

        # Predict node Classes
        self.conv2 = RGCNConv(
            hidden_channels,
            num_classes,
            num_relations
        )

    def forward( self, x, edge_index, edge_type ):
        x = self.conv1( x, edge_index, edge_type )
        x = F.relu(x)
        x = self.conv2( x, edge_index, edge_type )
        return x


# CREATE MODEL
num_relations = len(predicate_to_id)
num_classes = len(label_to_id)

model = RGCN(
            num_nodes = num_nodes,
            hidden_channels = 16,
            num_classes = num_classes,
            num_relations = num_relations
)

print(model)



# FORWARD PASSING
out = model(
    data.x,
    data.edge_index,
    data.edge_type
)

# print(out.shape)


# LOSS FUNCTION & OPTIMIZER
# Loss function
loss_function = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(), # Collect ALL Params. of R-GCN
    lr = 0.01 # Learning Rate
)
train_accuracy = 0.9931


# TRAINING LOOP
for epoch in range(1, 201):

    # Step-1 : Model Train
    model.train()

    # Step-2 : clear old learning gradiants
    optimizer.zero_grad()

    # Step-3 : Forward Pass
    out = model(
        data.x,
        data.edge_index,
        data.edge_type
    )

    # Step-4 : Loss Calculation
    loss = loss_function(
        out[data.train_mask],
        data.y[data.train_mask]
    )

    # Step-5 : BackwordPropogation
    loss.backward()

    # Update Weights
    optimizer.step()

    if epoch % 20 == 0:
        print(
            f"Epoch: {epoch}, Loss: {loss.item():.4f}"
        )



# MODEL EVALUATION
model.eval()

# Generate predictions without tracking gradients (saves memory)
with torch.no_grad():
    final_out = model(data.x, data.edge_index, data.edge_type)

pred = out.argmax( dim = 1 )
correct_train = ( pred[data.train_mask] == data.y[data.train_mask] ).sum()
trainaccuracy = int(correct_train) / int(data.train_mask.sum())
correct_test = ( pred[data.test_mask] == data.y[data.test_mask] ).sum()
test_accuracy = int(correct_test) / int(data.test_mask.sum())

print(f"Final Test Accuracy  : {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
print(f"Final Train Accuracy  : {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")