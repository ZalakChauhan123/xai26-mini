#!/usr/bin/env python
# coding: utf-8
#
# 3__model_train.py
# -----------------------------------------------------------------------------
# Trains a Relational Graph Convolutional Network (R-GCN) on the AIFB graph to
# predict a researcher's affiliation (research group).
#
# Converted from notebooks/3__model_train.ipynb.  The notebook loaded the
# pre-processing state with the Jupyter magic  `%run "2__Graph_PreProcessing.ipynb"`;
# in a plain script we import the same variables from the src library instead.
# The explainer exploration that trailed the notebook now lives in its own
# stages (4__Explainability.py / 5__Explainability_Captus.py), mirroring the
# src/model_explainability*.py modules.
# -----------------------------------------------------------------------------

# Model Training Imports
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv

# Load the pre-processed graph tensors (node_to_id, edge_index, edge_type,
# predicate_to_id, label_to_id, y, num_nodes, ...) from the src library.
# `model.sh` puts <repo>/src on PYTHONPATH; add it here too so the script also
# runs standalone.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from feature_engineering import *


print("Data loaded successfully!")
print("y shape:", y.shape)


# Create Labeled Node List
# Training only makes sense on nodes that actually carry a label.
labeled_nodes = (y != -1).nonzero(as_tuple=True)[0]
print("Total Labeled Nodes :-", len(labeled_nodes))
print(labeled_nodes[:10])


# TRAIN/TEST SPLIT
train_nodes, test_nodes = train_test_split(
    labeled_nodes,
    test_size = 0.2,
    random_state = 42
)

print("Train Nodes :- ", len(train_nodes))
print("Test Nodes :- ", len(test_nodes))


# CREATE TRAIN/TEST MASKS
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[train_nodes] = True
test_mask[test_nodes] = True

print("Train mask nodes :", int(train_mask.sum()))
print("Test mask nodes  :", int(test_mask.sum()))


# CREATE NODE FEATURES
# Each node gets a unique one-hot vector (identity matrix).
x = torch.eye(num_nodes)
print("Feature matrix shape :", x.shape)


# CREATE PyTorch Geometric DATA OBJECT
data = Data(
    x = x,                       # Node Features
    edge_index = edge_index,
    edge_type = edge_type,
    y = y
)

data.train_mask = train_mask
data.test_mask = test_mask

print(data)


# BUILD R-GCN MODEL
class RGCN(torch.nn.Module):

    def __init__(self, num_nodes, hidden_channels, num_classes, num_relations):
        super().__init__()

        # learn graph embeddings
        self.conv1 = RGCNConv(num_nodes, hidden_channels, num_relations)

        # Predict node Classes
        self.conv2 = RGCNConv(hidden_channels, num_classes, num_relations)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type)
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


# LOSS FUNCTION & OPTIMIZER
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),   # Collect ALL Params. of R-GCN
    lr = 0.01             # Learning Rate
)


# TRAINING LOOP
print("\nTraining R-GCN ...")
for epoch in range(1, 201):

    # Step-1 : Model Train
    model.train()

    # Step-2 : clear old learning gradiants
    optimizer.zero_grad()

    # Step-3 : Forward Pass
    out = model(data.x, data.edge_index, data.edge_type)

    # Step-4 : Loss Calculation
    loss = loss_function(out[data.train_mask], data.y[data.train_mask])

    # Step-5 : Back Propagation
    loss.backward()

    # Update Weights
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")


# MODEL EVALUATION
model.eval()

pred = out.argmax(dim=1)

correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
accuracy = int(correct) / int(data.test_mask.sum())

print("\nAccuracy - ", accuracy)
