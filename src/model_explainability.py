# Model Training Imports
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F

# Model Explainability Imports
from torch_geometric.nn import GCNConv
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer

# Other imports
from model_train import *

# TEST NODE
test_node = test_nodes[0]
print("Test Node : ", test_node)


# CREATE GCN MODEL
class GCN(torch.nn.Module):

    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        self.con1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        # Safeguard: ensure edge_weight is a valid Tensor before passing to GCNConv
        if not isinstance(edge_weight, torch.Tensor):
            edge_weight = None

        x = self.con1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x

# Create Model
gcn_model = GCN(
    num_features = data.x.shape[1],
    hidden_channels = 16,
    num_classes = num_classes
)

# Forward Pass
out = gcn_model(
    data.x,
    data.edge_index
)


# =====================================================================
# GCN Explainer Code (GNNExplainer)
# =====================================================================
explainer = Explainer(
    model = gcn_model,
    algorithm = GNNExplainer(epochs = 200),
    explanation_type = "model",
    node_mask_type = "attributes",
    edge_mask_type = "object",
    model_config = dict(
        mode = "multiclass_classification",
        task_level = "node",
        return_type = "raw"
    )
)

# Generate GCN Explanation
explaination_gnn = explainer(
    x = data.x,
    edge_index = data.edge_index,
    index = int(test_node)
)

print("\n=== GNNExplainer Results ===")
print(explaination_gnn)
print("Max Edge Mask Importance:", explaination_gnn.edge_mask.max().item())
