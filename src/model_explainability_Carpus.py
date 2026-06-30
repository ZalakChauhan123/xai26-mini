# Model Training Imports
import torch
import torch.nn.functional as F

# Model Explainability Imports
from torch_geometric.nn import GCNConv
from torch_geometric.explain.algorithm import CaptumExplainer

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


# =====================================================================
# EXPLAINER 2: CaptumExplainer (Integrated Gradients)
# =====================================================================
# 1. Initialize algorithm instance
captum_algo = CaptumExplainer(attribution_method="IntegratedGradients")

# 2. Setup the parent PyG Explainer container
explainer_captum = Explainer(
    model = gcn_model,
    algorithm = captum_algo,
    explanation_type = "model",
    node_mask_type = "attributes",
    edge_mask_type = "object",
    model_config = dict(
        mode = "multiclass_classification",
        task_level = "node",
        return_type = "raw"
    )
)

# # 3. FORCE INJECTION: Directly bind allow_unused=True into the initialized Captum core object attribute.
# # This bypasses PyG wrapper argument routing constraints completely.
# if hasattr(captum_algo, 'attribution_method_instance'):
#     captum_algo.attribution_method_instance.allow_unused = True

# 4. Run the explanation generation call cleanly without any experimental parameter args
explanation_captum = explainer_captum(
    x = data.x,
    edge_index = data.edge_index,
    index = int(test_node)
)

# print("\n=== Captum (Integrated Gradients) Results ===")
print(explanation_captum)
print("Captum Max Edge Mask Importance:", explanation_captum.edge_mask.max().item())
print("Captum Max Node/Feature Importance:", explanation_captum.node_mask.max().item())

# this is a nigga