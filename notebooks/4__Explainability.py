#!/usr/bin/env python
# coding: utf-8
#
# 4__Explainability.py
# -----------------------------------------------------------------------------
# GNNExplainer-based explainability for the R-GCN model.
# Converted from notebooks/4__Explainability.ipynb.
#
# Produces (into outputs/plots/Model_Explainability/GNN_Explainer/):
#   - Single Node - Top Edges.png
#   - Multiple Nodes - Imp. RDF Relations.png
#
# and prints faithfulness / performance metrics for GNNExplainer to the console.
# -----------------------------------------------------------------------------

# Imports
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless-safe
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Paths + make the src library importable (model.sh also sets PYTHONPATH=src)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

GNN_DIR = ROOT / "outputs" / "plots" / "Model_Explainability" / "GNN_Explainer"
GNN_DIR.mkdir(parents=True, exist_ok=True)

# Custom Imports (these build & train the model and set up the explainer)
from model_explainability import (test_nodes, explaination_gnn, explainer)
from model_train import (model, data)
from feature_engineering import (predicate_to_id, edge_type)

test_node = test_nodes[0]
print("Test Node : ", test_node)

# Explore Explanations
print(explaination_gnn)
print(explaination_gnn.edge_mask.max())


# ---------------------------------------------------------------------------
# PLOT 1 : Single Node - Top Edges
# Most important RDF relations behind the explanation of a single test node.
# ---------------------------------------------------------------------------
top_edges = explaination_gnn.edge_mask.topk(10)
important_edge_indices = top_edges.indices

# Create reverse mapping for relations
id_to_relation = {v: k for k, v in predicate_to_id.items()}

important_relations = []
for idx in important_edge_indices:
    idx = idx.item()
    rel_id = edge_type[idx].item()
    relation = id_to_relation[rel_id]
    clean_relation = relation.split("#")[-1]
    important_relations.append(clean_relation)

relation_counts = Counter(important_relations)
print(relation_counts)

relations = list(relation_counts.keys())
counts = list(relation_counts.values())

plt.figure(figsize=(8, 5))
plt.bar(relations, counts)
plt.xlabel("Relation Type")
plt.ylabel("Importance Frequency")
plt.title("Most Important RDF Relations")
plt.tight_layout()
plt.savefig(GNN_DIR / "Single Node - Top Edges.png", bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# PLOT 2 : Multiple Nodes - Imp. RDF Relations
# Aggregate the top edges across all test nodes -> global feature importance.
# ---------------------------------------------------------------------------
all_relation_lists = []

for node in test_nodes:
    print("Explaining Node:", node)
    print("-------")

    explanation = explainer(
        x=data.x,
        edge_index=data.edge_index,
        index=int(node)
    )

    important_edges = explanation.edge_mask.topk(20).indices

    relations = []
    for idx in important_edges:
        idx = idx.item()
        rel_id = edge_type[idx].item()
        relation = id_to_relation[rel_id]
        clean_relation = relation.split("#")[-1]
        relations.append(clean_relation)
    all_relation_lists.append(relations)

# Combine All Relations
all_relations = []
for relation_list in all_relation_lists:
    all_relations.extend(relation_list)

relation_counter = Counter(all_relations)
print(relation_counter)

# Sort from least to most frequent (largest ends up at the top of a barh plot)
sorted_relation_counts = sorted(relation_counter.items(), key=lambda x: x[1], reverse=False)
relations = [item[0] for item in sorted_relation_counts]
counts = [item[1] for item in sorted_relation_counts]

plt.figure(figsize=(10, 6))
plt.barh(relations, counts, color='skyblue', edgecolor='black')
plt.xlabel("Aggregation Importance Frequency")
plt.ylabel("RDF Relation Predicate")
plt.title("Global Feature Importance via Local Attribute Aggregation")
plt.tight_layout()
plt.savefig(GNN_DIR / "Multiple Nodes - Imp. RDF Relations.png", bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# FAITHFULNESS BATCH CHECK
# Remove the top explanatory edges and measure the drop in confidence / flips.
# ---------------------------------------------------------------------------
# RESET EXPLAIN MODE FOR SAFETY
model.conv1.explain = False
model.conv2.explain = False
model.eval()

eval_nodes = test_nodes[:10]

confidence_drops = []
label_flips = 0
total_nodes_evaluated = 0

print(f"\nRunning GNNExplainer Faithfulness batch check...\n")

for node_id in eval_nodes:
    node_idx = int(node_id)

    try:
        # 1. GET ORIGINAL PREDICTION & CONFIDENCE
        with torch.no_grad():
            orig_out = model(data.x, data.edge_index, data.edge_type)

        orig_prediction = orig_out[node_idx]
        orig_probs = F.softmax(orig_prediction, dim=0)
        orig_class = torch.argmax(orig_probs).item()
        orig_confidence = orig_probs[orig_class].item()

        # 2. GENERATE GNNEXPLAINER ATTRIBUTIONS FOR THIS NODE
        explanation_gnn = explainer(x=data.x, edge_index=data.edge_index, index=node_idx)

        # Extract top 50 important edges
        important_edges = explanation_gnn.edge_mask.topk(50).indices

        # 3. CREATE GRAPH PERTURBATION MASK
        num_edges = data.edge_index.shape[1]
        keep_mask = torch.ones(num_edges, dtype=torch.bool)
        keep_mask[important_edges] = False

        new_edge_index = data.edge_index[:, keep_mask]
        new_edge_type = data.edge_type[keep_mask]

        # 4. RUN MODEL ON MODIFIED GRAPH STRUCTURE
        with torch.no_grad():
            modified_out = model(data.x, new_edge_index, new_edge_type)

        modified_prediction = modified_out[node_idx]
        modified_probs = F.softmax(modified_prediction, dim=0)

        # Track confidence drop for the original predicted class
        new_confidence = modified_probs[orig_class].item()
        conf_drop = orig_confidence - new_confidence
        confidence_drops.append(conf_drop)

        # Check if the overall prediction label flipped
        new_class = torch.argmax(modified_probs).item()
        if orig_class != new_class:
            label_flips += 1

        total_nodes_evaluated += 1

    except Exception as e:
        print(f"Skipping Node {node_idx} due to structural limits: {e}")

# COMPUTE FINAL AGGREGATIONS
if total_nodes_evaluated > 0:
    avg_conf_drop = sum(confidence_drops) / total_nodes_evaluated
    flip_rate = (label_flips / total_nodes_evaluated) * 100
else:
    avg_conf_drop = 0.0
    flip_rate = 0.0

print("=" * 50)
print(f"{'FINAL FAITHFULNESS SUMMARY METRICS':^50}")
print("=" * 50)
print(f"Total Nodes Evaluated                 : {total_nodes_evaluated}")
print(f"Average Confidence Drop (Delta Conf) : {avg_conf_drop:.4f} ({avg_conf_drop * 100:.1f}%)")
print(f"Prediction Label Flip Rate            : {flip_rate:.1f}%")
print("=" * 50)


# ---------------------------------------------------------------------------
# OVERALL PERFORMANCE REPORT  (timing + sparsity + faithfulness)
# ---------------------------------------------------------------------------
# RESET EXPLAIN MODE FOR SAFETY
model.conv1.explain = False
model.conv2.explain = False
model.eval()

eval_nodes = test_nodes[:10]

confidence_drops = []
sparsities = []
execution_times = []
label_flips = 0
total_nodes_evaluated = 0

print(f"\nCalculating Overall Performance Metrics for GNNExplainer over {len(eval_nodes)} nodes...\n")

for node_id in eval_nodes:
    node_idx = int(node_id)

    try:
        # 1. TIME GENERATION OF GNNEXPLAINER ATTRIBUTIONS
        start_time = time.perf_counter()
        explanation_gnn = explainer(x=data.x, edge_index=data.edge_index, index=node_idx)
        end_time = time.perf_counter()

        node_time = end_time - start_time
        execution_times.append(node_time)

        # 2. CALCULATE SUBGRAPH SPARSITY
        raw_mask = explanation_gnn.edge_mask
        total_neighborhood_edges = raw_mask.shape[0]

        k = 20

        if total_neighborhood_edges > k:
            node_sparsity = 1.0 - (k / total_neighborhood_edges)
        else:
            node_sparsity = 0.0
        sparsities.append(node_sparsity)

        # 3. GET ORIGINAL PREDICTION & CONFIDENCE
        with torch.no_grad():
            orig_out = model(data.x, data.edge_index, data.edge_type)
        orig_probs = F.softmax(orig_out[node_idx], dim=0)
        orig_class = torch.argmax(orig_probs).item()
        orig_confidence = orig_probs[orig_class].item()

        # 4. REMOVE IMPORTANT EDGES AND PERTURB GRAPH
        top_edges = raw_mask.topk(k).indices
        num_edges = data.edge_index.shape[1]
        keep_mask = torch.ones(num_edges, dtype=torch.bool)
        keep_mask[top_edges] = False

        new_edge_index = data.edge_index[:, keep_mask]
        new_edge_type = data.edge_type[keep_mask]

        # 5. RUN MODEL ON PERTURBED GRAPH
        with torch.no_grad():
            modified_out = model(data.x, new_edge_index, new_edge_type)
        mod_probs = F.softmax(modified_out[node_idx], dim=0)

        new_confidence = mod_probs[orig_class].item()
        conf_drop = orig_confidence - new_confidence
        confidence_drops.append(conf_drop)

        new_class = torch.argmax(mod_probs).item()
        if orig_class != new_class:
            label_flips += 1

        total_nodes_evaluated += 1

    except Exception as e:
        print(f"Skipping Node {node_idx} due to calculation error: {e}")

# COMPUTE FINAL SYSTEM METRICS
if total_nodes_evaluated > 0:
    avg_conf_drop = np.mean(confidence_drops)
    avg_sparsity = np.mean(sparsities)
    avg_time = np.mean(execution_times)
    flip_rate = (label_flips / total_nodes_evaluated) * 100
else:
    avg_conf_drop, avg_sparsity, avg_time, flip_rate = 0.0, 0.0, 0.0, 0.0

print("=" * 60)
print(f"{'GNNEXPLAINER OVERALL PERFORMANCE REPORT':^60}")
print("=" * 60)
print(f"Total Nodes Evaluated                   : {total_nodes_evaluated}")
print(f"Average Execution Time                  : {avg_time:.4f} seconds / node")
print(f"Average Sparsity (K={k})                 : {avg_sparsity * 100:.2f}%")
print(f"Average Confidence Drop (Delta Conf)    : {avg_conf_drop:.4f} ({avg_conf_drop * 100:.1f}%)")
print(f"Prediction Label Flip Rate              : {flip_rate:.1f}%")
print("=" * 60)

print("\n✅ GNNExplainer plots saved to:", GNN_DIR)
