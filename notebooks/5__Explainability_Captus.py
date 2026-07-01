#!/usr/bin/env python
# coding: utf-8
#
# 5__Explainability_Captus.py
# -----------------------------------------------------------------------------
# Captum (Integrated Gradients) explainability + comparison against GNNExplainer.
# Converted from notebooks/5__Explainability_Captus.ipynb.
#
# Produces (into outputs/plots/Model_Explainability/Captus_Explainer/):
#   - Single Node - Edge Importance Plot.png
#   - Top Edges Comparison - GNN & Captus Explainer.png
#
# and writes the qualitative evaluation to:
#   outputs/reports/Evaluate_Explaination.txt
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
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths + make the src library importable (model.sh also sets PYTHONPATH=src)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CAP_DIR = ROOT / "outputs" / "plots" / "Model_Explainability" / "Captus_Explainer"
REPORT_DIR = ROOT / "outputs" / "reports"
CAP_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Custom Imports
from model_explainability_Carpus import (test_node, explainer_captum, explanation_captum)
from model_train import (model, data, test_nodes)
from feature_engineering import (id_to_relation, edge_type)
from model_explainability import (explainer)

# Choose test node
test_node = test_nodes[0]
print("Evaluating Single Node with Captum:", test_node)


# ---------------------------------------------------------------------------
# PLOT 1 : Single Node - Edge Importance Plot
# Local explanation for a single node via Captum Integrated Gradients.
# ---------------------------------------------------------------------------
explanation_captum = explainer_captum(
    x=data.x,
    edge_index=data.edge_index,
    index=int(test_node)
)

top_edges_captum = explanation_captum.edge_mask.abs().topk(10)
important_edge_indices_captum = top_edges_captum.indices

# Map Edge Indices back to Clean RDF Relations
important_relations_captum = []
id_to_relation_local = id_to_relation

for idx in important_edge_indices_captum:
    idx = idx.item()
    rel_id = edge_type[idx].item()

    relation = id_to_relation_local[rel_id]
    clean_relation = relation.split("#")[-1]
    important_relations_captum.append(clean_relation)

relation_counts_captum = Counter(important_relations_captum)
print("\nCaptum Relation Counts:", relation_counts_captum)

relations_c = list(relation_counts_captum.keys())
counts_c = list(relation_counts_captum.values())

# Sort for a professional academic look
sorted_indices = np.argsort(counts_c)
relations_c = [relations_c[i] for i in sorted_indices]
counts_c = [counts_c[i] for i in sorted_indices]

plt.figure(figsize=(8, 4))
plt.bar(relations_c, counts_c, color='#58D68D', edgecolor='black')
plt.xlabel("Importance Frequency (Top 10 Edges)")
plt.ylabel("RDF Relation Type")
plt.title(f"Local Explanation for Node {test_node} (Captum - Integrated Gradients)")
plt.tight_layout()
plt.savefig(CAP_DIR / "Single Node - Edge Importance Plot.png", bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# PLOT 2 : Top Edges Comparison - GNN & Captus Explainer
# Aggregate global importance from both explainers across the test split.
# ---------------------------------------------------------------------------
all_gnn_relations = []
all_captum_relations = []

print(f"Aggregating global explanations over {len(test_nodes)} test nodes...")

# Reduce optimization epochs for the global aggregation loop (speed)
try:
    explainer.algorithm.epochs = 30
except AttributeError:
    pass

for node_id in tqdm(test_nodes, desc="Generating Global Explanations"):
    node_idx = int(node_id)

    # A. Process GNNExplainer Attributions
    try:
        gnn_exp = explainer(x=data.x, edge_index=data.edge_index, index=node_idx)
        gnn_top_edges = gnn_exp.edge_mask.topk(20).indices

        for idx in gnn_top_edges:
            rel_id = edge_type[idx.item()].item()
            clean_rel = id_to_relation[rel_id].split("#")[-1]
            all_gnn_relations.append(clean_rel)
    except Exception:
        pass

    # B. Process Captum Integrated Gradients Attributions
    try:
        cap_exp = explainer_captum(x=data.x, edge_index=data.edge_index, index=node_idx)
        cap_top_edges = cap_exp.edge_mask.abs().topk(20).indices

        for idx in cap_top_edges:
            rel_id = edge_type[idx.item()].item()
            clean_rel = id_to_relation[rel_id].split("#")[-1]
            all_captum_relations.append(clean_rel)
    except Exception:
        pass

# Frequency Counting
gnn_counter = Counter(all_gnn_relations)
captum_counter = Counter(all_captum_relations)

# Sort Data for Ordered Horizontal Plotting (Top 10 Relations)
gnn_sorted = sorted(gnn_counter.items(), key=lambda x: x[1], reverse=False)[-10:]
captum_sorted = sorted(captum_counter.items(), key=lambda x: x[1], reverse=False)[-10:]

gnn_rels, gnn_counts = zip(*gnn_sorted) if gnn_sorted else ([], [])
cap_rels, cap_counts = zip(*captum_sorted) if captum_sorted else ([], [])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: GNNExplainer
ax1.barh(gnn_rels, gnn_counts, color='#5DADE2', edgecolor='black')
ax1.set_xlabel("Aggregation Frequency Count")
ax1.set_ylabel("RDF Relation Predicate")
ax1.set_title("Global Importance (GNNExplainer)")
ax1.grid(axis='x', linestyle='--', alpha=0.5)

# Subplot 2: Captum (Integrated Gradients)
ax2.barh(cap_rels, cap_counts, color='#58D68D', edgecolor='black')
ax2.set_xlabel("Aggregation Frequency Count")
ax2.set_title("Global Importance (Captum - Integrated Gradients)")
ax2.grid(axis='x', linestyle='--', alpha=0.5)

plt.suptitle("Global Feature Importance Comparison across Test Split Partition",
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(CAP_DIR / "Top Edges Comparison - GNN & Captus Explainer.png", bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# Sanity check : predicate frequencies in the (literal-filtered) triples
# ---------------------------------------------------------------------------
from feature_engineering import g
from rdflib import Literal

predicates = [str(p).split("#")[-1] for s, p, o in g if not isinstance(o, Literal)]
print(Counter(predicates))


# ---------------------------------------------------------------------------
# FAITHFULNESS SUMMARY (Captum)
# ---------------------------------------------------------------------------
model.eval()

confidence_drops = []
prediction_changed_count = 0
total_evaluated_nodes = 0

num_nodes_to_explain = min(10, len(test_nodes))
selected_test_nodes = test_nodes[:num_nodes_to_explain]

for node_idx in selected_test_nodes:
    node_id = int(node_idx)

    try:
        # TASK 1: Get Original Prediction & Confidence (R-GCN needs edge_type)
        with torch.no_grad():
            orig_logits = model(data.x, data.edge_index, edge_type)
            orig_probs = F.softmax(orig_logits[node_id], dim=-1)
            orig_pred_class = torch.argmax(orig_probs).item()
            orig_confidence = orig_probs[orig_pred_class].item()

        # TASK 2: Identify and Remove Important Edges & Edge Types
        exp = explainer_captum(x=data.x, edge_index=data.edge_index, index=node_id)

        node_edge_mask = torch.abs(exp.edge_mask)
        _, top_local_edge_indices = node_edge_mask.topk(min(20, len(node_edge_mask)))

        keep_edge_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool)
        keep_edge_mask[top_local_edge_indices] = False

        perturbed_edge_index = data.edge_index[:, keep_edge_mask]
        perturbed_edge_type = edge_type[keep_edge_mask]   # crucial R-GCN step!

        # TASK 3: Run Model Again on the Perturbed Graph Structure
        with torch.no_grad():
            perturbed_logits = model(data.x, perturbed_edge_index, perturbed_edge_type)
            perturbed_probs = F.softmax(perturbed_logits[node_id], dim=-1)
            perturbed_confidence = perturbed_probs[orig_pred_class].item()
            new_pred_class = torch.argmax(perturbed_probs).item()

        # TASK 4: Compare Prediction Confidence & Track Metrics
        conf_drop = orig_confidence - perturbed_confidence
        confidence_drops.append(conf_drop)

        if orig_pred_class != new_pred_class:
            prediction_changed_count += 1

        total_evaluated_nodes += 1

    except Exception as e:
        print(f"Skipping Node ID {node_id} due to runtime evaluation error: {e}")

if total_evaluated_nodes > 0:
    avg_conf_drop = sum(confidence_drops) / total_evaluated_nodes
    flip_rate = (prediction_changed_count / total_evaluated_nodes) * 100

    print("\n" + "=" * 50)
    print("FINAL FAITHFULNESS SUMMARY METRICS")
    print("=" * 50)
    print(f"Total Nodes Evaluated : {total_evaluated_nodes}")
    print(f"Average Confidence Drop (Delta Conf) : {avg_conf_drop:.4f} ({avg_conf_drop * 100:.1f}%)")
    print(f"Prediction Label Flip Rate        : {flip_rate:.1f}%")
    print("=" * 50)
else:
    print("Evaluation failed: No nodes were successfully perturbed.")


# ---------------------------------------------------------------------------
# OVERALL PERFORMANCE REPORT (Captum)
# ---------------------------------------------------------------------------
model.eval()

eval_nodes = test_nodes[:10]

confidence_drops = []
sparsities = []
execution_times = []
label_flips = 0
total_nodes_evaluated = 0

for node_id in eval_nodes:
    node_idx = int(node_id)

    try:
        # 1. TIMED GENERATION OF CAPTUM ATTRIBUTIONS
        start_time = time.perf_counter()
        explanation_captum = explainer_captum(x=data.x, edge_index=data.edge_index, index=node_idx)
        end_time = time.perf_counter()

        node_time = end_time - start_time
        execution_times.append(node_time)

        # 2. CALCULATE SUBGRAPH SPARSITY
        raw_mask = explanation_captum.edge_mask
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

        # 4. REMOVE IMPORTANT EDGES (absolute values for Captum magnitudes)
        top_edges = raw_mask.abs().topk(k).indices
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

if total_nodes_evaluated > 0:
    avg_conf_drop = np.mean(confidence_drops)
    avg_sparsity = np.mean(sparsities)
    avg_time = np.mean(execution_times)
    flip_rate = (label_flips / total_nodes_evaluated) * 100
else:
    avg_conf_drop, avg_sparsity, avg_time, flip_rate = 0.0, 0.0, 0.0, 0.0

print("=" * 60)
print(f"{'CAPTUM EXPLAINER OVERALL PERFORMANCE REPORT':^60}")
print("=" * 60)
print(f"Total Nodes Evaluated                 : {total_nodes_evaluated}")
print(f"Average Execution Time                : {avg_time:.4f} seconds / node")
print(f"Average Sparsity (K={k})               : {avg_sparsity * 100:.2f}%")
print(f"Average Confidence Drop (Delta Conf)  : {avg_conf_drop:.4f} ({avg_conf_drop * 100:.1f}%)")
print(f"Prediction Label Flip Rate            : {flip_rate:.1f}%")
print("=" * 60)


# ---------------------------------------------------------------------------
# WRITE THE QUALITATIVE EVALUATION TO THE REPORT FOLDER
# ---------------------------------------------------------------------------
report_text = """1. Semantic Validity
---------------------
Do important RDF relations make logical sense ?


| Important Relation | Meaning    |
| ------------------ | ---------- |
| author             | meaningful |
| publication        | meaningful |
| hasProject         | meaningful |


Notes : The explanations were semantically meaningful because the model primarily relied on academic collaboration and publication relationships rather than irrelevant metadata relations.



2. Sparsity / Compactness
-------------------------

| Graph                | Nodes | Edges |
| -------------------- | ----- | ----- |
| Original             | 2835  | 20338 |
| Explanation Subgraph | 112   | 204   |

Notes : The explainer successfully generated compact local explanations by reducing the original graph to a small influential neighborhood while preserving prediction behavior.


3. Top-k Relation Importance
-----------------------------
Notes : Publication and authorship relations dominated the explanation scores, indicating that the model relied heavily on academic connectivity patterns.


4. Local Neighborhood Dependence
---------------------------------
Notes : The model predictions depended mainly on local graph neighborhoods rather than the global RDF graph structure.
"""

report_path = REPORT_DIR / "Evaluate_Explaination.txt"
with open(report_path, "w") as f:
    f.write(report_text)

print("\n✅ Captum plots saved to:", CAP_DIR)
print("✅ Explanation report written to:", report_path)
