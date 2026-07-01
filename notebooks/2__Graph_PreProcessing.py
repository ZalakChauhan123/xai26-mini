#!/usr/bin/env python
# coding: utf-8
#
# 2__Graph_PreProcessing.py
# -----------------------------------------------------------------------------
# Turns the AIFB RDF graph into the tensors an R-GCN needs:
#   node_to_id / id_to_node, predicate_to_id / id_to_relation,
#   edge_index, edge_type and the label tensor y.
#
# Converted from notebooks/2__Graph_PreProcessing.ipynb.
# The same logic lives in src/feature_engineering.py, which is the importable
# library version used by the training / explainability stages.
# -----------------------------------------------------------------------------

from pathlib import Path

import rdflib as rdf
from rdflib import Graph
from rdflib import Literal
import torch
from collections import Counter

# ---------------------------------------------------------------------------
# Paths (resolved relative to the repo root, independent of the CWD)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "dataset" / "aifbfixed_complete.n3"

g = Graph()
g.parse(str(DATASET_PATH))

print("Graph loaded successfully!")


# ---------------------------------------------------------------------------
# NODES
# ---------------------------------------------------------------------------
all_nodes = set()

for s, p, o in g:
    # subject is a node
    all_nodes.add(s)
    # Only add Object if not literal
    if not isinstance(o, Literal):
        all_nodes.add(o)

print("Total unique nodes", len(all_nodes))

# Create Node-to-ID Mapping
node_to_id = {}

for idx, node in enumerate(all_nodes):
    node_to_id[node] = idx

# Create Reverse Mapping
id_to_node = {}

for node, idx in node_to_id.items():
    id_to_node[idx] = node

# Test the Mapping
print("\nSample Node IDs:\n")
for i, (node, idx) in enumerate(node_to_id.items()):
    print(idx, node)

    if i >= 10:
        break


# ---------------------------------------------------------------------------
# EDGES
# ---------------------------------------------------------------------------
predicate_to_id = {}

relation_id = 0

for s, p, o in g:
    # Skip Literals
    if isinstance(o, Literal):
        continue

    if p not in predicate_to_id:
        predicate_to_id[p] = relation_id
        relation_id += 1

print("Total Relation Types :- ", len(predicate_to_id))

# Reverse Mapping (needed by the explainability stages)
id_to_relation = {idx: rel for rel, idx in predicate_to_id.items()}

# RDF triple into Edge Ids
REMOVE_RELATIONS = [
    "http://swrc.ontoware.org/ontology#affiliation",
    "http://swrc.ontoware.org/ontology#employs"
]

edges = []
edge_types = []

for s, p, o in g:
    # Skip Literals
    if isinstance(o, Literal):
        continue

    # Remove Leakage Relations
    if(str(p) in REMOVE_RELATIONS):
        continue

    source = node_to_id[s]
    target = node_to_id[o]

    relation_id = predicate_to_id[p]

    edges.append((source, target))
    edge_types.append(relation_id)

print(edges[0:10])

# Seperate Edges
edge_ids = []
for s, p, o in g:
    if isinstance(o, Literal):
        continue
    source = node_to_id[s]
    target = node_to_id[o]
    edge_id = len(edge_ids)
    edge_ids.append((edge_id, source, target, p))

print(edge_ids[0:10])

# Edge Index
edge_index = torch.tensor(
    edges,
    dtype = torch.long
).t().contiguous()

print("Edge Index Shape : ", edge_index.shape)
print("edge_index : ", edge_index[:, :10])

# Convert Edge Type into Tensors
edge_type = torch.tensor(
    edge_types,
    dtype = torch.long
)

print("Edge Type Shape : ", edge_type.shape)


# ---------------------------------------------------------------------------
# LABELS
# ---------------------------------------------------------------------------
LABEL_PREDICATE = "http://swrc.ontoware.org/ontology#affiliation"

# Collect unique Labels
unique_labels = set()

for s, p, o in g:
    if str(p) == LABEL_PREDICATE:
        unique_labels.add(o)

print("Classes :- ", unique_labels)
print("Total Classes :- ", len(unique_labels))

# Create label IDs
label_to_id = {
    label: idx
    for idx, label in enumerate(unique_labels)
}
print("Label IDs :- ", label_to_id)

# Create Node Labels
labels = {}

for s, p, o in g:
    if str(p) == LABEL_PREDICATE:
        node_id = node_to_id[s]
        class_id = label_to_id[o]
        labels[node_id] = class_id

# Convert Labels to Tensors
num_nodes = len(node_to_id)

y = torch.full(
    (num_nodes,),
    -1,
    dtype = torch.long
)

for node_id, class_id in labels.items():
    y[node_id] = class_id

print("y shape :", y.shape)
print("Unique labels in y :", torch.unique(y))
